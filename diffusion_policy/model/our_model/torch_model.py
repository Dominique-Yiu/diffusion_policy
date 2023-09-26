from typing import List, Tuple, Optional
from einops import reduce, pack, unpack, rearrange, repeat
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from diffusion_policy.model.our_model.byol.byol import pretrain_model
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.our_model.discretizer.k_means import KMeansDiscretizer

logger = logging.getLogger(__name__)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class TransformerForOurs(ModuleAttrMixin):
    def __init__(self,
                 kmeans_class: int,
                 action_dim: int,
                 state_dim: int,
                 fea_dim: int,
                 latent_dim: int, # 32
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int=None,
                 n_cvae_layer: int=8,
                 n_cond_layer: int=8,
                 n_layer: int=8,
                 n_head: int=8,
                 n_emb: int=256,
                 p_drop_emb: float=0.1,
                 p_drop_attn: float=0.1,
                 causal_attn: bool=False):
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = n_obs_steps + 1 + T

        # CVAE encoder configuration
        self.action_emb = nn.Linear(action_dim, n_emb)
        self.state_emb = nn.Linear(state_dim, n_emb)
        self.cluster_emb = SinusoidalPosEmb(n_emb)
        self.latent_emb = nn.Linear(n_emb, latent_dim * 2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(T_cond, n_emb))

        # Transformer encoder configuration
        self.cond_emb = nn.Linear(fea_dim, n_emb)
        self.latent_out_emb = nn.Linear(latent_dim, n_emb)
        self.trans_state_emb = nn.Linear(state_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # Transformer decoder configuration
        self.query_emb = nn.Embedding(T, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # CVAE encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        cvae_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_cvae_layer,
        )
        self.cvae_encoder = [cvae_encoder for _ in range(kmeans_class)] # FIXME
        # Transformer encoder
        self.trans_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_cond_layer,
        )
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.trans_decoder: nn.TransformerDecoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # attention mask FIXME
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            S = T_cond
            t, s = torch.meshgrid(
                torch.arange(T),
                torch.arange(S),
                indexing='ij'
            )
            mask = t >= (s-1) # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('memory_mask', mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constant
        self.T = T
        self.T_cond = T_cond
        self.horizon = self.horizon
        self.kmeans_class = kmeans_class
        self.latent_dim = latent_dim

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_optim_groups(self, weight_decay: float=1e-3):
        decay = list()
        no_decay = list()
        parameters_to_decay = ('weight', 'bias')
        for name, param in self.named_parameters():
            if any(param_name in name for param_name in parameters_to_decay) and param.requires_grad:
                decay.append(param)
            else:
                no_decay.append(param)
        
        optim_group = [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            }
        ]

        return optim_group


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self,
                joint_states: torch.Tensor,
                cond: torch.Tensor,
                n_class: int,
                action: torch.Tensor=None):
        """
        joint_states: low dim states (batch, To, joint_dim,)
        cond: image features extracted by resnet (batch, To, fea_dim)
        n_class: class info obtained by k-means discretizer (batch,)
        action: action info (batch, T, act_dim)
        """

        device = self.device
        dtype = self.dtype

        is_training = action is not None
        batch = joint_states.shape[0]

        if is_training:
            # CVAE encoder
            # 2.1 encode class, action, and joints -> pack
            joint_emb = self.state_emb(joint_states) # batch, To, joint_dim -> batch, To, n_emb
            class_emb = self.cluster_emb(n_class).unsqueeze(1) # batch, -> batch 1 n_emb
            action_emb = self.action_emb(action) # batch, T, act_dim -> batch, T, n_emb
            cvae_emb, _ = pack([class_emb, joint_emb, action_emb], 'bs * n_emb') # batch, 1+To+T(T_cond), n_emb

            # 2.2 add position embedding from pos_table NOTE
            pos_embed = self.pos_table.clone().detach() # 1, T_cond, n_emb

            cvae_input = self.drop(cvae_emb + pos_embed) # batch, T_cond, n_emb

            # 2.3 cvae encoder
            # [(1, n_emb) * len(n_class)]
            # 2.4 take out the first element [0]
            cvae_output = [self.cvae_encoder[cluster](cvae_input[idx].unsqueeze(0))[:,0] for idx, cluster in enumerate(n_class)]
            # 2.5 project into 64 dim, first 32 is named mu, the last 32 dim is named logvar
            # [(1, latent_dim * 2) * len(n_class)]
            cvae_output = [self.latent_emb(item) for item in cvae_output]
            cvae_output = torch.cat(cvae_output, dim=0) # batch, latent_dim * 2
            mu = cvae_output[:, :self.latent_dim]
            logvar = cvae_output[:, self.latent_dim:]

            # 2.6 reparametrize mu and logvar to produce latent code 
            latent_sample = reparametrize(mu, logvar).unsqueeze(1) # batch, 1, latent_dim

            # 2.7 project latent code
            latent_code = self.latent_out_emb(latent_sample) # batch, 1, n_emb
        else:
            mu = logvar = None
            latent_sample = torch.zeros([batch, self.latent_dim], dtype=dtype).to(device)
            latent_code = self.latent_out_emb(latent_sample)
        # CVAE decoder
        # transformer encoder
        cond_embeddings = self.cond_emb(cond) # batch, To, fea_dim -> batch, To, n_emb
        proprio_input = self.trans_state_emb(joint_states) # batch, To, joint_dim -> batch, To, n_emb

        cond_embeddings, _ = pack([latent_code, proprio_input, cond_embeddings], 'bs * n_emb') # batch, 1+To+T, n_emb
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :] # each position maps to a (learnable) vector

        trans_encoder_ipt = self.drop(cond_embeddings + position_embeddings)
        trans_encoder_opt = self.trans_encoder(trans_encoder_ipt) # batch, 1+To+T / T_cond, n_emb FIXME
        memory = trans_encoder_opt

        # transformer decoder
        query_embeddings = self.query_emb.weight # (T, n_emb)
        query_embeddings = repeat(query_embeddings, 'T n_emb -> r T n_emb', r=batch) # (batch T n_emb)

        token_embeddings = torch.zeros_like(query_embeddings, dtype=dtype).to(device)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        trans_decoder_ipt = self.drop(token_embeddings + position_embeddings)
        trans_decoder_opt = self.trans_decoder(
            tgt = trans_decoder_ipt,
            memory = memory,
            tgt_mask = self.mask,
            memory_mask = self.memory_mask,
        ) # batch T n_emb

        # head
        pred_nactions = self.head(self.ln_f(trans_decoder_opt))

        return pred_nactions, [mu, logvar]