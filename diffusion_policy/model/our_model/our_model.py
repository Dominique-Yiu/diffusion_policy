from typing import List, Tuple, Optional
from einops import reduce, pack, unpack, rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from diffusion_policy.model.our_model.byol.byol import pretrain_model
from diffusion_policy.model.our_model.discretizer.k_means import KMeansDiscretizer
from diffusion_policy.model.ACT.transformer import Transformer, TransformerEncoder
from diffusion_policy.model.ACT.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine

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

class our_model(nn.Module):
    def __init__(self,
                 backbones: pretrain_model,
                 transformer: Transformer,
                 encoder: TransformerEncoder,
                 byol_channels: int,
                 kmeans_class: int,
                 action_dim: int,
                 state_dim: int,
                 center_point_dim: int,
                 num_queries: int,
                 camera_names: List[str],
                 kmeans_discretizer: KMeansDiscretizer) -> None:
        """ Initializes the mpdel.
        Parameters:
            backbones: torch module of the backbone to be used, ResNet.
            transformer: torch module of the transformer architecture.
            encoder: encoder of CVAE.
            kmeans_class: number of classes while doing k-means
            action_dim: robot action dimension of the environment.
            state_dim: robot joint state dimension of the environment.
            center_point_dim: k-means centers dimension
            num_queries: number of object queries, i.e. the number
                predicted action.
            camera_names: list of cameras whose images will be used as conditions.
        """
        super().__init__()
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = [encoder for _ in range(kmeans_class)]
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.kmeans_discretizer: KMeansDiscretizer = kmeans_discretizer

        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(byol_channels, hidden_dim, kernel_size=1)
        self.backbones = backbones
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent code z
        self.encoder_center_proj = nn.Linear(center_point_dim, hidden_dim) # project obs image class to embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim) # project joints to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2) # project hideden state to latend std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

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

    def forward(self,
                joint_states: torch.Tensor,
                images: torch.Tensor,
                action: torch.Tensor=None,
                is_pad: torch.Tensor=None):
        """
        joint_states: batch, joint_dim
        image: batch, num_cam, channel, height, width
        action: batch, horizon, action_dim
        """
        device = images.device
        is_training = action is not None
        bs = joint_states.shape[0]

        """
            NOTE: fit multi images input, now only support one image
        """
        # all_cam_features = []
        # all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            query_embedding = self.backbones(images[:, cam_id], return_embedding=True, return_projection=False)

            k_class = self.kmeans_discretizer.encode_into_latent(query_embedding)

            pos = self.position_embedding(query_embedding)
            src = self.input_proj(query_embedding)
            # all_cam_features.append(self.input_proj(query_embedding))
            # all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(joint_states)
        # fold camera dimension into width dimension
        # src = torch.cat(all_cam_features, axis=3)
        # pos = torch.cat(all_cam_pos, axis=3)

        if is_training:
            center_embed = self.encoder_center_proj(k_class) 
            center_embed = rearrange(center_embed, 'bs center_dim -> bs 1 center_dim') # (bs, 1, hidden_dim)
            action_embed = self.encoder_action_proj(action) # (bs, horizon, hidden_dim)
            joint_embed = self.encoder_joint_proj(joint_states)
            joint_embed = rearrange(joint_embed, 'bs joint_dim -> bs 1 joint_dim') # (bs, 1, hidden_dim)

            encoder_input, _ = pack([center_embed, joint_embed, action_embed], 'bs * hidden_dim')
            encoder_input = rearrange(encoder_input, 'bs seq hidden_dim -> seq bs hidden_dim')

            center_joint_is_pad = torch.full((bs, 2), False).to(device)
            is_pad, _ = pack([center_joint_is_pad, is_pad], 'bs *')

            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = rearrange(pos_embed, 'l seq hidden_dim -> seq l hidden_dim')

            # query model
            encoder_output = self.encoder[k_class](encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(device)
            latent_input = self.latent_out_proj(latent_sample)

        hs = self.transformer(src=src, 
                              mask=None, 
                              query_embed=self.query_embed.weight, 
                              pos_embed=pos, 
                              latent_input=latent_input, 
                              proprio_input=proprio_input, 
                              additional_pos_embed=self.additional_pos_embed.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]



