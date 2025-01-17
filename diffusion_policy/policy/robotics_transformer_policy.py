from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms as T

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.RT1.robotic_transformer_pytorch import RT1, MaxViT
from diffusion_policy.model.RT1.action_tokenizer import RT1ActionTokenizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer

from einops import rearrange, repeat, reduce, pack, unpack
import math

class Robotics_Transformer_policy(BaseImagePolicy):
    def __init__(self,
                 shape_meta,
                 crop_shape,
                 camera_name,
                 # Robotics Transformer Configuration
                 vit: MaxViT,
                 action_bins: int=256,
                 depth: int=6,
                 heads: int=8,
                 dim_head: int=64,
                 token_learner_ff_mult: int=2,
                 token_learner_num_layers: int=2,
                 token_learner_num_output_tokens: int=2,
                 cond_drop_prob: float=0.2,
                 use_attn_conditioner: bool=False,
                 conditioner_kwargs: dict=dict()):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        model = RT1(vit=vit,
                    num_actions=action_dim,
                    action_bins=action_bins,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    token_learner_ff_mult=token_learner_ff_mult,
                    token_learner_num_layers=token_learner_num_layers,
                    token_learner_num_output_tokens=token_learner_num_output_tokens,
                    cond_drop_prob=cond_drop_prob,
                    use_attn_conditioner=use_attn_conditioner,
                    conditioner_kwargs=conditioner_kwargs)
        
        self.model: RT1
        self.model = model
        self.action_dim = action_dim
        self.normalizer = LinearNormalizer()
        self.camera_name = camera_name
        self.preprocess = T.Compose([T.Resize((crop_shape[0], crop_shape[1]))])
        self.tokenizer = RT1ActionTokenizer(action_bins)

    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        result: must inculde 'action' key
        """
        # narmalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = nobs[self.camera_name]
        bs, frames = nobs.shape[0], nobs.shape[1]

        nobs = rearrange(nobs, 'bs frames c h w -> (bs frames) c h w')
        nobs = self.preprocess(nobs) # (bs frames) c crop_h crop_w
        video = rearrange(nobs, '(bs frames) c crop_h crop_w -> bs c frames crop_h crop_w', bs=bs, frames=frames)

        device = self.device
        dtype = self.dtype
        
        instructions = None
        nact_pred_logits = self.model(video, instructions)
        nact_pred_logits = nact_pred_logits [:, -1]
        nact_pred_token = torch.argmax(nact_pred_logits, dim=-1)
        nact_pred = self.tokenizer.detokenize(nact_pred_token)
        action_pred = self.normalizer['action'].unnormalize(nact_pred)
        action_pred = action_pred.unsqueeze(1)

        result = {
            'action': action_pred,
            'action_pred': action_pred,
        }
        return result

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nobs = nobs[self.camera_name]   # extract the eye in hand camera [bs, horizon, h, w, c]
        bs, frames = nobs.shape[0], nobs.shape[1]

        nobs = rearrange(nobs, 'bs frames c h w -> (bs frames) c h w')
        nobs = self.preprocess(nobs) # (bs frames) c crop_h crop_w
        video = rearrange(nobs, '(bs frames) c crop_h crop_w -> bs c frames crop_h crop_w', bs=bs, frames=frames)

        act_target = self.normalizer['action'].normalize(batch['action'])
        act_token = self.tokenizer.tokenize(act_target)

        # RT-1 defalt frames/sequence length is 6, just images, no low_dim vectors. [b c f h w]
        instructions = None
        act_logits = self.model(video, instructions)

        return self.sequence_loss(act_logits=act_logits, act_target=act_token)

    def sequence_loss(self, act_logits, act_target):
        """
            Applies sparse cross entropy loss between logits and target labels
        """
        act_target = F.one_hot(act_target.to(dtype=torch.int64),
                  act_logits.shape[-1]).squeeze(2)
        loss = -act_target * F.log_softmax(act_logits)
        return torch.sum(torch.mean(loss, dim=-1))

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self,
                      weight_decay: float,
                      learning_rate: float,
                      betas: Tuple[float, float]) -> torch.optim.Optimizer:
        optim_group = self.model.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=betas)

        return optimizer
