from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.ACT.detr_vae import DETRVAE, build
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.ACT.backbone import Joiner
from diffusion_policy.model.ACT.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer

# %%
class ActionChunkTransformerPolicy(BaseImagePolicy):
    def __init__(self,
                 # DETR arch
                 joiner: Joiner,
                 transformer: Transformer,
                 trans_encoder_layer: TransformerEncoderLayer,
                 state_dim,
                 num_queries,
                 camera_names,
                 num_encoder_layers,
                 kl_weight,
                 temporal_agg,
                 shape_meta,
                 **kwargs):
        super().__init__()
        backbones = []
        joiner.num_channels = joiner[0].num_channels
        backbones.append(joiner)
        encoder_norm = nn.LayerNorm(trans_encoder_layer.d_model) if trans_encoder_layer.normalize_before else None
        transformerEncoder = TransformerEncoder(trans_encoder_layer, num_encoder_layers, encoder_norm)
        
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        model = DETRVAE(
            backbones=backbones,
            transformer=transformer,
            encoder=transformerEncoder,
            state_dim=state_dim,
            num_queries=num_queries,
            camera_names=camera_names,
        )

        self.model: DETRVAE
        self.model = model
        self.kl_weight = kl_weight
        self.num_queries = num_queries
        self.action_dim = action_dim
        self.temporal_agg = temporal_agg
        self.normalizer = LinearNormalizer()
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        
    def predict_action(self, obs_dict):
        env_state = None
        nobs = self.normalizer.normalize(obs_dict)
        if 'agentview_image' in nobs.keys():
            agent_view_img = nobs['agentview_image'][:, 0] # [bs, c, h, w]
            hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            image_data = torch.cat((agent_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
        elif 'sideview_image' in nobs.keys():
            side_view_img = nobs['sideview_image'][:, 0] # [bs, c, h, w]
            hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            image_data = torch.cat((side_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
        else:
            raise NotImplementedError

        robot0_eef_pos = nobs['robot0_eef_pos'][:, 0] # [bs, 3]
        robot0_eef_quat = nobs['robot0_eef_quat'][:, 0] # [bs, 4]
        robot0_gripper_qpos = nobs['robot0_gripper_qpos'][:, 0] # [bs, 2]
        qpos_data = torch.cat((robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos), dim=1) # [bs, 9]

        a_hat, _, (_, _) = self.model(qpos_data, image_data, env_state)
        # NOTE: unormalize
        result = {
            "action": a_hat,
            "action_pred": a_hat,
        }
        return result


    
    def compute_loss(self, batch):
        """
        In ACT, its __getitem__ gets    image_data [k, c h, w],
                                        qpos_data [14],
                                        action_data [episode_len, 14],
                                        is_pad [episode_len]
                                    do preprocessing at last
        """
        env_state = None
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']) # [bs, horizon, 7]

        # do preprocessing
        if 'agentview_image' in nobs.keys():
            agent_view_img = nobs['agentview_image'][:, 0] # [bs, c, h, w]
            hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            image_data = torch.cat((agent_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
        elif 'sideview_image' in nobs.keys():
            side_view_img = nobs['sideview_image'][:, 0] # [bs, c, h, w]
            hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            image_data = torch.cat((side_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
        else:
            raise NotImplementedError

        robot0_eef_pos = nobs['robot0_eef_pos'][:, 0] # [bs, 3]
        robot0_eef_quat = nobs['robot0_eef_quat'][:, 0] # [bs, 4]
        robot0_gripper_qpos = nobs['robot0_gripper_qpos'][:, 0] # [bs, 2]
        qpos_data = torch.cat((robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos), dim=1) # [bs, 9]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = 1

        is_pad = torch.zeros(nactions.shape[1]).bool()
        is_pad = is_pad.to(nactions.device)
        is_pad = torch.unsqueeze(is_pad, axis=0).repeat(batch_size, 1)

        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos_data, image_data, env_state, nactions, is_pad)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss_dict = dict()
        all_l1 = F.l1_loss(nactions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

        return loss_dict

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            weight_decay: float,
            learning_rate: float,
            lr_backbone: float
    ) -> torch.optim.Optimizer:
        
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=weight_decay)

        return optimizer
    
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld