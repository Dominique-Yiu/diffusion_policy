from typing import Optional, List, Tuple
from omegaconf import OmegaConf
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.our_model.our_model import our_model
from diffusion_policy.model.our_model.byol.byol import VisualEncoder
from diffusion_policy.model.our_model.discretizer.k_means import KMeansDiscretizer
from diffusion_policy.model.ACT.backbone import Joiner
from diffusion_policy.model.ACT.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer

class OurPolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 discretizer: KMeansDiscretizer,
                 # our model
                 byol: VisualEncoder,
                 byol_path: str,
                 byol_channels: int,
                 transformer: Transformer,
                 trans_encoder_layer: TransformerEncoderLayer,
                 num_encoder_layers: int,
                 center_point_dim: int,
                 num_queries: int,
                 camera_names: List[str],
                 # others
                 kl_weights,
                 temporal_agg,
                 state_dim,
                 **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        backbones = []
        # load pretrain byol model
        checkpoint = torch.load(byol_path, map_location='cuda:0') # FIXME
        byol.load_state_dict(checkpoint['model_state_dict'])
        backbones.append(byol)

        encoder_norm = nn.LayerNorm(trans_encoder_layer.d_model)\
              if trans_encoder_layer.normalize_before else None
        transformerEncoder = TransformerEncoder(trans_encoder_layer, num_encoder_layers, encoder_norm)
        


        self.num_clusters = discretizer.discretized_space
        self.model: our_model = our_model(
            backbones = backbones,
            transformer = transformer,
            encoder = transformerEncoder,
            byol_channels = byol_channels,
            kmeans_class = self.num_clusters,
            action_dim = action_dim,
            state_dim = state_dim,
            center_point_dim = center_point_dim,
            num_queries = num_queries,
            camera_names = camera_names,
            kmeans_discretizer = discretizer
        )
        
        self.normalizer = LinearNormalizer()
        self.kl_weights = kl_weights
        self.num_queries = num_queries
        self.action_dim = action_dim
        self.temporal_agg = temporal_agg
        self.query_frequency = 1 if self.temporal_agg else self.num_queries

    def predict_action(self):
        raise NotImplementedError

    def compute_loss(self, batch):
        nobs = self.normalizer(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # do preprocessing
        if 'agentview_image' in nobs.keys():
            agent_view_img = nobs['agentview_image'][:, 0] # [bs, c, h, w]
            image_data = agent_view_img.unsqueeze(1)
            # hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            # image_data = torch.cat((agent_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
        elif 'sideview_image' in nobs.keys():
            side_view_img = nobs['sideview_image'][:, 0] # [bs, c, h, w]
            image_data = side_view_img.unsqueeze(1)
            # hand_img = nobs['robot0_eye_in_hand_image'][:, 0] # [bs, c, h ,w]
            # image_data = torch.cat((side_view_img.unsqueeze(1), hand_img.unsqueeze(1)), dim=1) # [bs, 2, c, h ,w]
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

        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos_data, image_data, nactions, is_pad)
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
    
    def get_optimizer(self,
                      weight_decay: float,
                      learning_rate: float,
                      betas: Tuple[float, float]) -> torch.optim.Optimizer:
        optim_group = self.model.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=betas)

        return optimizer
    
    def fit_discretizer(self, input_pic):
        input_features = list()

        with torch.no_grad():
            with tqdm.tqdm(input_pic, desc=f"Extract image representation: ",
                        leave=False, mininterval=1.0) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    projection, representation = self.model.backbones[0](batch, return_embedding=True, return_projection=True)
                    input_features.append(projection)

        input_features = torch.cat(input_features, axis=0)
        self.model.kmeans_discretizer.fit_discretizer(input_features=input_features)

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