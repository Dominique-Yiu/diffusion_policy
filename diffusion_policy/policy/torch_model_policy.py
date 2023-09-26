import ipdb
import tqdm
from omegaconf import OmegaConf
from typing import Optional, Dict, List, Tuple
from einops import repeat, reduce, rearrange, pack, unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.loss_lib import kl_divergence
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.our_model.byol.byol import VisualEncoder
from diffusion_policy.model.our_model.discretizer.k_means import KMeansDiscretizer
from diffusion_policy.model.our_model.torch_model import TransformerForOurs


class Torch_Model_Policy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 # discretizer
                 discretizer: KMeansDiscretizer,
                 kmeans_class: int,
                 # task params
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 # image
                 crop_shape: [int, int],
                 # visual encoder
                 ViE: VisualEncoder,
                 ViE_path: str,
                 # model
                 fea_dim: int,
                 latent_dim: int,
                 n_cvae_layer: int,
                 n_cond_layer: int,
                 n_layer: int,
                 n_head: int,
                 n_emb: int,
                 p_drop_emb: float,
                 p_drop_attn: float,
                 causal_attn: bool,
                 # others
                 kl_weights: float,
                 temporal_agg: bool,
                 **kwargs,):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = shape

            obs_type = attr.get('type', 'low_dim')

            if obs_type in obs_config.keys():
                obs_config[obs_type].append(key)
            else:
                raise RuntimeError(f'Unsupported obs type {obs_type}.')
        ## calculate state dim
        state_dim = 0
        for item in obs_config['low_dim']:
            low_dim_shape = obs_shape_meta[item]['shape']
            assert len(low_dim_shape) == 1
            state_dim += low_dim_shape[0]
        
        # ViE
        self.VisEncoder: VisualEncoder = ViE
        VIE_PARAMS = torch.load(ViE_path, map_location=self.device)
        self.VisEncoder.load_state_dict(VIE_PARAMS['model_state_dict'])
        
        # Discretizer
        self.discretizer: KMeansDiscretizer = discretizer

        # Model
        self.model: TransformerForOurs = TransformerForOurs(
            kmeans_class = kmeans_class,
            action_dim = action_dim,
            state_dim = state_dim,
            fea_dim = fea_dim,
            latent_dim = latent_dim,
            output_dim = action_dim,
            horizon = horizon,
            n_obs_steps = n_obs_steps,
            n_cvae_layer = n_cvae_layer,
            n_cond_layer = n_cond_layer,
            n_layer = n_layer,
            n_head = n_head,
            n_emb = n_emb,
            p_drop_emb = p_drop_emb,
            p_drop_attn = p_drop_attn,
            causal_attn = causal_attn,
        )

        # Crucial constants
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.kl_weights = kl_weights
        self.temporal_agg = temporal_agg
        self.obs_config = obs_config

        self.preprocess = T.Compose([T.Resize((crop_shape[0], crop_shape[1]))])
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            nobs = self.normalizer.normalize(obs_dict)
        except Exception as e:
            ipdb.set_trace()

        device = self.device
        dtype = self.dtype

        T = self.horizon
        To = self.n_obs_steps
        Ta = self.n_action_steps
        value = next(iter(nobs.values()))
        B = value.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...])
        state_obs, image_obs = self.seperate_batch(this_nobs=this_nobs)

        state_obs, _ = pack(state_obs, 'bs To *') # bs To dim'
        image_obs, _ = pack(image_obs, 'bs To * h w') # bs To c' h w
        
        # 1. obtain image features
        image_obs = rearrange(image_obs, 'batch To c h w -> batch (To c) h w')

        image_fea, _ = self.VisEncoder(
            image_obs,
            return_embedding = True,
            return_projection=True,
        ) # batch fea_dim

        # 2. obtain class for per batch
        n_clusters = self.discretizer.encode_into_latent(image_fea) # batch,

        # 3. forward
        pred_nactions, (_, _) = self.model(
            joint_states = state_obs,
            cond = image_fea,
            n_class = n_clusters,
            action = None,
        ) # batch T action_dim

        # 4. unnormalize actions
        action_pred = self.normalizer['action'].unnormalize(pred_nactions)

        # 5. get action
        start = To - 1
        end = start + Ta
        action = action_pred[:, start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result
    
    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        T = self.horizon
        To = self.n_obs_steps
        Ta = self.n_action_steps
        bs = nactions.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...])
        # seperate low_dim and rgb obs
        state_obs, image_obs = self.seperate_batch(this_nobs=this_nobs)

        state_obs, _ = pack(state_obs, 'bs To *') # bs To dim'
        image_obs, _ = pack(image_obs, 'bs To * h w') # bs To c' h w
        
        # 1. obtain image features
        image_obs = rearrange(image_obs, 'batch To c h w -> batch (To c) h w')

        image_fea, _ = self.VisEncoder(
            image_obs,
            return_embedding = True,
            return_projection=True,
        ) # batch fea_dim

        # 2. obtain class for per batch
        n_clusters = self.discretizer.encode_into_latent(image_fea) # batch,

        # 3. forward
        pred_nactions, (mu, logvar) = self.model(
            joint_states = state_obs,
            cond = image_fea,
            n_class = n_clusters,
            action = nactions,
        )
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        L1_loss = F.l1_loss(nactions, pred_nactions, reduction='none').mean()

        LOSS = total_kld[0] * self.kl_weights + L1_loss
        return LOSS

    def seperate_batch(self, this_nobs):
        state_obs = list()
        image_obs = list()
        for item in self.obs_config['low_dim']:
            state_obs.append(this_nobs[item]) # bs To dim
        for item in self.obs_config['rgb']:
            image_obs.append(this_nobs[item]) # bs To c h w
        
        return state_obs, image_obs

    
    def get_optimizer(self,
                      weight_decay: float,
                      learning_rate: float,
                      betas: Tuple[float, float]) -> torch.optim.Optimizer:
        optim_group = self.model.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=betas)

        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def fit_discretizer(self, data: DataLoader):
        input_features = list()

        with torch.no_grad():
            with tqdm.tqdm(data, desc=f'Extract image representation: ',
                           leave=False, mininterval=1.0) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    projection, _ = self.VisEncoder(
                        batch = batch, 
                        obs_config = self.obs_config, 
                        return_embedding = True, 
                        return_projection = True,
                    ) # batch fea_dim
                    input_features.append(projection)

                input_features, _ = pack(input_features, '* fea_dim')

                self.discretizer.fit_discretizer(input_features=input_features)

