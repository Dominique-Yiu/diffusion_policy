from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from byol_pytorch import BYOL


class VINNBoylImagePolicy(BaseImagePolicy):
    def __init__(self,
                 net: nn.Module,
                 image_size,
                 hidden_layer,
                 projection_size,
                 projection_hidden_size,
                 moving_average_decay,
                 use_momentum,
                 shape_meta,
                 horizon,
                 crop_shape,
                 top_k,
                 **kwargs):
        super().__init__()

        # parse action meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        self.model = BYOL(
            net=net,
            image_size = image_size,
            hidden_layer = hidden_layer,
            projection_size = projection_size,
            projection_hidden_size = projection_hidden_size,
            moving_average_decay = moving_average_decay,
            use_momentum = use_momentum
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.crop_shape = crop_shape
        self.preprocess = T.Compose([T.Resize(crop_shape)])
        self.top_k = top_k
        self.kwargs = kwargs


    """ INFERENCE """
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], dataset) -> Dict[str, torch.tensor]:
        try:
            nobs = self.normalizer.normalize(obs_dict)
            nobs = nobs['robot0_eye_in_hand_image']
            # nobs = self.preprocess(nobs)
        except Exception as e:
            import ipdb; ipdb.set_trace()
        
        device = self.device
        dtype = self.dtype

        B = nobs.shape[0]
        nactions = torch.zeros(B, self.action_dim)
        
        for idx, img in enumerate(nobs):
            action = torch.zeros(self.action_dim)
            top_k_weights = torch.zeros(self.top_k)
            dist_list = self.calculate_nearest_neighbors(img, dataset)
            for i in range(self.top_k):
                top_k_weights[i] = dist_list[i][0]
            top_k_weights = torch.softmax(-1 * top_k_weights, dim=0)
            for i in range(self.top_k):
                action = torch.add(top_k_weights[i] * dist_list[i][1], action)
            nactions[idx] = action
        
        nactions = nactions.unsqueeze(1)
        nactions = nactions.to(device)
        result = {
            'action': nactions,
            'action_pred': nactions
        }
        return result
        

    def calculate_nearest_neighbors(self, query_img, dataset):
        dist_list = []

        query_embedding = self.model(query_img, return_embedding=True, return_projection=False)
        dataset_length = len(dataset)
        query_embedding = query_embedding.to('cpu')
        for dataset_index in range(dataset_length):
            dataset_embedding, dataset_translation = dataset[dataset_index]
            distance = torch.norm(query_embedding - dataset_embedding).item()
            dist_list.append((distance, dataset_translation))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])

        return dist_list[:self.top_k]

        
    
    """ TRAINING """
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self,
                      learning_rate: float,
                      betas: Tuple[float, float]):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=learning_rate,
                                     betas=betas)
        
        return optimizer
    
    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        # just need 'robot0_eye_in_hand_image'
        assert 'robot0_eye_in_hand_image' in nobs.keys()
        nobs = nobs['robot0_eye_in_hand_image'].squeeze(1)
        # nobs = self.preprocess(nobs)

        loss = self.model(nobs)

        return loss