import pathlib
import sys
import ipdb
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent.parent)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import repeat, pack, unpack, reduce, rearrange

from typing import Optional, List, Tuple
from byol_pytorch import BYOL
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

class VisualEncoder(nn.Module):
    def __init__(self,
                 n_obs_steps: int,
                 image_size: int,
                 hidden_layer: Optional[int],
                 projection_size: int,
                 projection_hidden_size: int,
                 moving_average_decay: float,
                 use_momentum: bool,
                 pretrain: bool=True) -> None:
        super().__init__()
        
        resnet = models.resnet50(weights="IMAGENET1K_V1") if pretrain else models.resnet50(weights=None)

        self.normalizer = LinearNormalizer()
        self.model = BYOL(
            net = resnet,
            image_size = image_size,
            hidden_layer = hidden_layer,
            projection_size = projection_size,
            projection_hidden_size = projection_hidden_size,
            moving_average_decay = moving_average_decay,
            use_momentum = use_momentum
        )
        self.n_obs_steps = n_obs_steps
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

    def forward(self, batch,
                obs_config,
                ret_loss: bool=True,
                return_embedding = True,
                return_projection = True):
        if isinstance(batch, dict):
            nobs = self.normalizer.normalize(batch['obs'])
            if not ret_loss:
                nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...])

            value = next(iter(nobs.values()))
            B, T = value.shape[:2]
            N = len(obs_config['rgb'])
            # concat all rgb data in channel
            cam_images = list()
            for item in obs_config['rgb']:
                cam_images.append(nobs[item])
            cam_images, _ = pack(cam_images, 'batch T * c h w')
            cam_images = rearrange(cam_images, 'batch T N c h w -> (batch T N) c h w')

            if ret_loss:
                return self.model(cam_images)

            # return_embedding and return_projection must be true
            result, _ = self.model(cam_images, 
                            return_embedding=return_embedding, 
                            return_projection=return_projection)
            result = rearrange(result, '(batch T N) dim -> batch (T N dim)', T=T, N=N)
            return result
        elif isinstance(batch, torch.Tensor):
            B, T = batch.shape[:2]
            N = len(obs_config['rgb'])
            batch = rearrange(batch, 'batch T N c h w -> (batch T N) c h w')

            if ret_loss:
                return self.model(cam_images)

            # return_embedding and return_projection must be true
            result, _ = self.model(batch, 
                            return_embedding=return_embedding, 
                            return_projection=return_projection)
            result = rearrange(result, '(batch T N) dim -> batch (T N dim)', T=T, N=N)
            return result
        else:
            raise RuntimeError(f'Unsupported data type {type(batch)}.')

if __name__=='__main__':

    ViE = VisualEncoder(
        image_size=76,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        use_momentum=True,
        pretrain=True,
    )
    ViE_path = 'diffusion_policy/model/our_model/byol/state_dict/latest_epoch90.pt'
    VIE_PARAMS = torch.load(ViE_path, map_location='cuda:0')
    ViE.load_state_dict(VIE_PARAMS['model_state_dict'])