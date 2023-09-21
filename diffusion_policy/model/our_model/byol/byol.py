import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import Optional, List, Tuple
from byol_pytorch import BYOL
from diffusion_policy.model.common.normalizer import LinearNormalizer

class pretrain_model(nn.Module):
    def __init__(self,
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

    def forward(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        # just need 'robot0_eye_in_hand_image'
        assert 'robot0_eye_in_hand_image' in nobs.keys()
        nobs = nobs['robot0_eye_in_hand_image'].squeeze(1)
        return self.model(nobs)
