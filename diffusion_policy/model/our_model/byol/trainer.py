from typing import Dict, Tuple
import hydra
import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pathlib
import sys
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent.parent)
sys.path.append(ROOT_DIR)

from byol import pretrain_model
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")), 
    config_name="train")
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    pretrain_our_model(cfg)

def pretrain_our_model(cfg: OmegaConf):
    model: pretrain_model = hydra.utils.instantiate(cfg.model)
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.dataset)
    assert (isinstance(dataset, BaseImageDataset)), 'loading dataset error'

    image_processing = T.Compose([T.Resize((cfg.task.crop_shape[0], cfg.task.crop_shape[1]))])

    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()
    optimizer = model.get_optimizer(**cfg.optimizer)

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    model.set_normalizer(normalizer=normalizer)

    device = torch.device(cfg.training.device)
    model.to(device)
    optimizer_to(optimizer, device)

    model.train()
    for local_epoch_idx in range(cfg.training.num_epochs):
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {local_epoch_idx}",
                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = model(batch)
                loss.backward()
                loss_cpu = loss.item()
                tepoch.set_postfix(loss=loss_cpu, refresh=False)

if __name__=='__main__':
    main()