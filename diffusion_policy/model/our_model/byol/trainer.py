from typing import Dict, Tuple
import hydra
import tqdm
import wandb
import numpy as np
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

    wandb_run = wandb.init(
        dir=str(pathlib.Path(__file__).parent),
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging
    )
    wandb.config.update({"output_dir": str(pathlib.Path(__file__).parent)})
    
    GLOBAL_STEP = 0
    for local_epoch_idx in range(cfg.training.num_epochs):
        model.train()
        train_losses = list()
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {local_epoch_idx}",
                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # compute loss
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                raw_loss = model(batch)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()
                loss_cpu = loss.item()
                tepoch.set_postfix(loss=loss_cpu, refresh=False)

                # step optimizer
                if GLOBAL_STEP % cfg.training.gradient_accumulate_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # logging
                step_log = {
                    "train_loss": loss_cpu,
                    "global_step": GLOBAL_STEP,
                    "epoch": local_epoch_idx,
                }
                train_losses.append(loss_cpu)
                wandb_run.log(step_log, step=GLOBAL_STEP)

                GLOBAL_STEP += 1
        train_loss = np.mean(train_losses)
        step_log['avg_train_loss_per_epoch'] = train_loss

        model.eval()
        if local_epoch_idx % cfg.training.val_every == 0:
            with torch.no_grad():
                val_losses = list()
                with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {local_epoch_idx}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        val_loss = model(batch)

                        val_losses.append(val_loss)

                if len(val_losses) > 0:
                    val_loss = torch.mean(torch.tensor(val_losses)).item()
                    step_log['avg_val_loss_per_epoch'] = val_loss

        # checkpoint
        if local_epoch_idx % cfg.training.checkpoint_every == 0:
            checkpoint_dir = pathlib.Path(__file__).parent.joinpath("state_dict")
            
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir.joinpath(f"latest_epoch{local_epoch_idx}.pt")

            torch.save(
                {
                    'epoch': local_epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                checkpoint_path,
            )

        wandb_run.log(step_log, step=GLOBAL_STEP)
        GLOBAL_STEP += 1
        

if __name__=='__main__':
    main()