from typing import Dict, Tuple
import hydra
import tqdm
import copy
import wandb
import random
import os
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

from byol import VisualEncoder
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")), 
    config_name="train")
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    workspace = pretrain_our_model(cfg)
    workspace.run()

class pretrain_our_model(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir = None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure policy
        self.model: VisualEncoder = hydra.utils.instantiate(cfg.model)
        
        # configrue training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.dataset)
        assert (isinstance(dataset, BaseImageDataset)), 'loading dataset error'

        image_processing = T.Compose([T.Resize((cfg.task.crop_shape[0], cfg.task.crop_shape[1]))])

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer=normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            name = cfg.training.lr_scheduler,
            optimizer = self.optimizer,
            num_warmup_steps = cfg.training.lr_warmup_steps,
            num_training_steps = (
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,

            last_epoch = self.global_step - 1
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk,
        )


        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        obs_shape_meta = cfg.shape_meta['obs']
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

        wandb_run = wandb.init(
            dir=str(pathlib.Path(__file__).parent),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": str(pathlib.Path(__file__).parent)})

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                train_losses = list()

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # compute loss
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        raw_loss = self.model(batch, obs_config)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        loss_cpu = loss.item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        train_losses.append(loss_cpu)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))

                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                self.model.eval()
                if self.epoch % cfg.training.val_every == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {local_epoch_idx}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_loss = self.model(batch, obs_config)

                                val_losses.append(val_loss)

                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

               # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
            

if __name__=='__main__':
    main()