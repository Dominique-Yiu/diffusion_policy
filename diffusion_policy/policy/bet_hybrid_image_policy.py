from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

class BETHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # discretizer
            action_ae: KMeansDiscretizer, 
            # task params
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            obs_as_cond=True,
            n_layer=4,
            n_head=4,
            n_embed=72,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            vocab_size=24,
            predict_offsets=True,
            offset_loss_scale=1000.0,
            focal_loss_gamma=2.0,
            discrete_input=False,
            **kwargs):
        super().__init__()

        #  parse shape_meta
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
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type {key}.")
        
        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        obs_encoding_net = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoding_net,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoding_net,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoding_net.output_shape()[0]
        input_dim = obs_feature_dim
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = MinGPT(
            input_dim=input_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embed,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            block_size=horizon,
            vocab_size=vocab_size,
            action_dim=action_dim,
            discrete_input=discrete_input,
            predict_offsets=predict_offsets,
            offset_loss_scale=offset_loss_scale,
            focal_loss_gamma=focal_loss_gamma
        )


        self.obs_encoding_net = obs_encoding_net
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.state_prior = model

        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kwargs = kwargs

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        try:
            nobs = self.normalizer.normalize(obs_dict)
        except Exception as e:
            import ipdb; ipdb.set_trace()
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoding_net(this_nobs)
        # reshape back
        enc_obs = nobs_features.reshape(B, To, -1)

        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))
    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # mask out observations after n_obs_steps
        To = self.n_obs_steps
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        this_nobs = dict_apply(nobs,
                               lambda x: x[:, :To, ...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoding_net(this_nobs)
        enc_obs = nobs_features.reshape(batch_size, To, -1)
        latent = self.action_ae.encode_into_latent(nactions, enc_obs)
        _, loss, loss_components = self.state_prior.get_latent_and_loss(
            obs_rep=enc_obs,
            target_latents=latent,
            return_loss_components=True,
        )
        return loss, loss_components
