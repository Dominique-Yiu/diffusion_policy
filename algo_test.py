from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
import robomimic.utils.obs_utils as ObsUtils

config = get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='image',
        task_name='square',
        dataset_type='ph'
        )
ObsUtils.initialize_obs_utils_with_config(config)

obs_key_shapes = dict()
obs_key_shapes['agent_pos'] = list([2])
obs_key_shapes['image'] = list([3, 96, 96])
action_dim = 2

policy: PolicyAlgo = algo_factory(
        algo_name = config.algo_name,
        config = config,
        obs_key_shapes=obs_key_shapes,
        ac_dim = action_dim,
        device = 'cpu')
import ipdb; ipdb.set_trace()
print(policy)
