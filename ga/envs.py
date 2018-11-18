import gym

import gym_duckietown
from gym_duckietown.envs import *
from gym_duckietown.wrappers import *

def make_env(env_id, seed, rank, log_dir, start_container):
    def _thunk():
        env = gym.make(env_id)
        env = DiscreteWrapper(env)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        return env

    return _thunk
