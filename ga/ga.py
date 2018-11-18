import copy
import glob
import os
import time
import operator
from functools import reduce

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

generations = 10000
population = 10000
seed = 1

torch.manual_seed(args.seed)
#if args.cuda:
#	 torch.cuda.manual_seed(args.seed)

envs = SubprocVecEnv([make_env(args.env_name, args.seed, i, args.log_dir, args.start_container)
			for i in range(population)])

obs_shape = envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
obs_size = reduce(operator.mul, obs_shape, 1)

model = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
if args.cuda:
	model.cuda()


if envs.action_space.__class__.__name__ == "Discrete":
	action_shape = 1
else:
	action_shape = envs.action_space.shape[0]


#optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)

current_obs = torch.zeros(args.num_processes, *obs_shape)

def update_current_obs(obs):
	shape_dim0 = envs.observation_space.shape[0]
	obs = torch.from_numpy(obs).float()
	if args.num_stack > 1:
		current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
	current_obs[:, -shape_dim0:] = obs

obs = envs.reset()
update_current_obs(obs)

for j in range(generations):
	for individual in range(policy.num_steps):
		# Sample actions
		value, action, action_log_prob, states = actor_critic.act(
			Variable(rollouts.observations[step]),
			Variable(rollouts.states[step]),
			Variable(rollouts.masks[step])
		)
		cpu_actions = action.data.squeeze(1).cpu().numpy()

		# Observation, reward and next obs
		obs, reward, done, info = envs.step(cpu_actions)

		update_current_obs(obs)

