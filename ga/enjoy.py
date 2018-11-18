import argparse
import os
import types
import time

import numpy as np
import torch
from torch .autograd import Variable
from pytorch_rl .vec_env .dummy_vec_env import DummyVecEnv

from envs import make_env

parser = argparse .ArgumentParser (description='RL')
parser .add_argument ('--seed', type=int, default=1, help='random seed (default: 1)')
parser .add_argument ('--num-stack', type=int, default=4, help='number of frames to stack (default: 4)')
parser .add_argument ('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
parser .add_argument ('--env-name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
parser .add_argument ('--load-dir', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
parser .add_argument ('--start-container', action='store_true', default=False, help='start the Duckietown container image')

args = parser .parse_args ()

env = make_env (args .env_name, args .seed, 0, None, args .start_container)
env = DummyVecEnv ([env])

# get model

render = env .envs [0] .render
def display ():
	render ('human')

observation_shape = env .observation_space .shape
observation_shape = (observation_shape [0] * args .num_stack, * observation_shape [1:])
current_observation = torch .zeros (1, * observation_shape)
states = torch .zeros (1, model .state_size)
masks = torch .zeros (1, 1)

def update_current_observation (observation):
    shape_dim0 = env .observation_space .shape [0]
    observation = torch .from_numpy(observation) .float()
    if args .num_stack > 1:
        current_observation [:, :-shape_dim0] = current_observation [:, shape_dim0:]
    current_observation [:, -shape_dim0:] = observation

display ()
observation = env .reset ()
update_current_observation (observation)

window = env .envs [0] .unwrapped .window
@window .event
def on_key_press (symbol, modifiers):
    from pyglet .window import key
    import sys
    if symbol == key .ESCAPE:
        env .close()
        sys .exit (0)

try:
    while True:
        value, action, _, states = model .act (
            Variable (current_observation),
            Variable (states),
            Variable (masks),
            deterministic = True
        )
        states = states .data
        cpu_actions = action .data .squeeze (1) .cpu () .numpy ()

        print (cpu_actions)

        # Obser reward and next observation
        observation, reward, done, _ = env .step (cpu_actions)
        time .sleep (1 / env .envs [0] .unwrapped .frame_rate)

        masks .fill_ (0.0 if done else 1.0)

        if current_observation .dim () == 4:
            current_observation *= masks .unsqueeze (2) .unsqueeze(2)
        else:
            current_observation *= masks
        update_current_observation (observation)

        display ()

except:
    env .envs [0] .unwrapped .close ()
    time .sleep (0.25)
