import torch
import numpy as np
from torch import nn
from torch .nn import functional as F
from __.vae.model_etc import *

# hyperparameters

agents = ['ltd']
agent = 'ltd'
encoding_dimensions = 200
activations = ['relu', 'leaky_relu', 'selu']
activation = 'selu'
action_size = (2,)
	
def load_agent (params):
	model = load_model (params)
	agent_params = params ['agent']
	if agent_params ['agent'] == 'ltd':
		return ltd (** agent_params, model = model)
	else:
		panic ('unrecognized agent kind: ' + str (agent_params ['agent']))
def save_agent (agent):
	return (
	{ 'agent': 
		{ ** agent .params
		, 'state': { layer: parameter for layer, parameter in agent .state_dict () if not 'vae' in layer } }
	, ** save_model (agent .vae) })

class ltd (nn .Module):
	def __init__ (self, encoding_dimensions, activation, action_size, model, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'ltd'
			, 'encoding_dimensions': encoding_dimensions
			, 'activation': activation
			, 'action_size': action_size })

		self .vae = model
		self .forward = nn .Sequential (
			[ nn .Linear (encoding_dimensions, 8)
			, nn .Linear (8, action_size)
			, nn .Tanh ])

		if activation == 'relu':
			self .activation = F .relu
		elif activation == 'leaky_relu':
			self .activation = F .leaky_relu
		elif activation == 'selu':
			self .activation = F .selu
		else:
			raise Exception ('unknown activation', self .activation)

		if state:
			self .load_state_dict (state)
	def sense (observation):
		self .recall = self .vae (F .interpolate (observation, image_size))
	def recognition ():
		return self .recall

# add agent with recall

def panic (reason):
	raise Exception (reason)
