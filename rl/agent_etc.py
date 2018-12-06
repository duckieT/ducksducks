import torch
import numpy as np
from torch import nn
from torch .nn import functional as F
from __.vae.model_etc import *
from __.utils import *

# hyperparameters

agents = ['ltd', 'memory_ltd', 'conv_ltd']
agent = 'ltd'
memory_size = 5
observation_size = (3, 120, 160)
feature_dimensions = (20, 20, 10, 10)
action_size = (2,)
	
def load_agent (params):
	agent_params = params ['agent']
	if agent_params ['agent'] == 'conv_ltd':
		return conv_ltd (** agent_params)
	else:
		panic ('unrecognized agent kind: ' + str (agent_params ['agent']))
def save_agent (agent):
	return (
	{ ** save_agent_distinct (agent)
	, ** save_agent_shared (agent) })
def save_agent_distinct (agent):
	return (
	{ 'agent': 
		{ ** agent .params
		, 'state': { layer: parameter for layer, parameter in agent .state_dict () .items () if not 'shared_' in layer } } })
def save_agent_shared (agent):
	return (
	{ 'agent-shared': 
		{ 'state': { layer: parameter for layer, parameter in agent .state_dict () .items () if 'shared_' in layer } } })

class ltd (nn .Module):
	def __init__ (self, activation, action_size, model, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'ltd'
			, 'action_size': action_size })

		self .action_size = action_size

		action_dimensions = np .product (action_size)

		self .vae = model
		self .policy = nn .Sequential ( *
			[ nn .Linear (model .params ['encoding_dimensions'], 8)
			, nn .ReLU ()
			, nn .Linear (8, action_dimensions)
			, nn .Tanh () ])

		if state:
			self .load_state_dict (state)
	def sense (self, observation):
		upsampling = F .interpolate (torch .stack ([observation]), (image_size [1], image_size [0]))
		encoding, variation = self .vae .encode (upsampling)
		self .recall = encoding .view (-1)
	def recognition (self):
		return self .recall
	def forward (self, recall):
		return self .policy (recall) .view (* self .action_size)

class memory_ltd (nn .Module):
	def __init__ (self, memory_size, feature_dimensions, action_size, model, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'memory_ltd'
			, 'memory_size': memory_size
			, 'feature_dimensions': feature_dimensions
			, 'action_size': action_size })

		if not isinstance (feature_dimensions, tuple):
			feature_dimensions = (feature_dimensions,)

		action_dimensions = np .product (action_size)
		encoding_dimensions = model .params ['encoding_dimensions']

		self .memory_size = memory_size
		self .encoding_dimensions = encoding_dimensions
		self .feature_dimensions = feature_dimensions
		self .action_size = action_size

		self .recall = torch .zeros (memory_size, encoding_dimensions)

		self .vae = model
		self .policy = nn .Sequential ( *
			[ nn .Linear (memory_size * encoding_dimensions, feature_dimensions [0])
			, *
			[ layer for i in range (len (feature_dimensions) - 1)
				for layer in 
				( nn .ReLU ()
				, nn .Linear (feature_dimensions [i], feature_dimensions [i + 1]) ) ]
			, nn .ReLU ()
			, nn .Linear (feature_dimensions [-1], action_dimensions)
			, nn .Tanh () ])

		if state:
			self .load_state_dict (state)
	def sense (self, observation):
		upsampling = F .interpolate (torch .stack ([observation]), (image_size [1], image_size [0]))
		encoding, variation = self .vae .encode (upsampling)
		self .recall = torch .cat (
			( self .recall .narrow (0, 1, self .memory_size - 1)
			, encoding .view (1, self .encoding_dimensions) ) )
	def recognition (self):
		return self .recall
	def forward (self, recall):
		return self .policy (recall .view (-1)) .view (* self .action_size)

class conv_ltd (nn .Module):
	def __init__ (self, observation_size, action_size, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'conv_ltd'
			, 'observation_size': observation_size
			, 'action_size': action_size })

		self .observation_size = observation_size
		self .action_size = action_size

		action_dimensions = np .product (action_size)

		self .encoder = nn .Sequential ( *
			[ nn .Conv2d (image_channels, 4, kernel_size = 3, stride = 2)
			, nn .ReLU ()
			, nn .Conv2d (4, 8, kernel_size = 3, stride = 2)
			, nn .ReLU ()
			, nn .Conv2d (8, 16, kernel_size = 3, stride = 2)
			, nn .ReLU ()
			, nn .Conv2d (16, 32, kernel_size = 3, stride = 2) ])

		encoding_size = output_size (self .encoder, observation_size)
		encoding_dimensions = np .product (encoding_size)

		self .policy = nn .Sequential ( *
			[ nn .Linear (encoding_dimensions, 8)
			, nn .ReLU ()
			, nn .Linear (8, action_dimensions)
			, nn .Tanh () ])

		if state:
			self .load_state_dict (state)
	def sense (self, observation):
		encoding = self .encoder (F .avg_pool2d (observation, 4, 4) .unsqueeze (0))
		self .recall = encoding .view (-1)
	def recognition (self):
		return self .recall
	def forward (self, recall):
		return self .policy (recall) .view (* self .action_size)

def output_size (model, input_size):
	x = torch .randn (input_size) .unsqueeze (0)
	return model (x) .size () [1:]
