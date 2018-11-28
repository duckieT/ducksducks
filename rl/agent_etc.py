import torch
import numpy as np
from torch import nn
from torch .nn import functional as F
from __.vae.model_etc import *
from __.utils import *

# hyperparameters

agents = ['ltd', 'memory_ltd']
agent = 'ltd'
memory_size = 5
feature_dimensions = (20, 20, 10, 10)
activations = ['relu', 'leaky_relu', 'selu']
activation = 'relu'
action_size = (2,)
	
def load_agent (params):
	agent_params = params ['agent']
	if agent_params ['agent'] == 'ltd':
		model = load_model (params)
		state = (
			{ ** agent_params ['state']
			, ** { 'vae.' + layer: parameter for layer, parameter in model .state_dict () .items () } } if 'state' in agent_params else
			None )
		return ltd (** { ** agent_params, 'state': state }, model = model)
	elif agent_params ['agent'] == 'memory_ltd':
		model = load_model (params)
		state = (
			{ ** agent_params ['state']
			, ** { 'vae.' + layer: parameter for layer, parameter in model .state_dict () .items () } } if 'state' in agent_params else
			None )
		return memory_ltd (** { ** agent_params, 'state': state }, model = model)
	else:
		panic ('unrecognized agent kind: ' + str (agent_params ['agent']))
def save_agent (agent):
	return (
	{ ** save_agent_only (agent)
	, ** save_model (agent .vae) })
def save_agent_only (agent):
	return (
	{ 'agent': 
		{ ** agent .params
		, 'state': { layer: parameter for layer, parameter in agent .state_dict () .items () if not 'vae' in layer } } })

class ltd (nn .Module):
	def __init__ (self, activation, action_size, model, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'ltd'
			, 'activation': activation
			, 'action_size': action_size })

		self .action_size = action_size

		action_dimensions = np .product (action_size)

		self .vae = model
		self .policy = nn .Sequential ( *
			[ nn .Linear (model .params ['encoding_dimensions'], 8)
			, nn .Linear (8, action_dimensions)
			, nn .Tanh () ])

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
	def sense (self, observation):
		upsampling = F .interpolate (torch .stack ([observation]), (image_size [1], image_size [0]))
		encoding, variation = self .vae .encode (upsampling)
		self .recall = encoding .view (-1)
	def recognition (self):
		return self .recall
	def forward (self, recall):
		return self .policy (recall) .view (* self .action_size)

class memory_ltd (nn .Module):
	def __init__ (self, memory_size, feature_dimensions, activation, action_size, model, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'agent': 'memory_ltd'
			, 'memory_size': memory_size
			, 'feature_dimensions': feature_dimensions
			, 'activation': activation
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
			[ nn .Linear (feature_dimensions [i], feature_dimensions [i + 1]) 
				for i in range (len (feature_dimensions) - 1) ]
			, nn .Linear (feature_dimensions [-1], action_dimensions)
			, nn .Tanh () ])

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

# add agent with recall
