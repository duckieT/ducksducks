import torch
from torch import nn
from torch .nn import functional as F

# hyperparameters
input_image_size = (480, 640)
input_image_channels = 3

image_dimensions = input_image_channels * input_image_size [0] * input_image_size [1]

feature_dimensions = 1000
encoding_dimensions = 40
activation = 'selu'
	
def load_model (params):
	model_params = params ['model']
	if model_params ['model'] == 'naive':
		return naive_vae (** model_params)
def save_model (model):
	return (
	{ 'model': { ** model .params, 'state': model .state_dict () } })

class naive_vae (nn .Module):
	def __init__ (self, feature_dimensions, encoding_dimensions, activation, state = None, **kwargs):
		super () .__init__ ()
		self .params = (
			{ 'model': 'naive'
			, 'feature_dimensions': feature_dimensions
			, 'encoding_dimensions': encoding_dimensions
			, 'activation': activation })

		self .fc1 = nn .Linear (image_dimensions, feature_dimensions)
		self .fc21 = nn .Linear (feature_dimensions, encoding_dimensions)
		self .fc22 = nn .Linear (feature_dimensions, encoding_dimensions)
		self .fc3 = nn .Linear (encoding_dimensions, feature_dimensions)
		self .fc4 = nn .Linear (feature_dimensions, image_dimensions)

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

	def encode (self, x):
		h1 = self .activation (self .fc1 (x))
		return self .fc21 (h1), self .fc22 (h1)

	def reparameterize (self, mu, logvar):
		std = torch .exp (0.5 * logvar)
		eps = torch .randn_like (std)
		return eps .mul (std) .add_ (mu)

	def decode (self, z):
		h3 = self .activation (self .fc3 (z))
		return torch .sigmoid (self .fc4 (h3))

	def forward (self, x):
		mu, logvar = self .encode (x .view (-1, image_dimensions))
		z = self .reparameterize (mu, logvar)
		return self .decode (z), mu, logvar
