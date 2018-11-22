import torch
import numpy as np
from torch import nn
from torch .nn import functional as F

# hyperparameters
image_size = (640, 480)
image_channels = 3

image_dimensions = image_channels * image_size [0] * image_size [1]

models = ['naive', 'snconv', 'snconv2', 'snconv3']
feature_dimensions = 1000
encoding_dimensions = 40
activation = 'selu'
	
def load_model (params):
	model_params = params ['model']
	if model_params ['model'] == 'naive':
		return naive_vae (** model_params)
	elif model_params ['model'] == 'snconv':
		return snconv_vae (** model_params)
	elif model_params ['model'] == 'snconv2':
		return snconv2_vae (** model_params)
	elif model_params ['model'] == 'snconv3':
		return snconv3_vae (** model_params)
	else:
		panic ('unrecognized model kind: ' + str (model_params ['model']))
def save_model (model):
	return (
	{ 'model': { ** model .params, 'state': model .state_dict () } })

class naive_vae (nn .Module):
	def __init__ (self, feature_dimensions, encoding_dimensions, activation, state = None, ** kwargs):
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

class snconv_vae (nn .Module):
	def __init__ (self, encoding_dimensions, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'model': 'snconv'
			, 'encoding_dimensions': encoding_dimensions })

		self .feature_encoder = nn .Sequential ( *
			[ nn .Conv2d (image_channels, 32, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (32, 64, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (64, 128, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (128, 256, kernel_size = 4, stride = 2) ])

		self .feature_size = feature_size = output_size ((image_channels, image_size [1], image_size [0]), self .feature_encoder)
		self .feature_dimensions = feature_dimensions = np .prod (feature_size)

		self .feature_decoder = nn .Sequential ( *
			[ nn .ConvTranspose2d (256, 128, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (128, 64, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (64, 32, kernel_size = 4, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (32, image_channels, kernel_size = 6, stride = 2) ])

		self .latent_encoding = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_variation = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_decoder = nn .Linear (encoding_dimensions, feature_dimensions)

		if state:
			self .load_state_dict (state)

	def encode (self, x):
		features = F .selu (self .feature_encoder (x)) .view (-1, self .feature_dimensions)
		return self .latent_encoding (features), self .latent_variation (features)

	def reparameterize (self, mu, logvar):
		std = torch .exp (0.5 * logvar)
		eps = torch .randn_like (std)
		return eps .mul (std) .add_ (mu)

	def decode (self, z):
		decoder_features = F .selu (self .latent_decoder (z)) .view (-1, * self .feature_size)
		return torch .sigmoid (self .feature_decoder (decoder_features))

	def forward (self, x):
		mu, logvar = self .encode (x)
		z = self .reparameterize (mu, logvar)
		return self .decode (z), mu, logvar

class snconv2_vae (nn .Module):
	def __init__ (self, encoding_dimensions, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'model': 'snconv2'
			, 'encoding_dimensions': encoding_dimensions })

		self .feature_encoder = nn .Sequential ( *
			[ nn .Conv2d (image_channels, 2, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (2, 4, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (4, 8, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (8, 16, kernel_size = 3, stride = 2) ])

		self .feature_size = feature_size = output_size ((image_channels, image_size [1], image_size [0]), self .feature_encoder)
		self .feature_dimensions = feature_dimensions = np .prod (feature_size)

		self .feature_decoder = nn .Sequential ( *
			[ nn .ConvTranspose2d (16, 8, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (8, 4, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (4, 2, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (2, image_channels, kernel_size = 4, stride = 2) ])

		self .latent_encoding = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_variation = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_decoder = nn .Linear (encoding_dimensions, feature_dimensions)

		if state:
			self .load_state_dict (state)

	def encode (self, x):
		features = F .selu (self .feature_encoder (x)) .view (-1, self .feature_dimensions)
		return self .latent_encoding (features), self .latent_variation (features)

	def reparameterize (self, mu, logvar):
		std = torch .exp (0.5 * logvar)
		eps = torch .randn_like (std)
		return eps .mul (std) .add_ (mu)

	def decode (self, z):
		decoder_features = F .selu (self .latent_decoder (z)) .view (-1, * self .feature_size)
		return torch .sigmoid (self .feature_decoder (decoder_features))

	def forward (self, x):
		mu, logvar = self .encode (x)
		z = self .reparameterize (mu, logvar)
		return self .decode (z), mu, logvar

class snconv3_vae (nn .Module):
	def __init__ (self, encoding_dimensions, state = None, ** kwargs):
		super () .__init__ ()
		self .params = (
			{ 'model': 'snconv3'
			, 'encoding_dimensions': encoding_dimensions })

		self .feature_encoder = nn .Sequential ( *
			[ nn .Conv2d (image_channels, 4, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (4, 8, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (8, 16, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .Conv2d (16, 32, kernel_size = 3, stride = 2) ])

		self .feature_size = feature_size = output_size ((image_channels, image_size [1], image_size [0]), self .feature_encoder)
		self .feature_dimensions = feature_dimensions = np .prod (feature_size)

		self .feature_decoder = nn .Sequential ( *
			[ nn .ConvTranspose2d (32, 16, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (16, 8, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (8, 4, kernel_size = 3, stride = 2)
			, nn .SELU ()
			, nn .ConvTranspose2d (4, image_channels, kernel_size = 4, stride = 2) ])

		self .latent_encoding = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_variation = nn .Linear (feature_dimensions, encoding_dimensions)
		self .latent_decoder = nn .Linear (encoding_dimensions, feature_dimensions)

		if state:
			self .load_state_dict (state)

	def encode (self, x):
		features = F .selu (self .feature_encoder (x)) .view (-1, self .feature_dimensions)
		return self .latent_encoding (features), self .latent_variation (features)

	def reparameterize (self, mu, logvar):
		std = torch .exp (0.5 * logvar)
		eps = torch .randn_like (std)
		return eps .mul (std) .add_ (mu)

	def decode (self, z):
		decoder_features = F .selu (self .latent_decoder (z)) .view (-1, * self .feature_size)
		return torch .sigmoid (self .feature_decoder (decoder_features))

	def forward (self, x):
		mu, logvar = self .encode (x)
		z = self .reparameterize (mu, logvar)
		return self .decode (z), mu, logvar

def output_size (input_size, model):
	x = torch .randn (input_size) .unsqueeze (0)
	return model (x) .size () [1:]
def panic (reason):
	raise Exception (reason)
