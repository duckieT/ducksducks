from __future__ import print_function

import torch
from torch import nn, optim
from torch .nn import functional as F

from torchvision .utils import save_image


# hyperparameters
input_image_size = (480, 640)
input_image_channels = 3
feature_dimensions = 1000
encoding_dimensions = 40

learning_rate = 1e-3

# test hyperparameters
test_sampling_n = 8


def params ():
	global learning_rate
	global feature_dimensions
	global encoding_dimensions

	import argparse
	import os
	import sys
	parser = argparse .ArgumentParser (description = 'vae x ducks')
	
	parser .add_argument ('--data', type = str, required = True, metavar = 'path', help = 'path to a folder containing training images for the vae')
	parser .add_argument ('--test', type = str, default = '', metavar = 'path', help = 'path to a folder containing test images for the vae (default: training dataset)')
	parser .add_argument ('--init', type = str, default = '', metavar = 'path', help = 'path to a trained model file for initializing training')

	parser .add_argument ('--learning-rate', type = float, default = learning_rate, metavar = 'n', help = 'learning rate for adam (default: ' + str (learning_rate) + ')')
	parser .add_argument ('--feature-dim', type = int, default = feature_dimensions, metavar = 'd', help = 'number of feature dimonsions (default: ' + str (feature_dimensions) + ')')
	parser .add_argument ('--encoding-dim', type = int, default = encoding_dimensions, metavar = 'd', help = 'number of encoding dimensions (default: ' + str (encoding_dimensions) + ')')
	
	parser .add_argument ('--batch-size', type = int, default = 10, metavar = 'n', help = 'batch size for training (default: 10)')
	parser .add_argument ('--epochs', type = int, default = 10, metavar = 'n', help = 'number of epochs to train (default: 10)')

	parser .add_argument ('--log-interval', type = int, default = 10, metavar = 's', help = 'how many batches to wait before logging training status (default: 10)')
	parser .add_argument ('--seed', type = int, default = 1, metavar = 's', help = 'random seed (default: 1)')
	parser .add_argument ('--no-cuda', action = 'store_true', default = False, help = 'disables CUDA training')

	parser .add_argument ('--out', type = str, required = True, metavar = 'path', help = 'path to a folder to store output')

	args = parser .parse_args ()
	args .cuda = not args .no_cuda and torch .cuda .is_available ()
	args .test = args .test or args .data

	learning_rate = args .learning_rate
	feature_dimensions = args .feature_dim
	encoding_dimensions = args .encoding_dim

	os .makedirs (args .out, exist_ok = True)
	if os .listdir (args .out):
		print ('Warning: ' + args .out + ' is not empty!', file = sys .stderr)

	return args

def load_data (path, cuda = True):
	import os
	import tempfile
	from torch .utils .data import DataLoader
	from torchvision import datasets, transforms

	image_folder_path = tempfile .TemporaryDirectory () .name
	os .makedirs (image_folder_path)
	os .symlink (os .path .realpath (path), os .path .join (image_folder_path, 'data'))

	cuda_args = {'num_workers': 1, 'pin_memory': True} if args .cuda else {}
	return DataLoader (
		dataset = datasets .ImageFolder (image_folder_path, transform = transforms .ToTensor ()),
		batch_size = args .batch_size,
		shuffle = True,
		**cuda_args)

def out_file (filename):
	import os
	return os .path .join (args .out, filename)

class VAE (nn .Module):
	def __init__ (self, feature_dimensions, encoding_dimensions):
		super (VAE, self) .__init__ ()

		self .fc1 = nn .Linear (input_image_size [0] * input_image_size [1], feature_dimensions)
		self .fc21 = nn .Linear (feature_dimensions, encoding_dimensions)
		self .fc22 = nn .Linear (feature_dimensions, encoding_dimensions)
		self .fc3 = nn .Linear (encoding_dimensions, feature_dimensions)
		self .fc4 = nn .Linear (feature_dimensions, input_image_size [0] * input_image_size [1])

	def encode (self, x):
		h1 = F .relu (self .fc1 (x))
		return self .fc21 (h1), self .fc22 (h1)

	def reparameterize (self, mu, logvar):
		std = torch .exp (0.5 * logvar)
		eps = torch .randn_like (std)
		return eps .mul (std) .add_ (mu)

	def decode (self, z):
		h3 = F .relu (self .fc3 (z))
		return torch .sigmoid (self .fc4 (h3))

	def forward (self, x):
		mu, logvar = self .encode (x .view (-1, input_image_size [0] * input_image_size [1]))
		z = self .reparameterize (mu, logvar)
		return self .decode (z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function (recon_x, x, mu, logvar):
	BCE = F .binary_cross_entropy (recon_x, x .view (-1, input_image_size [0] * input_image_size [1]), reduction = 'sum')

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum (1 + log (sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch .sum (1 + logvar - mu .pow (2) - logvar .exp ())

	return BCE + KLD


def train (epoch):
	model .train ()
	train_loss = 0
	for batch_idx, (data, _) in enumerate (train_loader):
		data = data .to (device)
		optimizer .zero_grad ()
		recon_batch, mu, logvar = model (data)
		loss = loss_function (recon_batch, data, mu, logvar)
		loss .backward ()
		train_loss += loss .item ()
		optimizer .step ()
		if batch_idx % args .log_interval == 0:
			print ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' .format (
				epoch, batch_idx * len (data), len (train_loader .dataset),
				100. * batch_idx / len (train_loader),
				loss .item () / len (data)))

	print ('====> Epoch: {} Average loss: {:.4f}' .format (
		  epoch, train_loss / len (train_loader .dataset)))


def test (epoch):
	model .eval ()
	total_test_loss = 0
	with torch .no_grad ():
		for i, (data, _) in enumerate (test_loader):
			data = data .to (device)
			recon_batch, mu, logvar = model (data)
			total_test_loss += loss_function (recon_batch, data, mu, logvar) .item ()
			if i == 0:
				n = min (data .size (0), test_sampling_n)
				comparison = torch .cat ([
						data [:n],
						recon_batch .view (args .batch_size, input_image_channels, input_image_size [0], input_image_size [1]) [:n] ])
				save_image (comparison .cpu (),
						 out_file ('reconstruction_' + str (epoch) + '.png'), nrow = n)

	test_loss = total_test_loss / len (test_loader .dataset)
	print ('====> Test set loss: {:.4f}' .format (test_loss))



args = params ()

torch .manual_seed (args .seed)

device = torch .device ('cuda' if args .cuda else 'cpu')


train_loader = load_data (args .data, args .cuda)
test_loader = load_data (args .test, args .cuda)

model = VAE (feature_dimensions, encoding_dimensions) .to (device)
optimizer = optim .Adam (model .parameters (), lr = learning_rate)
if args .init:
	state = torch .load (args .init)
	model .load_state_dict (state ['model'])
	optimizer .load_state_dict (state ['optimizer'])


epoch_offset = 1 if not args .init else state ['epoch'] + 1

for epoch in range (epoch_offset, epoch_offset + args .epochs):
	train (epoch)
	test (epoch)
	with torch .no_grad ():
		sample = torch .randn (test_sampling_n ** 2, encoding_dimensions) .to (device)
		sample = model .decode (sample) .cpu ()
		save_image (sample .view (test_sampling_n ** 2, 1, input_image_size [0], input_image_size [1]),
				out_file ('sample_' + str (epoch) + '.png'))

		state = {
			'epoch': epoch,
			'model': model .state_dict (),
			'optimizer': optimizer .state_dict () }
		torch .save (state, out_file ('state_' + str (epoch) + '.pt'))
