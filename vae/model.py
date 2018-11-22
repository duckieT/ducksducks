import torch
from model_etc import *
from __.utils import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a vae model')
	
	parser .add_argument ('--model', type = str, required = True, metavar = 'm', help = 'kind of vae model (' + ', ' .join (models) + ')')

	parser .add_argument ('--feature-dim', type = int, default = None, metavar = 'd', help = 'number of feature dimonsions (default: ' + str (feature_dimensions) + ')')
	parser .add_argument ('--encoding-dim', type = int, default = None, metavar = 'd', help = 'number of encoding dimensions (default: ' + str (encoding_dimensions) + ')')
	parser .add_argument ('--activation', type = str, default = None, choices = ['relu', 'leaky_relu', 'selu'], metavar = 'a', help = 'activation function in the hidden layers (default: ' + activation + ')')

	args = parser .parse_args ()

	if args .model == 'naive':
		return (
		{ 'model': 
			{ 'model': 'naive'
			, 'feature_dimensions': if_none (feature_dimensions, args .feature_dim)
			, 'encoding_dimensions': if_none (encoding_dimensions, args .encoding_dim)
			, 'activation': if_none (activation, args .activation) } })
	elif args .model == 'snconv':
		return (
		{ 'model': 
			{ 'model': 'snconv'
			, 'encoding_dimensions': if_none (encoding_dimensions, args .encoding_dim) } })
	elif args .model == 'snconv2':
		return (
		{ 'model': 
			{ 'model': 'snconv2'
			, 'encoding_dimensions': if_none (encoding_dimensions, args .encoding_dim) } })
	elif args .model == 'snconv3':
		return (
		{ 'model': 
			{ 'model': 'snconv3'
			, 'encoding_dimensions': if_none (encoding_dimensions, args .encoding_dim) } })
	else:
		raise Exception ('unknown model', args .model)

just_say ('Generating model...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Model generated!')
