import torch
from model_etc import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a vae model')
	
	parser .add_argument ('--model', type = str, required = True, metavar = 'm', help = 'either: kind of vae model (naive, cnn), or path to existing model')

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
	elif args .model == 'cnn':
		raise Exception ('jay hasnt implemented this yet')
	else:
		raise Exception ('unknown model', args .model)
def if_none (default_, value):
	return default_ if value == None else value

def just_say (text):
	import os
	tty = os .fdopen (os .open ('/dev/tty', os .O_WRONLY | os .O_NOCTTY), 'w', 1)
	print (text, file = tty)

just_say ('Generating model...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Model generated!')
