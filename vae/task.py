import torch
from task_etc import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a vae task')
	
	parser .add_argument ('--task', type = str, default = 'instruct', metavar = 'm', help = 'kind of vae task (' + ', ' .join (tasks)+ ')')

	parser .add_argument ('--train', type = str, required = True, metavar = 'path', help = 'path to a folder containing training images for the vae')
	parser .add_argument ('--test', type = str, required = True, metavar = 'path', help = 'path to a folder containing test images for the vae')
	parser .add_argument ('--out', type = str, required = True, metavar = 'path', help = 'path to a folder to store output')

	parser .add_argument ('--learning-rate', type = float, default = learning_rate, metavar = 'n', help = 'learning rate for adam (default: ' + str (learning_rate) + ')')
	parser .add_argument ('--batch-size', type = int, default = batch_size, metavar = 'n', help = 'batch size for training (default: ' + str (batch_size) + ')')
	parser .add_argument ('--epoch-offset', type = int, default = epoch_offset, metavar = 'n', help = 'number of epochs to skip (default: ' + str (epoch_offset) + ')')
	parser .add_argument ('--epochs', type = int, default = epochs, metavar = 'n', help = 'number of epochs to train (default: ' + str (epochs) + ')')
	parser .add_argument ('--log-interval', type = int, default = log_interval, metavar = 's', help = 'how many batches to wait before logging training status (default: ' + str (log_interval) + ')')
	parser .add_argument ('--seed', type = int, default = 1, metavar = 's', help = 'random seed (default: 1)')
	parser .add_argument ('--no-cuda', action = 'store_true', default = False, help = 'disables cuda training')


	args = parser .parse_args ()

	if args .task == 'instruct':
		return (
		{ 'task': 
			{ 'task': 'instruct'
			, 'train_path': args .train
			, 'test_path': args .test
			, 'out_path': args .out
			, 'objective': 'vae'
			, 'learning_rate': args .learning_rate
			, 'batch_size': args .batch_size
			, 'epoch_offset': args .epoch_offset
			, 'epochs': args .epochs
			, 'log_interval': args .log_interval
			, 'cuda_ok': not args .no_cuda and torch .cuda. is_available ()
			, 'seed': args .seed } })
	else:
		panic ('unknown task', args .task)

def if_none (default_, value):
	return default_ if value == None else value

def just_say (text):
	import os
	tty = os .fdopen (os .open ('/dev/tty', os .O_WRONLY | os .O_NOCTTY), 'w', 1)
	print (text, file = tty)

just_say ('Generating task...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Instructor generated!')
