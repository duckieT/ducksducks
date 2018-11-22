import torch
from task_etc import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a rl task')
	
	parser .add_argument ('--task', type = str, default = 'evolve', metavar = 'x', help = 'kind of vae task (' + ', ' .join (tasks)+ ')')

	parser .add_argument ('--env', type = str, default = env_name, metavar = 'env', help = 'name of duckie environment')
	parser .add_argument ('--out', type = str, required = True, metavar = 'path', help = 'path to a folder to store output')

	parser .add_argument ('--iteration-offset', type = int, default = iteration_offset, metavar = 'n', help = 'number of iterations to skip (default: ' + str (iteration_offset) + ')')
	parser .add_argument ('--iterations', type = int, default = iterations, metavar = 'n', help = 'number of iterations to train (default: ' + str (iterations) + ')')
	parser .add_argument ('--parallelism', type = int, default = parallelism, metavar = 'n', help = 'parallelism for training (default: ' + str (parallelism) + ')')
	parser .add_argument ('--log-interval', type = int, default = log_interval, metavar = 'n', help = 'how many batches to wait before logging training status (default: ' + str (log_interval) + ')')
	parser .add_argument ('--seed', type = int, default = 1, metavar = 's', help = 'random seed (default: 1)')
	parser .add_argument ('--no-cuda', action = 'store_true', default = False, help = 'disables cuda training')


	args = parser .parse_args ()

	if args .task == 'evolve':
		return (
		{ 'task': 
			{ 'task': 'evolve'
			, 'env_name': args .env
			, 'out_path': args .out
			, 'iteration_offset': args .iteration_offset
			, 'iterations': args .iterations
			, 'parallelism': args .parallelism
			, 'log_interval': args .log_interval
			, 'seed': args .seed
			, 'cuda_ok': not args .no_cuda and torch .cuda. is_available () } })
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
just_say ('Task generated!')
