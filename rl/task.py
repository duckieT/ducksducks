import os
import torch
from task_etc import *
from __.utils import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a rl task')
	
	parser .add_argument ('--task', type = str, default = 'evolve', metavar = 'x', help = 'kind of rl task (' + ', ' .join (tasks)+ ')')

	parser .add_argument ('--out', type = str, metavar = 'path', help = 'path to a folder to store output')
	parser .add_argument ('--log-file', type = str, metavar = 'path', help = 'path to a file to store log info')

	parser .add_argument ('--map', type = str, default = map_name, metavar = 'map', help = 'name of duckie map (default: ' + map_name + ')')
	parser .add_argument ('--frame-skip', type = int, default = frame_skip, metavar = 'n', help = 'frames to skip per step (default: ' + str (frame_skip) + ')')
	parser .add_argument ('--distortion', type = bool, default = distortion, metavar = 'x', help = 'whether to fisheye the camera (default: ' + str (distortion) + ')')
	parser .add_argument ('--max-steps', type = int, default = max_steps, metavar = 'n', help = 'max steps per episode (default: ' + str (max_steps) + ')')

	parser .add_argument ('--iteration-offset', type = int, default = iteration_offset, metavar = 'n', help = 'number of iterations to skip (default: ' + str (iteration_offset) + ')')
	parser .add_argument ('--iterations', type = int, default = iterations, metavar = 'n', help = 'number of iterations to train (default: ' + str (iterations) + ')')
	parser .add_argument ('--parallelism', type = int, default = parallelism, metavar = 'n', help = 'parallelism for training (default: ' + str (parallelism) + ')')
	parser .add_argument ('--batch-size', type = int, default = batch_size, metavar = 'n', help = 'how many agents to evolve per batch (default: ' + str (batch_size) + ')')
	parser .add_argument ('--log-interval', type = int, default = log_interval, metavar = 'n', help = 'how many agents to evolve before logging again (default: ' + str (log_interval) + ')')
	parser .add_argument ('--seed', type = int, default = 1, metavar = 's', help = 'random seed (default: 1)')
	parser .add_argument ('--no-cuda', action = 'store_true', default = False, help = 'disables cuda training')



	args = parser .parse_args ()

	if args .task == 'evolve':
		return (
		{ 'task': 
			{ 'task': 'evolve'
			, 'map_name': args .map
			, 'out_path': args .out
			, 'frame_skip': args .frame_skip
			, 'distortion': args .distortion
			, 'max_steps': args .max_steps
			, 'iteration_offset': args .iteration_offset
			, 'iterations': args .iterations
			, 'parallelism': args .parallelism
			, 'batch_size': args .batch_size
			, 'log_interval': args .log_interval
			, 'seed': args .seed
			, 'cuda_ok': not args .no_cuda and torch .cuda. is_available () } })
	elif args .task == 'sample':
		return (
		{ 'task': 
			{ 'task': 'sample'
			, 'map_name': args .map
			, 'log_file': args .log_file
			, 'frame_skip': args .frame_skip
			, 'distortion': args .distortion
			, 'max_steps': args .max_steps
			, 'cuda_ok': not args .no_cuda and torch .cuda. is_available () } })
	elif args .task == 'visualize':
		return (
		{ 'task': 
			{ 'task': 'visualize'
			, 'map_name': args .map
			, 'out_path': args .out
			, 'frame_skip': args .frame_skip
			, 'distortion': args .distortion
			, 'max_steps': args .max_steps
			, 'iteration_offset': args .iteration_offset
			, 'parallelism': args .parallelism
			, 'batch_size': args .batch_size
			, 'seed': args .seed
			, 'cuda_ok': not args .no_cuda and torch .cuda. is_available () } })
	else:
		panic ('unknown task ' + str (args .task))

just_say ('Generating task...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Task generated!')
