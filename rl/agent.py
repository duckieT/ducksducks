import torch
from agent_etc import *
from __.utils import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a rl agent')
	
	parser .add_argument ('--agent', type = str, metavar = 'm', help = 'kind of rl agent (' + ', ' .join (agents) + ')')

	parser .add_argument ('--activation', type = str, choices = activations, metavar = 'a', help = 'activation function in the hidden layers (default: ' + activation + ')')

	parser .add_argument ('--memory', type = int, metavar = 'n', help = 'past frames to remember (default: ' + str (memory_size) + ')')
	parser .add_argument ('--feature-dim', type = int, metavar = 'd', help = 'number of feature dimensions (default: ' + str (feature_dimensions) + ')')

	args = parser .parse_args ()

	agent_arg = if_none (agent, args .agent)

	if agent_arg == 'ltd':
		return (
		{ 'agent': 
			{ 'agent': 'ltd'
			, 'activation': if_none (activation, args .activation)
			, 'action_size': action_size } })
	elif agent_arg == 'memory_ltd':
		return (
		{ 'agent': 
			{ 'agent': 'memory_ltd'
			, 'memory_size': if_none (memory_size, args .memory)
			, 'feature_dimensions': if_none (feature_dimensions, args .feature_dim)
			, 'activation': if_none (activation, args .activation)
			, 'action_size': action_size } })
	else:
		raise Exception ('unknown agent', args .agent)

just_say ('Generating agent...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Agent generated!')
