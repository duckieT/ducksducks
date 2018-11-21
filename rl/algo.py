import torch
from algo_etc import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a rl algo')
	
	parser .add_argument ('--algo', type = str, metavar = 'x', help = 'kind of rl algo (' + ', ' .join (algos) + ')')

	parser .add_argument ('--population', type = int, metavar = 'n', help = 'number of individuals per generation (default: ' + str (population_size) + ')')
	parser .add_argument ('--elite-proportion', type = float, metavar = 'p', help = 'proportion of population designated as elite (default: ' + str (elite_proportion) + ')')
	parser .add_argument ('--mutation-sd', type = float, metavar = 'mu', help = 'standard deviation of noise for mutation (default: ' + str (mutation_sd) + ')')

	args = parser .parse_args ()

	algo_arg = if_none (algo, args .algo)

	if algo_arg == 'ga':
		population_arg = if_none (population_size, args .population)
		elite_proportion_arg = if_none (elite_proportion, args .elite_proportion)
		mutation_sd_arg = if_none (mutation_sd, args .mutation_sd)
		return (
		{ 'algo': 
			{ 'algo': 'ga'
			, 'origination': ('random', population_arg, mutation_sd_arg)
			, 'culture': ('reward',)
			, 'discrimination': ('proportion', elite_proportion_arg)
			, 'reproduction': ('mutation-only', mutation_sd_arg, population_arg) } })
	else:
		raise Exception ('unknown algo', args .algo)
def if_none (default_, value):
	return default_ if value == None else value

def just_say (text):
	import os
	tty = os .fdopen (os .open ('/dev/tty', os .O_WRONLY | os .O_NOCTTY), 'w', 1)
	print (text, file = tty)

just_say ('Generating algo...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Algo generated!')
