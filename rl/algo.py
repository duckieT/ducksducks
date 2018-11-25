import torch
from algo_etc import *
from __.utils import *

def params ():
	import argparse
	parser = argparse .ArgumentParser (description = 'pick a rl algo')
	
	parser .add_argument ('--algo', type = str, metavar = 'x', help = 'kind of rl algo (' + ', ' .join (algos) + ')')

	parser .add_argument ('--population', type = int, metavar = 'n', help = 'number of individuals per generation (default: ' + str (population_size) + ')')
	parser .add_argument ('--elite-proportion', type = float, metavar = 'p', help = 'proportion of population designated as elite (default: ' + str (elite_proportion) + ')')
	parser .add_argument ('--elite-overselection', type = float, metavar = 'p', help = 'number of tentative elites for every elite (default: ' + str (elite_overselection) + ')')
	parser .add_argument ('--elite-trials', type = int, metavar = 'n', help = 'number of extra trials for every elite (default: ' + str (elite_trials) + ')')
	parser .add_argument ('--mutation-sd', type = float, metavar = 'mu', help = 'standard deviation of noise for mutation (default: ' + str (mutation_sd) + ')')

	args = parser .parse_args ()

	algo_arg = if_none (algo, args .algo)

	if algo_arg == 'ga':
		population_arg = if_none (population_size, args .population)
		elite_proportion_arg = if_none (elite_proportion, args .elite_proportion)
		elite_overselection_arg = if_none (elite_overselection, args .elite_overselection)
		elite_trials_arg = if_none (elite_trials, args .elite_trials)
		mutation_sd_arg = if_none (mutation_sd, args .mutation_sd)
		return (
		{ 'algo': 
			{ 'algo': 'ga'
			, 'origination': ('random', population_arg, mutation_sd_arg)
			, 'culture': ('reward',)
			, 'discrimination': ('overproportion', elite_proportion_arg, elite_overselection_arg, elite_trials)
			, 'reproduction': ('mutation-only', mutation_sd_arg, population_arg)
			, 'batch_size': batch_size } })
	else:
		raise Exception ('unknown algo', args .algo)

just_say ('Generating algo...')
just_say ('--------------------------------------------------------------------------------')
torch .save (params (), '/dev/stdout')
just_say ('--------------------------------------------------------------------------------')
just_say ('Algo generated!')
