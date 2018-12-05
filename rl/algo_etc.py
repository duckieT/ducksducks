import random
import torch
import itertools
import numpy as np
from torch import nn
from torch .nn import functional as F
from __.utils import *

# hyperparameters

algos = ['ga']
algo = 'ga'
elite_proportion = 0.05
elite_overselection = 2
elite_trials = 4
mutation_sd = 10
population_size = 300
	
def load_algo (params):
	algo_params = params ['algo']
	if algo_params ['algo'] == 'ga':
		return ga_algo (** algo_params)
	else:
		panic ('unrecognized algo kind: ' + str (algo_params ['algo']))
def save_algo (algo):
	if algo .params ['algo'] == 'ga':
		return (
		{ 'algo': 
			{ ** algo .params } })
	else:
		panic ('unrecognized algo kind: ' + str (algo .params ['algo']))

def ga_algo (culture, discrimination, reproduction, ** kwargs):
	cultivation_ = (
		culture [1:] if culture [0] == 'reward' else
		panic ('unrecognized culture kind: ' + str (culture [0])) )
	elite_proportion, elite_overselection, elite_trials = (
		(discrimination [1], 1, 1) if discrimination [0] == 'proportion' else
		discrimination [1:] if discrimination [0] == 'overproportion' else
		panic ('unrecognized discrimination kind: ' + str (discrimination [0])) )
	mutation_sd, population_size = (
		reproduction [1:] if reproduction [0] == 'mutation-only' else
		panic ('unrecognized reproduction kind: ' + str (reproduction [0])) )

	ga = thing ()
	ga .params = (
		{ 'algo': 'ga'
		, 'culture': culture
		, 'discrimination': discrimination
		, 'reproduction': reproduction })

	def evolve (habitat):
		generation = yield 'generation', population_size

		generation = naive_reproduction (mutation_sd, population_size) (generation)
		generation = hedonistic_culture (habitat) (generation)
		generation = fixed_selection (elite_proportion * elite_overselection * population_size) (generation)

		generation = yield 'pre-elites', generation

		generation = hedonistic_culture (habitat, trials = elite_trials) (generation)
		generation = fixed_selection (elite_proportion * population_size) (population)

		generation = yield 'elites', generation
	ga .evolve = evolve
	return ga

def naive_reproduction (mutation_sd, population_size):
	mutate = naive_mutation (mutation_sd) 
	def next_gen (generation):
		parents = [ desiderata (character) for character in generation ]

		children_room = population_size - len (parents)
		children = ( mutate (parents [i]) for i in random .choices (range (len (parents)), k = children_room) )

		yield from iter (parents)
		yield from children
	return next_gen

# desiderata ignores intermediate events
def hedonistic_culture (habitat, trials = 1):
	def culture (generation):
		def conception (individual):
			lives = ( desiderata (habitat .incarnate (individual)) for i in range (trials) )
			def conception ():
				yield from iter (())
				individual .fitness = sum ( desiderata (life) for life in lives ) / trials
				return individual
			return conception ()
		yield from ( conception (desiderata (character)) for character in generation )
	return culture

def fixed_selection (elites_size):
	import math
	elites_size = math .ceil (elites_size)
	def elites (generation):
		elites = []
		for individual in ( desiderata (character) for character in generation ):
			just_say ('received one survivor')
			individual_index = rank (elites, individual)
			if individual_index < elites_size:
				just_say ('accepted one elite')
				elites = elites [:individual_index] + [ individual ] + elites [individual_index:]
			yield iter (elites)
	return elites

def naive_mutation (mutation_sd):
	def mutate (individual):
		import copy

		genotype = individual .genotype
		blueprint = genotype .state_dict ()
		shared_genes = [ layer for layer in blueprint if 'shared_' in layer ]
		distinct_genes = { layer: parameter for layer, parameter in blueprint .items () if not layer in shared_genes }
		gene_sensitivities = parameter_sensitivities (genotype)
		mutated_distinct_genes = (
			{ layer: (chromosome + torch .normal (torch .zeros (chromosome .size ()), mutation_sd) * gene_sensitivities [layer]) .to (chromosome .device)
				for layer, chromosome in distinct_genes .items () })
		blueprint .update (mutated_distinct_genes)
		mutated_genotype = copy .deepcopy (genotype)
		for layer in shared_genes: setattr (mutated_genotype, getattr (genotype, layer))
		mutated_genotype .load_state_dict (blueprint)
		return bloodline (mutated_genotype)
	return mutate

def live (env, moralize = duckie_morals):
	def live (individual):
		life = individual .reincarnate ()
		value = None

		alive = True
		perception = env .reset ()
		while alive:
			observation = torch .from_numpy (perception) .to (dtype = torch.float32) .permute (2, 0, 1)
			action = life .choose (observation)
			try:
				perception, reward, dead, info = env .step (action)

				life .killed = (reward != 0)
				life .crashed = False
			except AssertionError:
				print ('unluck!', file = sys .stderr) 
				yield 'crash reset', None

				value = yield from live (env, moralize) (individual)
				return value
			except:
				import traceback
				print ('crashed!', file = sys .stderr) 
				print (perception, reward, dead, info, file = sys .stderr) 
				print (traceback .format_exc (), file = sys .stderr) 

				life .killed = True
				life .crashed = True
			yield 'step', (perception, reward, dead, info)
			value = moralize (env, life, action, perception, reward, dead, info) (value)
			alive = not dead
			
		return value
	return live

def duckie_morals (env, life, action, perception, reward, dead, info):
	if not hasattr (env, 'pos_history'):
		env .pos_history = []

	position_reward = reward

	forward, turn = action
	velocity_reward = max (0, forward * 0.4)

	rotation_penalty = - abs (turn) * 0.02

	step_size = 0.04
	cur_pos = tuple (env .cur_pos)
	x, y, z = cur_pos
	idle_steps = len ([ i for i, (x_, y_, z_) in enumerate (env .pos_history)
		if (x - x_) ** 2 + (y - y_) ** 2 + (z - z_) ** 2 < (step_size * (i + 1) * 0.8) ** 2 ])
	idle_penalty = - 0.02 * ( idle_steps ** 2 )

	if not dead:
		env .pos_history = [ cur_pos ] + env .pos_history
	else:
		env .pos_history = []

	reward = position_reward + velocity_reward + rotation_penalty + idle_penalty 

	def value (value):
		if value is None: return reward
		else: return value + reward
	return value

def rank (elites, individual):
	if len (elites) == 0:
		return 0
	elif len (elites) == 1:
		if elites [0] .fitness < individual .fitness:
			return 0
		else:
			return 1
	else:
		median = len (elites) // 2
		if elites [median] .fitness < individual .fitness:
			return rank (elites [:median], individual)
		else:
			return median + rank (elites [median:], individual)

def desiderata (generator):
	x = yield_ (generator)
	for _ in x: pass
	return x .value

def original (agent):
	agent = agent .cpu ()
	shared_genes = [ layer for layer, _ in agent .named_parameters () if 'shared_' in layer ]
	blueprint = genotype .state_dict ()
	parameters_origin = { layer: torch .zeros (parameter .size ()) for layer, parameter in blueprint .items () if not layer in shared_genes }
	blueprint .update (parameters_origin)
	agent .load_state_dict (blueprint)
	return agent

def character_from (individual):
	yield iter (())
	return individual)
def generation_from (population):
	return ( character_from (individual) for individual in population )

def bloodline (genotype):
	genotype .eval ()
	# device = next (genotype .parameters ()) .device
	def reincarnate ():
		import copy

		life = thing ()
		def choose (observation):
			with torch .no_grad ():
				# life .instinct .sense (observation .to (device))
				return tuple (life .instinct (life .instinct .recognition ()) .numpy ())
		life .instinct = copy .deepcopy (genotype)
		shared_genes = [ layer for layer, _ in genotype .named_parameters () if 'shared_' in layer ]
		for layer in shared_genes: setattr (mutated_genotype, getattr (genotype, layer))
		# life .instinct .eval ()
		life .choose = choose
		return life
	it = thing ()
	it .genotype = genotype
	it .reincarnate = reincarnate
	return it
