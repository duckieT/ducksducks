import math
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
mutation_scale = 3
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
	mutation_scale, population_size = (
		reproduction [1:] if reproduction [0] == 'mutation-only' else
		panic ('unrecognized reproduction kind: ' + str (reproduction [0])) )

	ga = thing ()
	ga .params = (
		{ 'algo': 'ga'
		, 'culture': culture
		, 'discrimination': discrimination
		, 'reproduction': reproduction })

	def evolve (habitat):
		generation = yield 'generation', ()

		generation = generation_from (sd_reproduction (mutation_scale, population_size) (generation))
		generation = hedonistic_culture (habitat) (generation)
		survivors = fixed_selection (elite_proportion * elite_overselection * population_size) (generation)

		generation = yield 'pre-elites', (survivors, population_size)

		generation = hedonistic_culture (habitat, trials = elite_trials) (generation)
		survivors = fixed_selection (elite_proportion * population_size) (generation)

		generation = yield 'elites', (survivors, math .ceil (elite_proportion * elite_overselection * population_size))
	ga .evolve = evolve
	return ga

def naive_reproduction (mutation_scale, population_size):
	mutate = naive_mutation (blind_sensitivities (mutation_scale)) 
	def next_gen (generation):
		parents = [ desiderata (character) for character in generation ]

		children_room = population_size - len (parents)
		children = ( mutate (parents [i]) for i in random .choices (range (len (parents)), k = children_room) )

		yield from iter (parents)
		yield from children
	return next_gen

def sd_reproduction (mutation_scale, population_size):
	def next_gen (generation):
		parents = [ desiderata (character) for character in generation ]
		mutate = naive_mutation (sd_sensitivities (parents, mutation_scale)) 

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
			# just_say ('issued one individual')
			def conception ():
				yield from iter (())
				individual .fitness = sum ( desiderata (life) for life in lives ) / trials
				individual .trials = (
					trials if not individual .trials else
					individual .trials + trials )
				return individual
			return conception ()
		yield from ( conception (desiderata (character)) for character in generation )
	return culture

def fixed_selection (elites_size):
	elites_size = math .ceil (elites_size)
	def elites (generation):
		selection = thing ()
		selection .elites = []
		def elites_with (character):
			yield from iter (())
			individual = desiderata (character)
			just_say ('received one survivor')
			individual_index = rank (selection .elites, individual)
			if individual_index < elites_size:
				just_say ('accepted one elite')
				selection .elites = selection .elites [:individual_index] + [ individual ] + selection .elites [individual_index:]
			return selection .elites
		yield from ( elites_with (character) for character in generation )
	return elites

def naive_mutation (parameter_sensitivities):
	def mutate (individual):
		import copy

		genotype = individual .genotype
		blueprint = genotype .state_dict ()
		shared_genes = [ layer for layer in blueprint if 'shared_' in layer ]
		distinct_genes = { layer: parameter for layer, parameter in blueprint .items () if not layer in shared_genes }
		gene_sensitivities = parameter_sensitivities (genotype)
		mutated_distinct_genes = (
			{ layer: (chromosome + torch .normal (gene_sensitivities [layer])) .to (chromosome .device)
				for layer, chromosome in distinct_genes .items () })
		blueprint .update (mutated_distinct_genes)
		mutated_genotype = copy .deepcopy (genotype)
		for layer in shared_genes: setattr (mutated_genotype, getattr (genotype, layer))
		mutated_genotype .load_state_dict (blueprint)
		return bloodline (mutated_genotype)
	return mutate

def blind_sensitivities (scale):
	def sensitivities (module):
		return { layer: scale * torch .ones (parameter .size ()) for layer, parameter in module .named_parameters () }
	return sensitivities
def sd_sensitivities (population, scale, epsilon = 0.01):
	if len (population) == 1:
		return blind_sensitivities (scale)
	else:
		typical_genome = next (iter (population)) .genotype
		gene_pool = [ individual .genotype .state_dict () for individual in population ]
		sensitivity = (
		{ layer: torch .clamp (torch .std (torch .stack (
			[ genome [layer] for genome in gene_pool ])
			, dim = 0), min = epsilon) * scale
			for layer, _ in typical_genome .named_parameters () } )
		def sensitivities (module):
			return sensitivity
		return sensitivities

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
		if (x - x_) ** 2 + (y - y_) ** 2 + (z - z_) ** 2 < (0.8 * step_size * (i + 1) ** 0.9) ** 2 ])
	idle_penalty = - 0.02 * ( idle_steps ** 2 )

	killed_penalty = ( - 100000000 if life .killed else 0 )

	if not dead:
		env .pos_history = [ cur_pos ] + env .pos_history
	else:
		env .pos_history = []

	reward = rotation_penalty + velocity_reward + position_reward + idle_penalty + killed_penalty

	def value (value):
		if value is None: return reward
		else: return value + reward
	return value

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

				life .killed = dead and (reward != 0)
				life .crashed = False
			except AssertionError:
				import sys

				print ('unluck!', file = sys .stderr) 
				yield 'crash reset', None

				value = yield from live (individual)
				return value
			except:
				import sys
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

def original (genotype):
	genotype = genotype .cpu ()
	shared_genes = [ layer for layer, _ in genotype .named_parameters () if 'shared_' in layer ]
	blueprint = genotype .state_dict ()
	parameters_origin = { layer: torch .zeros (parameter .size ()) for layer, parameter in blueprint .items () if not layer in shared_genes }
	blueprint .update (parameters_origin)
	genotype .load_state_dict (blueprint)
	return genotype

def character_from (individual):
	yield iter (())
	return individual
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
				life .instinct .sense (observation)
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
