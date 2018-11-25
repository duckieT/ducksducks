import torch
import numpy as np
from torch import nn
from torch .nn import functional as F
from __.utils import *

# hyperparameters

algos = ['ga']
algo = 'ga'
elite_proportion = 0.01
elite_overselection = 3
elite_trials = 10
mutation_sd = 1
population_size = 1000
batch_size = 100
	
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

def ga_algo (origination, culture, discrimination, reproduction, batch_size = 1, ** kwargs):
	aborigination = (
		random_sampling (origination [1], origination [2]) if origination [0] == 'random' else
		panic ('unrecognized origination kind: ' + str (origination [0])) )
	cultivate = (
		hedonistic_culture if culture [0] == 'reward' else
		panic ('unrecognized culture kind: ' + str (culture [0])) )

	elite_proportion, elite_overselection, elite_trials = (
		discrimination [1], 1, 1 if discrimination [0] == 'proportion' else
		discrimination [1:] if discrimination [0] == 'overproportion' else
		panic ('unrecognized discrimination kind: ' + str (discrimination [0])) )
	mutation_sd, population_size = (
		reproduction [1:] if reproduction [0] == 'mutation-only' else
		panic ('unrecognized reproduction kind: ' + str (reproduction [0])) )
	repopulate = naive_mutation_population (mutation_sd, population_size)

	ga = thing ()
	ga .params = (
		{ 'algo': 'ga'
		, 'origination': origination
		, 'culture': culture
		, 'discrimination': discrimination
		, 'reproduction': reproduction })

	def evolve (population, habitat, adam = None):
		yield 'generation', None
		if population is None:
			yield 'origination', population_size
			population = aborigination (adam)
			yield 'originated', population
		else:
			yield 'reproduce', population_size
			population = repopulate (population)
			yield 'reproduced', population
		survivors = cultivate (habitat, population, batch_size = batch_size)
		tentative_elites = yield from fixed_selection (elite_proportion * elite_overselection * population_size) (survivors)
		surviving_elites = cultivate (habitat, tentative_elites, elite_trials, batch_size = batch_size)
		elites = yield from fixed_selection (elite_proportion * population_size) (surviving_elites)
		return elites
	ga .next_generation = evolve
	return ga

def random_sampling (number, sd):
	def genesis (agent):
		agent .cpu ()
		# save my cuda memory
		parameters_origin = { i: torch .zeros (parameter .size ()) for i, parameter in agent .state_dict () .items () }
		agent .load_state_dict (parameters_origin)
		adam = bloodline (agent)

		random = naive_mutation (sd)

		return ( random (adam) for _ in range (number) )
	return genesis

def hedonistic_culture (habitat, population, trials = 1, batch_size = 1):
	def chunks (iterable, size):
		from itertools import chain, islice
		iterator = iter (iterable)
		for first in iterator:
			just_say ('chunking one batch')
			yield chain ([first], islice (iterator, size - 1))
	def contributed_individual (individual):
		contributions = []
		for i in range (trials):
			life = yield_ (habitat .tryout (individual))
			for moment in life: pass
			contributions += [ life .value ]
		return contributions, individual
	def cultured_individual (contributions, individual):
		def contribution_reward (contribution):
			reward = yield_ (contribution)
			for moment in reward: pass
			return reward .value
		rewards = ( contribution_reward (contribution) for contribution in contributions )
		individual .fitness = sum (rewards) / trials
		return individual
	def contribued_batch (individuals):
		just_say ('issuing one batch')
		return [ contributed_individual (individual) for individual in individuals ]
	def cultured_batch (contribued_batch):
		just_say ('culturing one batch') 
		return ( cultured_individual (contributions, individual) for contributions, individual in contribued_batch )
		
	return (individual for batch in chunks (population, batch_size) for individual in cultured_batch (contribued_batch (batch)))

def fixed_selection (elites_size):
	def elites (survivors):
		elites = []
		for individual in survivors:
			just_say ('received one survivor')
			individual_index = rank (elites, individual)
			if individual_index < elites_size:
				elites = elites [:individual_index] + [ individual ] + elites [individual_index:]
			yield 'discriminating', elites
		yield 'discriminated', elites
		return elites
	return elites
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

def naive_mutation_population (mutation_sd, population_size):
	mutate = naive_mutation (mutation_sd) 
	def next_gen (elites):
		import random
		from itertools import chain

		children_room = population_size - len (elites)
		children = ( mutate (elites [i]) for i in random .choices (range (len (elites)), k = children_room) )
		return chain (iter (elites), children)
		
	return next_gen

def live (env, individual):
	life = individual .reincarnate ()
	total_reward = 0

	try:
		alive = True
		observation = env .reset ()
		while alive:
			observation = torch .from_numpy (observation) .to (dtype = torch.float32) .permute (2, 0, 1)
			action = life .choice (observation)
			observation, reward, dead, info = env .step (action)
			yield 'step', (observation, reward, dead, info)
			total_reward += reward
			alive = not dead
		life .killed = (reward != 0)
		life .crashed = False
	except:
		import sys
		import traceback
		print ('crashed!', file = sys .stderr) 
		print (observation, reward, dead, info, file = sys .stderr) 
		print (traceback .format_exc (), file = sys .stderr) 
		life .killed = True
		life .crashed = True
		
	return total_reward

def naive_mutation (mutation_sd):
	def mutate (individual):
		import copy

		genotype = individual .genotype
		genotype_state = genotype .state_dict ()
		genes = { layer: parameter for layer, parameter in genotype_state .items () if not 'vae' in layer }
		mutated_genes = (
			{ i: chromosome + torch .normal (torch .zeros (chromosome .size ()) .to (chromosome .device), mutation_sd)
				for i, chromosome in genes .items () })
		genotype_state .update (mutated_genes)
		mutated_genotype = copy .deepcopy (genotype)
		mutated_genotype .vae = genotype .vae
		mutated_genotype .load_state_dict (genotype_state)
		return bloodline (mutated_genotype)
	return mutate

def bloodline (genotype):
	device = next (genotype .parameters ()) .device
	def reincarnate ():
		import copy

		life = thing ()
		def choice (observation):
			with torch .no_grad ():
				life .instinct .sense (observation .to (device))
				return life .instinct (life .instinct .recognition ())
		life .instinct = copy .deepcopy (genotype)
		life .instinct .vae = genotype .vae
		life .instinct .eval ()
		life .choice = choice
		return life
	it = thing ()
	it .genotype = genotype
	it .reincarnate = reincarnate
	return it
