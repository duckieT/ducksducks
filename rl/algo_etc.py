import torch
import numpy as np
from torch import nn
from torch .nn import functional as F

# hyperparameters

algos = ['ga']
algo = 'ga'
elite_proportion = 0.1
mutation_sd = 1
population_size = 10000
	
def load_algo (params):
	algo_params = params ['algo']
	if algo_params ['algo'] == 'algo':
		return init_ga (** algo_params)
	else:
		panic ('unrecognized algo kind: ' + str (algo_params ['algo']))
def save_algo (algo):
	if algo .params ['algo'] == 'ga':
		return (
		{ 'algo': 
			{ ** algo .params } })
	else:
		panic ('unrecognized algo kind: ' + str (algo .params ['algo']))

def init_ga (origination, culture, discrimination, reproduction, ** kwargs):
	aborigination = (
		random_sampling (origination [1], origination [2]) if origination [0] == 'random' else
		panic ('unrecognized origination kind: ' + str (origination [0])))
	cultivate = (
		hedonistic_culture if culture [0] == 'reward' else
		panic ('unrecognized culture kind: ' + str (culture [0])))
	elite_selection = (
		elite_proporiton (discrimination [1]) if discrimination [0] == 'proportion' else
		panic ('unrecognized discrimination kind: ' + str (discrimination [0])))
	repopulate = (
		naive_mutation_population (reproduction [1], reproduction [2]) if reproduction [0] == 'mutation-only' else
		panic ('unrecognized reproduction kind: ' + str (reproduction [0])))

	ga = thing ()
	ga .params = (
		{ 'algo': 'ga'
		, 'origination': origination
		, 'culture': culture
		, 'discrimination': discrimination
		, 'reproduction': reproduction })

	def evolve (population, habitat, adam = None):
		if population is None:
			yield 'origination'
			population = aborigination (adam)
		yield 'generation'
		yield 'cultivate'
		survivors = cultivate (habitat, population)
		yield 'discriminate'
		elites = yield from elite_selection (survivors)
		yield 'reproduce'
		return repopulate (elites)
	def to (device):
		# move to device
		return ga
	ga .next_generation = evolve
	ga .to = to
	return ga

def random_sampling (number, sd):
	def genesis (agent):
		parameters_origin = { i: torch .zeros (parameter .size ()) for i, parameter in agent .state_dict () }
		agent .load_state_dict (parameters_origin)
		adam = bloodline (agent)

		random = naive_mutation (sd)

		return [ random (adam) for _ in range (number) ]
	return genesis

def hedonistic_culture (habitat, population):
	def life_by_reward (individual):
		env = habitat ()
		reward = live (env, individual)
		individual .fitness = reward
		return individual
	return (life_by_reward (individual) for individual in population)

def elite_proportion (proportion):
	def elites (survivors):
		import math

		elites_size = math .ceil (proportion * len (evolution))
		elites = []
		def rank (elites, fitness):
			if len (elites) == 0:
				return 0
			elif len (elites) == 1:
				if elites [0] .fitness < fitness:
					return 0
				else:
					return 1
			else:
				median = len (elites) // 2
				if elites [median] .fitness < fitness:
					return rank (elites [:median], fitness)
				else:
					return median + rank (elites [median:], fitness)
		for individual in survivors:
			individual_index = rank (elites, individual)
			if individual_index >= elites_size:
				elites = elites [:individual_index] + [ individual ] + elites [individual_index:]
			yield 'discriminating', elites
		yield 'discriminated', elites
		return elites
	return elites
	
def naive_mutation_population (mutation_sd, population_size):
	mutate = naive_mutation (mutation_sd) 
	def next_gen (elites):
		import random

		children_room = population_size - len (elites)
		children = [ mutate (elites [i]) for i in random .sample (xrange (len (elites)), children_room) ]
		return elites + children
		
	return next_gen

def live (env, individual):
	life = individual .reincarnate ()
	total_reward = 0

	alive = True
	observation = env .reset ()
	while alive:
		action = life .choice (observation)
		observation, reward, dead, info = env .step (action)
		yield observation, reward, dead, info
		total_reward += reward
		alive = not dead
		
	life .killed = (reward != 0)
	return total_reward

def naive_mutation (mutation_sd):
	def mutate (individual):
		import copy

		genotype = individual .genotype
		genes = genotype .state_dict ()
		mutated_genes = (
			{ i: chromosome + torch .normal (torch .zeros (chromosome .size ()), mutation_sd)
				for i, chromosome in genes .items () })
		mutated_genotype = copy .deepcopy (genotype)
		mutated_genotype .load_state_dict (mutated_genes)
		return bloodline (mutated_genotype)
	return mutate

def bloodline (genotype):
	import copy

	def reincarnate ():
		life = thing ()
		def choice (observation):
			with torch .no_grad ():
				life .instict .sense (observation)
				return life .instinct (life .instinct .recognition ())
		life .instinct = copy .deepcopy (genotype)
		life .instinct .eval ()
		life .choice = choice
		return life
	it = thing ()
	it .genotype = genotype
	it .reincarnate = reincarnate
	return it

def thing ():
	class thing (dict):
		def __init__(self):
			pass
		def __getattr__(self, attr):
			try:
				return self [attr]
			except:
				return None
		def __setattr__(self, attr, val):
			self [attr] = val
	return thing ()
def panic (reason):
	raise Exception (reason)
