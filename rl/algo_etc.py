import torch
import numpy as np
from torch import nn
from torch .nn import functional as F

# hyperparameters

algos = ['ga']
algo = 'ga'
elite_proportion = 0.1
mutation_sd = 1
population_size = 100
	
def load_algo (params):
	algo_params = params ['algo']
	if algo_params ['algo'] == 'ga':
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
		proportional_selection (discrimination [1]) if discrimination [0] == 'proportion' else
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
		yield 'generation', None
		if population is None:
			yield 'origination', None
			population = aborigination (adam)
			yield 'originated', population
		else:
			yield 'reproduce', None
			population = repopulate (population)
			yield 'reproduced', population
		yield 'cultivate', None
		survivors = cultivate (habitat, population)
		yield 'cultivated', survivors
		yield 'discriminate', None
		elites = yield from elite_selection (survivors, population)
		yield 'discriminated', elites
		return elites
	ga .next_generation = evolve
	return ga

def random_sampling (number, sd):
	def genesis (agent):
		parameters_origin = { i: torch .zeros (parameter .size ()) for i, parameter in agent .state_dict () .items () }
		agent .load_state_dict (parameters_origin)
		adam = bloodline (agent)

		random = naive_mutation (sd)

		return [ random (adam) for _ in range (number) ]
	return genesis

def hedonistic_culture (habitat, population):
	def life_by_reward (individual):
		with habitat () as env:
			life = yield_ (live (env, individual))
			for moment in life: pass
			reward = life .value
			individual .fitness = reward
			return individual
	return (life_by_reward (individual) for individual in population)

def proportional_selection (proportion):
	def elites (survivors, population):
		import math

		elites_size = math .ceil (proportion * len (population))
		elites = []
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
		for individual in survivors:
			individual_index = rank (elites, individual)
			if individual_index < elites_size:
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
		children = [ mutate (elites [i]) for i in random .choices (range (len (elites)), k = children_room) ]
		return elites + children
		
	return next_gen

def live (env, individual):
	life = individual .reincarnate ()
	total_reward = 0

	alive = True
	observation = env .reset ()
	observation = torch .from_numpy (observation) .to (dtype = torch.float32) .permute (2, 0, 1)
	while alive:
		action = life .choice (observation)
		observation, reward, dead, info = env .step (action)
		observation = torch .from_numpy (observation) .to (dtype = torch.float32) .permute (2, 0, 1)
		yield 'step', (observation, reward, dead, info)
		total_reward += reward
		alive = not dead
		
	life .killed = (reward != 0)
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
	import copy

	device = next (genotype .parameters ()) .device
	def reincarnate ():
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
class yield_:
	def __init__ (self, gen):
		self .gen = gen
	def __iter__ (self):
		self .value = yield from self .gen
def panic (reason):
	raise Exception (reason)
