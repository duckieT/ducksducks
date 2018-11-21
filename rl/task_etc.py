import os
import sys
import torch
from algo_etc import *

# hyperparameters
tasks = ['evolve']
task = 'evolve'
iteration_offset = 0
iterations = 1000
batch_size = 20
log_interval = 10

def load_task (params):
	task_params = params ['task']
	if task_params ['task'] == 'evolve':
		algo = load_algo (params)
		return init_evolve (** task_params, algo = algo)
def save_task (task):
	return (
	{ 'task': { ** task .params, 'iteration_offset': task .iteration_offset, 'rng': torch .get_rng_state () } 
	, ** save_algo (task .algo) })

def load_population (params):
	population_params = params ['population']
	return [ load_agent ({ agent_params }) for agent_params in popuulation_params ]
def save_population (population):
	return (
	{ 'population': [ save_agent (agent) for agent in population ] })

def init_evolve (env_path, out_path, batch_size, iteration_offset, iterations, log_interval, cuda_ok, seed, algo, rng = None, population = None, ** kwargs):
	os .makedirs (out_path, exist_ok = True)
	if os .listdir (out_path):
		print ('Warning: ' + out_path + ' is not empty!', file = sys .stderr)

	task = thing ()
	task .params = (
		{ 'task': 'evolve'
		, 'env_path': env_path
		, 'out_path': out_path
		, 'batch_size': batch_size 
		, 'iterations': iterations 
		, 'log_interval': log_interval 
		, 'cuda_ok': cuda_ok 
		, 'seed': seed })
	task .algo = algo
	def file (filename):
		import os
		return os .path .join (out_path, filename)
	def go_evolve (population, adam = None):
		torch .manual_seed (seed)
		if not rng is None:
			torch .set_rng_state (rng)

		device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
		algo = algo .to (device)

		for iteration in range (iteration_offset, iterations):
			# comparison_visualization, sampling_visualization = None, None
			i = 0
			evolution = yield_ (algo .next_generation ())
			elites = None
			for status, progress in evolution:
				if status == 'generation':
					print ('Generation: {}' .format (iteration + 1))
				elif status == 'discriminating':
					if i != 0 and i % log_interval == 0:
						elites = progress
						print ('Generation: {} [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, len (algo .population), 100. * i / len (algo .population), elites [-1] .fitness, elites [len (elites) // 2] .fitness, elites [0] .fitness, sum ([ individual .fitness for individual in elites ]) / len (elites)))
					i += 1
				elif status == 'discriminated':
					elites = progress
				elif status == 'visualization':
					pass
					# comparison_visualization, sampling_visualization = progress
			population = evolution .value
			task .iteration_offset = iteration + 1
			# write_image (comparison_visualization, file ('comparison_' + str (iteration + 1) + '.png'))
			# write_image (sampling_visualization, file ('sampling_' + str (iteration + 1) + '.png'))
			torch .save (save_task (task), file ('task_' + str (iteration + 1) + '.pt'))
			torch .save (elites, file ('elites_' + str (iteration + 1) + '.pt'))
			torch .save (population, file ('population_' + str (iteration + 1) + '.pt'))
	task .go = go_evolve
	return task

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
