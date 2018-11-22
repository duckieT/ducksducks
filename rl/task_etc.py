import os
import sys
import torch
from agent_etc import *
from algo_etc import *

# hyperparameters
tasks = ['evolve']
task = 'evolve'
env_name = 'Duckietown-loop_pedestrians-v0'
iteration_offset = 0
iterations = 1000
parallelism = 1
log_interval = 10

def load_task (params):
	task_params = params ['task']
	if task_params ['task'] == 'evolve':
		algo = load_algo (params)
		population = (
			load_population (params) if 'population' in params else
			None )
		return init_evolve (** task_params, algo = algo, population = population)
def save_task (task):
	return (
	{ 'task':
		{ ** task .params
		, 'iteration_offset': task .iteration_offset
		, 'rng': torch .get_rng_state () } 
	, ** save_algo (task .algo) })

def load_population (params):
	population_params = params ['population']
	return [ bloodline (load_agent ({ ** agent_params, ** params })) for agent_params in population_params ]
def save_population (population):
	return (
	{ 'population': [ save_agent_only (bloodline .genotype) for bloodline in population ]
	, ** save_model (next (iter (population)) .genotype .vae) })

def init_evolve (env_name, out_path, parallelism, iteration_offset, iterations, log_interval, cuda_ok, seed, algo, rng = None, population = None, ** kwargs):
	os .makedirs (out_path, exist_ok = True)
	if os .listdir (out_path):
		print ('Warning: ' + out_path + ' is not empty!', file = sys .stderr)

	task = thing ()
	task .params = (
		{ 'task': 'evolve'
		, 'env_name': env_name
		, 'out_path': out_path
		, 'parallelism': parallelism 
		, 'iterations': iterations 
		, 'log_interval': log_interval 
		, 'cuda_ok': cuda_ok 
		, 'seed': seed })
	task .algo = algo
	task .population = population
	def file (filename):
		import os
		return os .path .join (out_path, filename)

	def go_evolve (adam = None):
		torch .manual_seed (seed)
		if not rng is None:
			torch .set_rng_state (rng)

		device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
		adam = (
			adam .to (device) if not adam is None else
			None )

		habitat = pooled_habitat (env_name, parallelism)

		for iteration in range (iteration_offset, iterations):
			i = 0
			evolution = yield_ (task .algo .next_generation (task .population, habitat, adam = adam))
			# TODO: add multiprocessing (aka process parallelism)
			# TODO: add visualization
			for status, progress in evolution:
				if status == 'generation':
					print ('--------------------------------------------------------------------------------')
					print ('Generation: {}' .format (iteration + 1))
					print ('--------------------------------------------------------------------------------')
				elif status == 'origination':
					print ('Initializing first generation...')
				elif status == 'reproduce':
					print ('Reproducing...')
				elif status == 'originated':
					task .population = population = progress
				elif status == 'reproduced':
					task .population = population = progress
				elif status == 'discriminating':
					i += 1
					if i % log_interval == 0:
						task .elites = elites = progress
						print ('Generation: {} [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, len (population), 100. * i / len (population), elites [-1] .fitness, elites [len (elites) // 2] .fitness, elites [0] .fitness, sum ([ individual .fitness for individual in elites ]) / len (elites)))
			task .population = population = evolution .value
			task .iteration_offset = iteration + 1
			# write_image (comparison_visualization, file ('comparison_' + str (iteration + 1) + '.png'))
			# write_image (sampling_visualization, file ('sampling_' + str (iteration + 1) + '.png'))
			torch .save (save_task (task), file ('task_' + str (iteration + 1) + '.pt'))
			torch .save (save_population (elites), file ('elites_' + str (iteration + 1) + '.pt'))
			with habitat () as env:
				sample_visualization (elites [0], env, file ('sample_' + str (iteration + 1) + '.mp4'))
	task .go = go_evolve
	return task

def env_habitat (env_name):
	class habitat ():
		def __init__ (self):
			import gym
			import gym_duckietown
			self .env = gym .make (env_name)
		def __enter__ (self):
			self .occupied = True
			return self .env
		def __exit__ (self, e_type, e_value, e_traceback):
			self .occupied = False
	return habitat

def pooled_habitat (env_name, parallelism):
	pool = thing ()
	pool .habitats = []
	new_habitat = env_habitat (env_name)
	class pooled_habitat ():
		def __enter__ (self):
			if len (pool .habitats) < parallelism:
				self .habitat = new_habitat ()
				pool .habitats += [ self .habitat ]
			else:
				for habitat in pool .habitats:
					if not habitat .occupied:
						self .habitat = habitat
						break
				else:
					panic ('habitat pool exhausted!')
					return # TODO: block until have habitat
			return self .habitat .__enter__ ()
		def __exit__ (self, e_type, e_value, e_traceback):
			self .habitat .__exit__ (e_type, e_value, e_traceback)
	return pooled_habitat

def sample_visualization (bloodline, env, path):
	from gym.wrappers.monitoring.video_recorder import VideoRecorder
	recorder = VideoRecorder (env, path)
	for moment in live (env, bloodline): recorder .capture_frame ()
	recorder .close ()

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
