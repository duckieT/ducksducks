import os
import sys
import torch
from agent_etc import *
from algo_etc import *

# hyperparameters
tasks = ['evolve']
task = 'evolve'
map_name = 'loop_pedestrians'
frame_skip = 3
distortion = True
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

def init_evolve (map_name, frame_skip, distortion, out_path, parallelism, iteration_offset, iterations, log_interval, cuda_ok, seed, algo, rng = None, population = None, ** kwargs):
	os .makedirs (out_path, exist_ok = True)
	if os .listdir (out_path):
		print ('Warning: ' + out_path + ' is not empty!', file = sys .stderr)

	task = thing ()
	task .params = (
		{ 'task': 'evolve'
		, 'map_name': map_name
		, 'frame_skip': frame_skip
		, 'distortion': distortion
		, 'out_path': out_path
		, 'parallelism': parallelism 
		, 'iterations': iterations 
		, 'log_interval': log_interval 
		, 'cuda_ok': cuda_ok 
		, 'seed': seed })
	task .algo = algo
	task .population = population

	def go_evolve (adam = None):
		torch .manual_seed (seed)
		if not rng is None:
			torch .set_rng_state (rng)

		device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
		adam = (
			adam .to (device) if not adam is None else
			None )

		habitat = pool_habitat (map_name, parallelism, frame_skip = frame_skip, distortion = distortion)
		population = task .population

		for iteration in range (iteration_offset, iterations):
			i = 0
			evolution = yield_ (task .algo .next_generation (population, habitat, adam = adam))
			# TODO: add multiprocessing (aka process parallelism)
			# TODO: move models to cuda per batch only
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
					population = progress
				elif status == 'reproduced':
					population = progress
				elif status == 'discriminating':
					i += 1
					if i % log_interval == 0:
						elites = progress
						print ('Generation: {} [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, len (population), 100. * i / len (population), elites [-1] .fitness, elites [len (elites) // 2] .fitness, elites [0] .fitness, sum ([ individual .fitness for individual in elites ]) / len (elites)))
			population = evolution .value
			task .iteration_offset = iteration + 1
			torch .save (save_task (task), file ('task_' + str (iteration + 1) + '.pt'))
			torch .save (save_population (elites), file ('elites_' + str (iteration + 1) + '.pt'))
			sample_visualization (elites [0], habitat, file ('sample_' + str (iteration + 1) + '.mp4'))
	task .go = go_evolve

	def file (filename):
		import os
		return os .path .join (out_path, filename)

	return task

def env_habitat (map_name, frame_skip = None, distortion = None):
	class habitat ():
		def __init__ (self):
			# import gym
			# import gym_duckietown
			# self .env = gym .make (env_name)
			from gym_duckietown.envs.duckietown_env import DuckietownEnv
			args = { 'map_name': map_name, 'frame_skip': frame_skip, 'distortion': distortion }
			self .env = DuckietownEnv (** { k: v for k, v in args .items () if not v is None })
		def __enter__ (self):
			self .occupied = True
			return self .env
		def __exit__ (self, e_type, e_value, e_traceback):
			self .occupied = False
		def find (self, then):
			job = thing ()
			def get ():
				with self as env:
					return then (env)
			job .get = get
			return job
	return habitat

def pool_habitat (map_name, parallelism, ** args):
	import torch.multiprocessing as mp
	jobs = mp .Queue ()
	returns = mp .Queue ()
	pool = [ mp .Process (target = __pooling, args = (map_name, args, jobs, returns)) .start () for i in range (parallelism) ]

	class pool_habitat ():
		def __enter__ (self):
			panic ('jay has not figured out what to do')
		def __exit__ (self, e_type, e_value, e_traceback):
			panic ('jay has not figured out what to do')
		def find (self, then, info):
			import cloudpickle
			jobs .put (cloudpickle .dumps ((then, info)))
			return returns
			# hack: this relies on being called in the same order
	return pool_habitat

class __pooled_habitat ():
	def __init__ (self, map_name, ** args):
		self .local_habitat = None
		self .new_habitat = env_habitat (map_name, ** args)
	def __enter__ (self):
		if self .local_habitat is None:
			self .local_habitat = self .new_habitat ()
		return self .local_habitat .__enter__ ()
	def __exit__ (self, e_type, e_value, e_traceback):
		self .local_habitat .__exit__ (e_type, e_value, e_traceback)
	def find (self, then):
		job = thing ()
		def get ():
			with self as env:
				return then (env)
		job .get = get
		return job

def __pooling (map_name, args, jobs, returns):
	import pickle
	local_habitat = __pooled_habitat (map_name, ** args)
	while True:
		_then, info = pickle .loads (jobs .get ())
		with local_habitat as env:
			returns .put (_then (env, * info))
	

def sample_visualization (bloodline, habitat, path):
	with habitat () as env:
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
