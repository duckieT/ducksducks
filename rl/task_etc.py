import os
import sys
import torch
from agent_etc import *
from algo_etc import *
from __.utils import *

# hyperparameters
tasks = ['evolve', 'experience']
task = 'evolve'
map_name = 'loop_pedestrians'
frame_skip = 3
distortion = True
iteration_offset = 0
iterations = 1000
parallelism = 2
log_interval = 10

def load_task (params):
	task_params = params ['task']
	if task_params ['task'] == 'evolve':
		algo = load_algo (params)
		population = (
			load_population (params) if 'population' in params else
			None )
		return init_evolve (** task_params, algo = algo, population = population)
	elif task_params ['task'] == 'experience':
		return init_experience (** task_params)
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

		habitat = (
			env_habitat (map_name, frame_skip = frame_skip, distortion = distortion) if parallelism == 1 else
			pool_habitat (map_name, parallelism, frame_skip = frame_skip, distortion = distortion))
		population = task .population

		for iteration in range (iteration_offset, iterations):
			i = 0
			evolution = yield_ (task .algo .next_generation (population, habitat, adam = adam))
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
			for moment in yield_ (sample_visualization (elites [0], habitat, file ('sample_' + str (iteration + 1) + '.mp4'))): pass
	task .go = go_evolve

	def file (filename):
		import os
		return os .path .join (out_path, filename)

	return task
	
def sample_visualization (bloodline, habitat, path):
	life = yield from habitat .eval (bloodline, record = path)
	value = yield from life
	return value

def init_experience (map_name, frame_skip, distortion, cuda_ok, ** kwargs):
	# dont complain when killed
	from signal import signal, SIGPIPE, SIG_DFL
	signal (SIGPIPE ,SIG_DFL) 

	habitat = env_habitat (map_name, frame_skip, distortion)
	# make this useful
	# device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
	task = thing ()
	def go_experience (log_file = '/dev/null'):
		for line in sys .stdin:
			line = line .split ('\n') [0]
			print ('received\t' + line, file = open (log_file, 'a'))
			job = line .split (':') [0]
			command = line .split (':') [1]
			rest = line .split (':') [2:]
			
			if command in [ 'experience', 'record' ]:
				agent_path = rest [0]
				genotype = load_agent (torch .load (agent_path))
				individual = bloodline (genotype)
				os .remove (agent_path)
				if command == 'experience':
					life = yield_ (habitat .eval (individual))
					# for now
					for moment in life: pass
					contribution = yield_ (life .value)
					# for now
					for moment in contribution: pass
					line = job + ':' + str (contribution .value)
					print ('sending\t' + line, file = open (log_file, 'a'))
					print (line)
				elif command == 'record':
					record_path = rest [1]
					life = yield_ (sample_visualization (individual, habitat, record_path))
					# for now
					for moment in life: pass
					line = job + ':' + str (life .value)
					print ('sending\t' + line, file = open (log_file, 'a'))
					print (line)
	task .go = go_experience
	return task

def env_habitat (map_name, frame_skip = None, distortion = None):
	class habitat ():
		def __init__ (self):
			from gym_duckietown.envs.duckietown_env import DuckietownEnv
			args = { 'map_name': map_name, 'frame_skip': frame_skip, 'distortion': distortion }
			self .env = DuckietownEnv (** { k: v for k, v in args .items () if not v is None })
		def __enter__ (self):
			self .occupied = True
			return self .env
		def __exit__ (self, e_type, e_value, e_traceback):
			self .occupied = False
	def eval (individual, record = None):
		yield 'moment', None
		def contribution ():
			from gym.wrappers.monitoring.video_recorder import VideoRecorder
			with habitat () as env:
				if record: recorder = VideoRecorder (env, record)
				if record: recorder .capture_frame ()
				yield 'env', env
				life = yield_ (live (env, individual))
				for moment in life:
					if record: recorder .capture_frame ()
					yield 'moment', moment
				if record: recorder .close ()
			return life .value
		return contribution ()
	habitat .eval = eval
	return habitat

def pool_habitat (map_name, parallelism, frame_skip = None, distortion = None):
	class habitat ():
		def __enter__ (self):
			panic ('can\'t handle synchronous')
		def __exit__ (self, e_type, e_value, e_traceback):
			panic ('can\'t handle synchronous')

	def eval (individual, record = None):
		job = next_job ()
		if not record:
			habitat .pool .put_work (job, 'experience', individual)
		else:
			habitat .pool .put_work (job, 'record', individual, record)
		yield 'moment', 'fake life'
		return collect (job)
	habitat .eval = eval

	def next_job ():
		job = 'job-' + str (habitat .jobs)
		habitat .jobs += 1
		return job
	def collect (_job):
		yield 'moment', 'fake life'
		for job, result, taken in habitat .pool .get_work ():
			if job == _job:
				taken ()
				return float (result)
		else:
			panic ('something is gravely ill')
	habitat .jobs = 0
	habitat .pool = pool (* 
		[ parallelism,
		{ 'task': 'experience'
		, 'map': map_name
		, 'frame_skip': frame_skip
		, 'distortion': distortion } ] )
	return habitat
