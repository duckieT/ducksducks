import os
import sys
import torch
from agent_etc import *
from algo_etc import *
from __.utils import *

# hyperparameters
tasks = ['evolve', 'sample', 'visualize']
task = 'evolve'
map_name = 'loop_pedestrians'
frame_skip = 3
distortion = True
max_steps = 1500
iteration_offset = 0
iterations = 1000
parallelism = 2
log_interval = 10
batch_size = 50

def load_task (params):
	task_params = params ['task']
	if task_params ['task'] == 'evolve':
		algo = load_algo (params)
		agent = (
			load_agent (params) if 'agent' in params else
			None )
		population = (
			load_population (params) if 'population' in params else
			None )
		return evolve_task (** task_params, algo = algo, adam = agent, population = population)
	elif task_params ['task'] == 'sample':
		return sample_task (** task_params)
	elif task_params ['task'] == 'visualize':
		population = load_population (params)
		return visualize_task (** task_params, population = population)
def save_task (task):
	return (
	{ 'task':
		{ ** task .params
		, 'iteration_offset': task .iteration_offset
		, 'rng': torch .get_rng_state () } 
	, ** save_algo (task .algo) })
# separate into save_task_only?

def load_population (params):
	population_params = params ['population']
	return [ bloodline (load_agent ({ ** agent_params, ** params })) for agent_params in population_params ]
def save_population (population):
	typical_agent = next (iter (population)) .genotype
	return (
	{ 'population': [ save_agent_only (bloodline .genotype) for bloodline in population ]
	, ** (
		save_model (typical_agent .vae) if hasattr (typical_agent, 'vae') else
		{} ) } )

def evolve_task (map_name, out_path, parallelism, iteration_offset, iterations, log_interval, cuda_ok, seed, algo, adam = None, population = None, frame_skip = None, distortion = None, max_steps = None, rng = None, ** kwargs):
	os .makedirs (out_path, exist_ok = True)
	if os .listdir (out_path):
		print ('Warning: ' + out_path + ' is not empty!', file = sys .stderr)

	task = thing ()
	task .params = (
		{ 'task': 'evolve'
		, 'map_name': map_name
		, 'frame_skip': frame_skip
		, 'distortion': distortion
		, 'max_steps': max_steps
		, 'out_path': out_path
		, 'parallelism': parallelism 
		, 'iterations': iterations 
		, 'log_interval': log_interval 
		, 'cuda_ok': cuda_ok 
		, 'seed': seed })
	task .algo = algo
	task .adam = adam
	task .population = population

	def go_evolve ():
		torch .manual_seed (seed)
		if not rng is None:
			torch .set_rng_state (rng)

		device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
		adam = task .adam
		adam = (
			adam .to (device) if not adam is None else
			None )

		pool_log_file = os .path .join (out_path, 'pool.log')
		sample_log_file = os .path .join (out_path, 'sample.log')
		habitat = pool_habitat (map_name, parallelism, frame_skip = frame_skip, distortion = distortion, max_steps = max_steps, pool_log_file = pool_log_file, sample_log_file = sample_log_file)
		population = elites = task .population

		for iteration in range (iteration_offset, iterations):
			task .iteration_offset = iteration

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
					population_size = progress
				elif status == 'reproduce':
					print ('Reproducing...')
					population_size = progress
				elif status == 'discriminating':
					i += 1
					if i % log_interval == 0:
						population = elites = progress
						print ('Generation: {} [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, population_size, 100. * i / population_size, elites [-1] .fitness, elites [len (elites) // 2] .fitness, elites [0] .fitness, sum ([ individual .fitness for individual in elites ]) / len (elites)))
			population = elites = evolution .value
			if i % log_interval != 0:
				print ('Generation: {} [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, population_size, 100. * i / population_size, elites [-1] .fitness, elites [len (elites) // 2] .fitness, elites [0] .fitness, sum ([ individual .fitness for individual in elites ]) / len (elites)))

			torch .save (save_task (task), file ('task_' + str (iteration + 1) + '.pt'))
			torch .save (save_population (elites), file ('elites_' + str (iteration + 1) + '.pt'))

			for moment in yield_ (sample_visualization (elites [0], habitat, file ('sample_champion_' + str (iteration + 1) + '.mp4'))): pass
			for moment in yield_ (sample_visualization (elites [1], habitat, file ('sample_first-runner_' + str (iteration + 1) + '.mp4'))): pass
			for moment in yield_ (sample_visualization (elites [2], habitat, file ('sample_second-runner_' + str (iteration + 1) + '.mp4'))): pass
	task .go = go_evolve

	def file (filename):
		import os
		return os .path .join (out_path, filename)

	return task
	
def sample_visualization (bloodline, habitat, path):
	life = yield from habitat .tryout (bloodline, record = path)
	value = yield from life
	return value

def sample_task (map_name, frame_skip, distortion, max_steps, cuda_ok, log_file, ** kwargs):
	# dont complain when killed
	from signal import signal, SIGPIPE, SIG_DFL
	signal (SIGPIPE, SIG_DFL) 

	habitat = env_habitat (map_name, frame_skip, distortion, max_steps)
	# make this useful
	# device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')
	task = thing ()
	def go_sample ():
		import fcntl
		print ('came alive', file = open (log_file, 'a'))
		for line in sys .stdin:
			line = line .split ('\n') [0]
			print ('received\t' + line, file = open (log_file, 'a'))
			job = line .split (':') [0]
			command = line .split (':') [1]
			rest = line .split (':') [2:]
			
			if command in [ 'model', 'plain', 'record' ]:
				if command == 'model':
					model_path = rest [0]
					task .model_params = torch .load (model_path)
				else:
					agent_path = rest [0]
					agent_params = torch .load (agent_path)
					os .remove (agent_path)
					genotype = load_agent ({ ** agent_params, ** (task .model_params or {}) })
					individual = bloodline (genotype)
					life = yield_ (
						habitat .tryout (individual) if command == 'plain' else
						habitat .tryout (individual, record = rest [1]) if command == 'record' else
						panic ('explode') )
					# for now
					for moment in life: pass
					contribution = yield_ (life .value)
					# for now
					for moment in contribution: pass
					line = job + ':' + str (contribution .value)
					lock = open (special_tmp ('lock'), 'w')
					fcntl .lockf (lock, fcntl .LOCK_EX)
					print ('sending\t' + line, file = open (log_file, 'a'), flush = True)
					print (line, flush = True)
					lock .close ()
	task .go = go_sample
	return task

def visualize_task (map_name, out_path, parallelism, iteration_offset, cuda_ok, seed, population, frame_skip = None, distortion = None, max_steps = None, rng = None, ** kwargs):
	os .makedirs (out_path, exist_ok = True)
	if os .listdir (out_path):
		print ('Warning: ' + out_path + ' is not empty!', file = sys .stderr)

	task = thing ()
	task .params = (
		{ 'task': 'visualize'
		, 'map_name': map_name
		, 'frame_skip': frame_skip
		, 'distortion': distortion
		, 'max_steps': max_steps
		, 'out_path': out_path
		, 'parallelism': parallelism 
		, 'cuda_ok': cuda_ok 
		, 'seed': seed })
	task .population = population

	def go_visualize ():
		torch .manual_seed (seed)
		if not rng is None:
			torch .set_rng_state (rng)

		device = torch .device ('cuda') if cuda_ok else torch .device ('cpu')

		pool_log_file = os .path .join (out_path, 'pool.log')
		sample_log_file = os .path .join (out_path, 'sample.log')
		habitat = pool_habitat (map_name, parallelism, frame_skip = frame_skip, distortion = distortion, max_steps = max_steps, pool_log_file = pool_log_file, sample_log_file = sample_log_file)
		population = elites = task .population
		iteration = iteration_offset

		for moment in yield_ (sample_visualization (elites [0], habitat, file ('sample_champion_' + str (iteration + 1) + '.mp4'))): pass
		for moment in yield_ (sample_visualization (elites [1], habitat, file ('sample_first-runner_' + str (iteration + 1) + '.mp4'))): pass
		for moment in yield_ (sample_visualization (elites [2], habitat, file ('sample_second-runner_' + str (iteration + 1) + '.mp4'))): pass
	task .go = go_visualize

	def file (filename):
		import os
		return os .path .join (out_path, filename)

	return task
def env_habitat (map_name, frame_skip = None, distortion = None, max_steps = None):
	class habitat ():
		def __init__ (self):
			from gym_duckietown.envs.duckietown_env import DuckietownEnv
			args = { 'map_name': map_name, 'frame_skip': frame_skip, 'distortion': distortion, 'max_steps': max_steps }
			self .env = DuckietownEnv (** { k: v for k, v in args .items () if not v is None })
		def __enter__ (self):
			self .occupied = True
			return self .env
		def __exit__ (self, e_type, e_value, e_traceback):
			self .occupied = False
	def tryout (individual, record = None):
		yield 'moment', 'fake'
		def contribution ():
			yield 'moment', 'starting'
			try:
				from gym.wrappers.monitoring.video_recorder import VideoRecorder
				with habitat () as env:
					if record: recorder = VideoRecorder (env, record)
					yield 'env', env
					life = yield_ (live (env, individual))
					for moment in life:
						if record: recorder .capture_frame ()
						yield 'moment', moment
					if record: recorder .close ()
					return life .value
			except AssertionError:
				self = yield from contribution ()
				return self
		return contribution ()
	habitat .tryout = tryout
	return habitat

def pool_habitat (map_name, parallelism, frame_skip = None, distortion = None, max_steps = None, pool_log_file = '/dev/null', sample_log_file = '/dev/null'):
	class habitat ():
		def __enter__ (self):
			panic ('can\'t handle synchronous')
		def __exit__ (self, e_type, e_value, e_traceback):
			panic ('can\'t handle synchronous')

	def tryout (individual, record = None):
		if habitat .jobs == habitat .jobs_offset:
			if hasattr (individual .genotype, 'vae'):
				habitat .pool .broadcast_work ('just', 'model', package (model = save_model (individual .genotype .vae)))
		job = next_job ()
		if not record:
			habitat .pool .put_work (job, 'plain', package (agent = save_agent_only (individual .genotype)))
		else:
			habitat .pool .put_work (job, 'record', package (agent = save_agent_only (individual .genotype)), record)
		yield 'moment', 'fake life'
		return collect (job)
	def reset ():
		habitat .pool .reset ()
		habitat .jobs_offset = habitat .jobs
	habitat .tryout = tryout
	habitat .reset = reset

	def next_job ():
		job = 'sample-' + str (habitat .jobs)
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
	habitat .jobs_offset = 0
	habitat .pool = pool (* 
		[ './go <(./task --task sample' \
			+ (' --log-file ' + sample_log_file if sample_log_file else '') \
			+ (' --map ' + map_name if map_name else '') \
			+ (' --frame-skip ' + str (frame_skip) if frame_skip else '') \
			+ (' --distortion ' + str (distortion) if distortion else '') \
			+ (' --max-steps ' + str (max_steps) if max_steps else '') \
			+ ')'
		, parallelism ]
		, max_jobs = 100000
		, log_file = pool_log_file )
	return habitat
