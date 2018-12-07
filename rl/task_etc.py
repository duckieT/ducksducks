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
max_steps = 300
iteration_offset = 0
iterations = 1000
parallelism = 2
batch_size = 100
log_interval = 10

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
	{ 'population': [ save_agent_distinct (individual .genotype) for individual in population ]
	, ** save_agent_shared (typical_agent) } )

def evolve_task (map_name, frame_skip, distortion, max_steps, out_path, parallelism, batch_size, log_interval, algo, cuda_ok, iteration_offset, iterations, seed, adam = None, population = None, rng = None, task = None):
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
		, 'batch_size': batch_size
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
		population = task .population
		adam = (
			adam .to (device) if not task .adam is None else
			None )
		survivors = (
			# [ individual .to (device) for individual in population ] if not population is None else
			population if not population is None else
			[ bloodline (original (adam)) ])

		pool_log_file = os .path .join (out_path, 'pool.log')
		sample_log_file = os .path .join (out_path, 'sample.log')
		habitat = parallel_habitat (* 
			[ map_name, parallelism ]
			, batch_size = batch_size
			, frame_skip = frame_skip, distortion = distortion, max_steps = max_steps
			, pool_log_file = pool_log_file, sample_log_file = sample_log_file )

		for iteration in range (iteration_offset, iterations):
			task .iteration_offset = iteration

			# TODO: move models to cuda per batch only
			with co_ (task .algo .evolve (habitat)) as (evolution, send):
				for stage, progress in evolution:
					if stage == 'generation':
						print ('--------------------------------------------------------------------------------')
						print ('Generation: {}' .format (iteration + 1))
						print ('--------------------------------------------------------------------------------')
						send (generation_from (survivors))
					elif stage == 'pre-elites':
						pre_elites_survivors, population_size = progress

						survivors = []
						i = 0
						# TODO: whatif log_interval > batch_size?
						for batch in chunks (pre_elites_survivors, batch_size):
							batch = list (batch)
							for subbatch in chunks (iter (batch), log_interval):
								subbatch = list (subbatch)
								* _, new_survivors = [ desiderata (character) for character in subbatch ]
								i += len (subbatch)
								survivors = new_survivors
								print ('Generation: {} Pre-elites [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, population_size, 100. * i / population_size, survivors [-1] .fitness, survivors [len (survivors) // 2] .fitness, survivors [0] .fitness, sum ([ individual .fitness for individual in survivors ]) / len (survivors)))
						send (generation_from (survivors))
					elif stage == 'elites':
						elites_survivors, population_size = progress

						survivors = []
						i = 0
						for batch in chunks (elites_survivors, batch_size):
							batch = list (batch)
							for subbatch in chunks (iter (batch), log_interval):
								subbatch = list (subbatch)
								* _, new_survivors = [ desiderata (character) for character in subbatch ]
								i += len (subbatch)
								survivors = new_survivors
								print ('Generation: {} Elites [{}/{} ({:.0f}%)]\tMin: {:.6f}\tMedian: {:.6f}\tMax: {:.6f}\tAverage: {:.6f}' .format (iteration + 1, i, population_size, 100. * i / population_size, survivors [-1] .fitness, survivors [len (survivors) // 2] .fitness, survivors [0] .fitness, sum ([ individual .fitness for individual in survivors ]) / len (survivors)))
						send (generation_from (survivors))

			torch .save (save_task (task), file ('task_' + str (iteration + 1) + '.pt'))
			torch .save (save_population (survivors), file ('elites_' + str (iteration + 1) + '.pt'))

			desiderata (sample_visualization (survivors [0], habitat, file ('sample_champion_' + str (iteration + 1) + '.mp4')))
			desiderata (sample_visualization (survivors [1], habitat, file ('sample_first-runner_' + str (iteration + 1) + '.mp4')))
			desiderata (sample_visualization (survivors [2], habitat, file ('sample_second-runner_' + str (iteration + 1) + '.mp4')))
	task .go = go_evolve

	def file (filename):
		import os
		return os .path .join (out_path, filename)
	def chunks (generation, batch_size):
		import itertools
		iterator = iter (generation)
		for first in iterator:
			yield itertools .chain ([first], itertools .islice (iterator, batch_size - 1))

	return task
	
def sample_task (map_name, frame_skip, distortion, max_steps, cuda_ok, log_file, task = None):
	# dont complain when killed
	from signal import signal, SIGPIPE, SIG_DFL
	signal (SIGPIPE, SIG_DFL) 

	habitat = local_habitat (map_name, frame_skip, distortion, max_steps)
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
			
			if command in [ 'shared', 'agent', 'agent-recorded' ]:
				if command == 'shared':
					model_path = rest [0]
					task .model_params = torch .load (model_path)
				else:
					agent_path = rest [0]
					agent_params = torch .load (agent_path)

					genotype = load_agent ({ ** agent_params, ** (task .model_params or {}) })
					individual = bloodline (genotype)
					incarnation = (
						habitat .incarnate (individual) if command == 'agent' else
						habitat .incarnate (individual, record = rest [1]) if command == 'agent-recorded' else
						panic ('explode') )
					life = desiderata (incarnation)
					value = desiderata (life)

					line = job + ':' + str (value)
					lock = open (special_tmp ('lock'), 'w')
					fcntl .lockf (lock, fcntl .LOCK_EX)
					print ('sending\t' + line, file = open (log_file, 'a'))
					print (line)
					lock .close ()
	task .go = go_sample
	return task

def visualize_task (map_name, frame_skip, distortion, max_steps, population, out_path, batch_size, parallelism, iteration_offset, cuda_ok, seed, rng = None, task = None):
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

		population = task .population
		survivors = (
			# [ individual .to (device) for individual in population ] if not population is None else
			population if not population is None else
			[ bloodline (original (adam)) ])

		pool_log_file = os .path .join (out_path, 'pool.log')
		sample_log_file = os .path .join (out_path, 'sample.log')
		habitat = parallel_habitat (* 
			[ map_name, parallelism ]
			, batch_size = batch_size
			, frame_skip = frame_skip, distortion = distortion, max_steps = max_steps
			, pool_log_file = pool_log_file, sample_log_file = sample_log_file )
		iteration = iteration_offset

		desiderata (sample_visualization (survivors [0], habitat, file ('sample_champion_' + str (iteration + 1) + '.mp4')))
		desiderata (sample_visualization (survivors [1], habitat, file ('sample_first-runner_' + str (iteration + 1) + '.mp4')))
		desiderata (sample_visualization (survivors [2], habitat, file ('sample_second-runner_' + str (iteration + 1) + '.mp4')))
	task .go = go_visualize

	def file (filename):
		import os
		return os .path .join (out_path, filename)

	return task

def sample_visualization (bloodline, habitat, path):
	life = yield from habitat .incarnate (bloodline, record = path)
	value = yield from life
	return value

def local_habitat (map_name, frame_skip = None, distortion = None, max_steps = None):
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
	def incarnate (individual, record = None):
		yield 'moment', 'incarnating'
		def life ():
			yield 'moment', 'born'
			try:
				from gym.wrappers.monitoring.video_recorder import VideoRecorder
				with habitat () as env:
					if record: recorder = VideoRecorder (env, record)
					yield 'env', env
					life = yield_ (live (env) (individual))
					for moment in life:
						if record: recorder .capture_frame ()
						yield 'moment', moment
					if record: recorder .close ()
					return life .value
			except AssertionError:
				self = yield from contribution ()
				return self
		return life ()
	habitat .incarnate = incarnate
	return habitat

def parallel_habitat (map_name, parallelism, frame_skip = None, distortion = None, max_steps = None, batch_size = -1, pool_log_file = '/dev/null', sample_log_file = '/dev/null'):
	class habitat ():
		def __enter__ (self):
			panic ('can\'t handle synchronous')
		def __exit__ (self, e_type, e_value, e_traceback):
			panic ('can\'t handle synchronous')

	def incarnate (individual, record = None):
		if habitat .jobs % batch_size == 0:
			if habitat .jobs != 0: habitat .pool .reset ()
			habitat .pool .broadcast_work ('just', 'shared', package (shared = save_agent_shared (individual .genotype)))
		job = next_job ()
		if not record:
			habitat .pool .put_work (job, 'agent', package (agent = save_agent_distinct (individual .genotype)))
		else:
			habitat .pool .put_work (job, 'agent-recorded', package (agent = save_agent_distinct (individual .genotype)), record)
		yield 'moment', 'incarnating'
		return collect (job)
	habitat .incarnate = incarnate

	def next_job ():
		job = 'sample-' + str (habitat .jobs)
		habitat .jobs += 1
		return job
	def collect (_job):
		yield 'moment', 'born'
		for job, result, taken in habitat .pool .get_work ():
			if job == _job:
				taken ()
				return float (result)
		else:
			panic ('something is gravely ill')
	habitat .jobs = 0
	habitat .pool = pool (* 
		[ './go <(./task --task sample' \
			+ (' --log-file ' + sample_log_file if sample_log_file else '') \
			+ (' --map ' + map_name if map_name else '') \
			+ (' --frame-skip ' + str (frame_skip) if frame_skip else '') \
			+ (' --distortion ' + str (distortion) if distortion else '') \
			+ (' --max-steps ' + str (max_steps) if max_steps else '') \
			+ ')'
		, parallelism ]
		, max_jobs = 100000000000
		, log_file = pool_log_file )
	return habitat
