import os
import random

def on_killed (fn):
	import sys
	import atexit
	from signal import signal, SIGTERM, SIGINT
	def do (* args):
		fn ()
		sys .exit ()
	atexit .register (do)
	signal (SIGTERM, do)
	signal (SIGINT, do)

def bash (cmd):
	escaped_cmd = '\'\'\\\'' .join (cmd .split ('\''))
	just_say (cmd)
	return os .system ('bash -c \'' + escaped_cmd + '\'')

tmp_dir = '/dev/shm/'
static_tmp_prefix = os .path .join (tmp_dir, 'pool-')

def tmp (for_what = None):
	if not hasattr (tmp, 'prefix'):
		tmp .prefix = os .path .join (tmp_dir, 'pool_' + str (random .getrandbits (32)))
		while os .path .exists (tmp .prefix):
			tmp .prefix = os .path .join (tmp_dir, 'pool_' + str (random .getrandbits (32)))
		os .mknod (tmp .prefix)
	marking = (
		'-' + for_what + '-' if not for_what is None else
		'-' )
	tmp_path = tmp .prefix + marking + str (random .getrandbits (32))
	if os .path .exists (tmp_path):
		return tmp ()
	else:
		return tmp_path
def special_tmp (special):
	return static_tmp_prefix + special
def package (goods = None, ** kwargs):
	num_of_goods = ( 0 if goods is None else 1 ) + len (kwargs)
	if num_of_goods != 1:
		panic ('ur doin it wrong bruh')
	import torch
	package = (
		tmp () if not goods is None else
		tmp (next (iter (kwargs .keys ()))) )
	if goods is None:
		goods = next (iter (kwargs .values ())) 
	just_say ('packaging ' + str (package))
	os .makedirs (package), os .rmdir (package), torch .save (goods, package)
	return package
def clean_tmp ():
	if hasattr (tmp, 'prefix'):
		bash ('find "' + tmp_dir + '" | grep "$(basename "' + tmp .prefix + '")" | xargs -I {} rm {}')
on_killed (clean_tmp)

def pool (command, parallelism, max_jobs = None, log_file = '/dev/null'):
	max_jobs = if_none (10 * parallelism, max_jobs)
	
	it = thing ()
	it .send = []
	it .send_fps = {}
	it .send_fds = {}
	it .jobs = []

	rcv = tmp ('rcv')
	os .makedirs (rcv), os .rmdir (rcv), os .mkfifo (rcv)
	it .rcv = rcv

	for i in range (parallelism):
		send = tmp ('send')
		os .makedirs (send), os .rmdir (send), os .mkfifo (send)
		it .send += [send]
		bash (command + ' <"' + send + '" >>"' + rcv + '" &')
		it .jobs += [{}]

	def clean_pool ():
		import psutil
		pids = psutil .Process () .children (recursive = True)
		for pid in pids:
			os .kill (pid .pid, signal)
	on_killed (clean_pool)
		
	def running_jobs ():
		return sum ([ len ([ job for job in jobs .values () if job is None ]) for jobs in it .jobs ])
	def spare (jobs):
		busiest = max ([len (job) for job in jobs])
		for i, job in enumerate (jobs):
			if len (job) != busiest:
				return i
		else:
			return 0
	
	def reset ():
		import psutil
		import time
		if sum ([ len (job) for job in it .jobs ]) == 0:
			pids = psutil .Process () .children (recursive = True)
			for pid in pids:
				os .kill (pid .pid, signal)
			if hasattr (tmp, 'prefix'):
				bash ('find "' + tmp_dir + '" | grep "$(basename "' + tmp .prefix + '-")" | grep -v "rcv" | xargs -I {} rm {}')
			for i in range (parallelism):
				send = it .send [i]
				if i in it .send_fds:
					it .send_fds [i] .close ()
				it .send_fds = {}
				it .send_fps = {}
				os .mkfifo (send)
				bash (command + ' <"' + send + '" >>"' + rcv + '" &')
		else:
			panic ('jay doesnt care yet')
		
	def assign_work (n, job, * order):
		import time
		if job != 'just':
			it .jobs [n] = { ** it .jobs [n], job: None }
		# what if job already exists?
		while not n in it .send_fps:
			try:
				it .send_fps [n] = os .open (it .send [n], os .O_WRONLY | os .O_NONBLOCK)
			except OSError:
				just_say ('waiting for worker ' + str (n) + ' to open pipe...')
				time .sleep (1)
		if not n in it .send_fds:
			it .send_fds [n] = os .fdopen (it .send_fps [n], 'w', 1)
		line = ':' .join ([job, * order ])
		print ('sending\t' + line, file = open (log_file, 'a'))
		print (line, file = it .send_fds [n])
		it .send_fds [n] .flush ()
		
		if running_jobs () > max_jobs:
			just_say ('pausing pool, # of jobs = ' + str (running_jobs) + ', > max_jobs = ' + str (max_jobs))
			for _ in get_work ():
				if running_jobs () > max_jobs:
					continue
				else:
					break
			else:
				panic ('something very grave has happened')
			
	def put_work (job, * order):
		n = spare (it .jobs)
		assign_work (n, job, * order)
	def broadcast_work (job, * order):
		# ONLY works for job == 'just'! other jobs will cause a mysterious job panic!
		for n in range (parallelism):
			assign_work (n, job, * order)
	def get_work ():
		def complete (job):
			def record_done ():
				done_job (job)
			return record_done
		for job, result in { job: result for jobs in it .jobs for job, result in jobs .items () if not result is None } .items ():
			yield job, result, complete (job)
		if it .rcv_fp is None:
			it .rcv_fd = open (it .rcv, 'r', 1)
			it .rcv_fp = it .rcv_fd .__enter__ ()
		for line in it .rcv_fp:
			line = line .split ('\n') [0]
			print ('received\t' + line, file = open (log_file, 'a'))
			job, result = line .split (':')
			collect_job (job, result)
			yield job, result, complete (job)
	def collect_job (job, result):
		for i, jobs in enumerate (it .jobs):
			if job in jobs:
				it .jobs [i] [job] = result
				break
		else:
			panic ('mysterious job ' + str (job))
	def done_job (job):
		for i, jobs in enumerate (it .jobs):
			if job in jobs:
				it .jobs [i] = { j: it .jobs [i] [j] for j in jobs if j != job }
				break
		else:
			panic ('mysterious job ' + str (job))
				
	it .reset = reset

	it .broadcast_work = broadcast_work
	it .put_work = put_work
	it .get_work = get_work
	return it





def if_none (default_, value):
	return default_ if value == None else value
def just_say (text):
	import os
	tty = os .fdopen (os .open ('/dev/tty', os .O_WRONLY | os .O_NOCTTY), 'w', 1)
	print (text, file = tty)





class param ():
	def __init__ (self):
		self .dict = {}
	def __iter__ (self):
		return iter (self .dict)
	def __contains__ (self, item):
		return item in self .dict
	def __getitem__ (self, item):
		try: return self .dict [item]
		except: pass
		raise Exception (str (item) + ' is not provided!')
	def __setitem__ (self, item, val):
		self .dict [item] = val
	def keys (self):
		return self .dict .keys ()
def param_splat (params):
	x = param ()
	for splat in params:
		for key in splat: x [key] = splat [key]
	return x





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
class co_ ():
	def __init__ (self, gen):
		self .gen = gen
		self .sending = False
	def __enter__ (self):
		def get ():
			if self .sending:
				self .sending = False
				return self .gen .send (self .val)
			else:
				return next (self .gen)
		def iterator ():
			try:
				while self .gen: yield get ()
			except StopIteration: pass
		def put (x):
			self .sending = True
			self .val = x
		return iterator (), put
	def __exit__ (self, x, y, z):
		self .gen .close ()
def panic (reason):
	raise Exception (reason)
