def pool (parallelism, args, tmp_dir = '/dev/shm/', log_file = '/dev/null'):
	import os
	os .system ('bash -c \'rm /dev/shm/pool-* 2>/dev/null\'')
	def tmp ():
		import random
		tmp_path = os .path .join (tmp_dir, 'pool-' + str (random .getrandbits (32)))
		if os .path .exists (tmp_path):
			return tmp ()
		else:
			return tmp_path

	def spare (jobs):
		busiest = max ([len (job) for job in jobs])
		for i, job in enumerate (jobs):
			if len (job) != busiest:
				return i
		else:
			return 0
	
	it = thing ()
	it .send = []
	it .send_fps = {}
	it .send_fds = {}
	it .jobs = []

	rcv = tmp ()
	os .makedirs (rcv), os .rmdir (rcv), os .mkfifo (rcv)
	it .rcv = rcv

	pretty_args = ' ' .join ('--' + '-' .join (key .split ('_')) + ' ' + str (val) for key, val in args .items ())
	for i in range (parallelism):
		send = tmp ()
		os .makedirs (send), os .rmdir (send), os .mkfifo (send)
		it .send += [send]
		os .system ('bash -c \'./go <(./task ' + pretty_args + ') <"' + send + '" >> "' + rcv + '" &\'')
		it .jobs += [{}]
		
	def assign_work (n, job, * command):
		it .jobs [n] = { ** it .jobs [n], job: None }
		if not n in it .send_fps:
			it .send_fps [n] = os .open (it .send [n], os .O_WRONLY | os .O_NONBLOCK)
		if not n in it .send_fds:
			it .send_fds [n] = os .fdopen (it .send_fps [n], 'w', 1)
		line = ':' .join ([job, * command ])
		print ('sending\t' + line, file = open (log_file, 'a'))
		print (line, file = it .send_fds [n])
		it .send_fds [n] .flush ()
	def put_work (job, command, individual, * rest):
		import torch
		from __.rl.agent_etc import save_agent

		n = spare (it .jobs)
		agent = tmp ()
		os .makedirs (agent), os .rmdir (agent)
		# change to only send agent (aka no model)
		torch .save (save_agent (individual .genotype), agent)
		assign_work (n, job, command, agent, * rest)
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
			record = { 'done': False }
			def complete ():
				record ['done'] = True
			yield job, result, complete
			if not record ['done']:
				record_job (job, result)
	def record_job (job, result):
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
				
	it .assign_work = assign_work
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
def panic (reason):
	raise Exception (reason)
