import sys
import torch
from agent_etc import *
from task_etc import *

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

# load_params = (
# 	{ 'map_location': lambda x, y: 'cpu' } if not torch .cuda .is_available () else
# 	{} )
params = param_splat ([ torch .load (file) for file in sys .argv [1:] ])

agent = (
	load_agent (params) if 'agent' in params else
	None )
task = load_task (params)
task .go (adam = agent)
