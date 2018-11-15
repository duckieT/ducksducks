import sys
import torch
from model_etc import *
from task_etc import *

class param ():
	def __getitem__ (self, item):
		try: return getattr (self, item)
		except: raise Exception (item + ' is not provided!')
	def __setitem__ (self, item, val):
		setattr (self, item, val)
def param_splat (params):
	x = param ()
	for splat in params:
		for key in splat: x [key] = splat [key]
	return x

params = param_splat ([ torch .load (file) for file in sys .argv [1:] ])
model = load_model (params)
task = load_task (params)
task .go (model)
