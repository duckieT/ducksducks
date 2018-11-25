import sys
import torch
from agent_etc import *
from task_etc import *
from __.utils import *

# load_params = (
# 	{ 'map_location': lambda x, y: 'cpu' } if not torch .cuda .is_available () else
# 	{} )
params = param_splat ([ torch .load (file) for file in sys .argv [1:] ])

task = load_task (params)
task .go ()
