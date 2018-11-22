import sys
import torch
from model_etc import *
from task_etc import *
from __.utils import *

params = param_splat ([ torch .load (file) for file in sys .argv [1:] ])
model = load_model (params)
task = load_task (params)
task .go (model)
