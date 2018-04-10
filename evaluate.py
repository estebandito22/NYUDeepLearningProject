import os
import sys
import json
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

import input.dataloader as loader
import layers.utils as utils
from csal import CSAL 

USE_CUDA = False
	 

def evaluate(dataloader, model, returntype = 'ALL'):
	#for Variables use volatile=True 
	if returntype == 'Bleu':
		return 0
	elif returntype == 'Loss':
		return 0
	return 0, 0

