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
import evaluate as evaluator

USE_CUDA = False

def test(modelname, modelfilename):
	modeldict = open(os.path.join('models/', modelfilename), 'rb')
	checkpoint = torch.load(modeldict)
	dict_args = checkpoint['dict_args']
	if modelname == 'csal':
		model = CSAL(dict_args)
	model.load_state_dict(checkpoint['state_dict'])

	vocabfile = open(os.path.join('models/', 'vocab.pkl'), 'rb')
	vocab = pickle.load(vocabfile)
	glovefile = open(os.path.join('models/', 'glove.pkl'), 'rb')
	glove = pickle.load(glovefile)

	test_batch_size = 8
	file_names = [('MSRVTT/captions.json', 'MSRVTT/testvideo.json', 'MSRVTT/Frames')]
	files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
	test_dataloader = loader.get_test_data(files, vocab, glove, test_batch_size)
	return evaluator.evaluate(test_dataloader, model, returntype = 'Blue')

