import os
import sys
import json
import pickle
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

import input.dataloader as loader
import layers.utils as utils
from csal import CSAL

try:
	from layers.wordspretrained import PretrainedEmbeddings
except:
	from wordspretrained import PretrainedEmbeddings


USE_CUDA = False
from evaluators.pycocotools.coco_video import COCO
from evaluators.pycocoevalcap.eval import COCOEvalCap


def evaluate(epoch, model_name, returntype = 'ALL'):

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	predcaptionsjson = 'epoch{}_predcaptions.json'.format(epoch)
	valscoresjson = 'epoch{}_valscores.json'.format(epoch)

	#for Variables use volatile=True
	if returntype == 'Bleu' or returntype == 'All':
		captionsjson = 'captions.json'
		captionsfile = os.path.join(cur_dir, input_dir, MSRVTT_dir, captionsjson)
		predsfile = os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name, predcaptionsjson)
		coco = COCO(captionsfile)
		cocopreds = coco.loadRes(predsfile)
		cocoEval = COCOEvalCap(coco, cocopreds)
		cocoEval.evaluate()
		scores = ["{}: {:0.4f}".format(metric, score) for metric, score in cocoEval.eval.items()]
		with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name, valscoresjson), 'w') as scoresout:
			json.dump(scores, scoresout)
		if returntype == 'Bleu':
			return scores
	return scores

if __name__=="__main__":

	"""
	Usage: Example of command to use the previously trained model "baseline1"
	from "epoch0" to generate predictions and evaluate on validaition.

	python evaluate.py -m baseline1 -e 100
	"""

	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--saved_model_dir", required=False,
		help="If predict True, the directory of the model you wish to load.")
	ap.add_argument("-e", "--saved_model_epoch", required=False,
		help="If predict True, the epoch you want to load e.g. 'epoch0'.")
	args = vars(ap.parse_args())

	EPOCH = int(args['saved_model_epoch'])

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	models_dir = 'models'
	saved_model_dir = args['saved_model_dir']
	epoch_dir = 'epoch'+args['saved_model_epoch']
	csalfile = 'csal.pth'
	glovefile = 'glove.pkl'
	vocabfile = 'vocab.pkl'
	captionsjson = 'captions.json'
	predcaptionsjson = 'epoch{}_predcaptions.json'.format(EPOCH)
	valscoresjson = 'epoch{}_valscores.json'.format(EPOCH)
	glove_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, glovefile)
	vocab_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, vocabfile)
	model_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, epoch_dir, csalfile)
	captions_filepath = os.path.join(cur_dir, input_dir, MSRVTT_dir, captionsjson)
	preds_filepath = os.path.join(cur_dir, output_dir, MSRVTT_dir, saved_model_dir, predcaptionsjson)

	bleu = evaluate(EPOCH, saved_model_dir, returntype='Bleu')
	print("Validation Scores: {}".format(bleu))
