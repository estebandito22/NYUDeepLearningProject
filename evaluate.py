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

from evaluators.pycocotools.coco_video import COCO
from evaluators.pycocoevalcap.eval import COCOEvalCap

USE_CUDA = False

def _caption(hyp, videoid, vocab):
	generatedstring = ' '.join([str(vocab.index2word[index.data[0]]) for index in hyp])
	string_hyp = {'videoid': str(videoid), 'captions': [generatedstring]}
	return string_hyp

def evaluate(dataloader, model, vocab, epoch, returntype = 'ALL'):

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	predcaptionsjson = 'epoch{}_baseline_predcaptions.json'.format(epoch)
	valscoresjson = 'epoch{}_baseline_val_scores.json'.format(epoch)

	stringcaptions = []

	criterion = nn.NLLLoss(reduce = False)
	if USE_CUDA:
		model = model.cuda()
		criterion = criterion.cuda()

	for i,batch in enumerate(dataloader):

		#######Load Data
		padded_imageframes_batch = Variable(torch.stack(batch[0]), volatile=True) #batch_size*num_frames*3*224*224
		frame_sequence_lengths = Variable(torch.LongTensor(batch[1]), volatile=True) #batch_size
		padded_inputwords_batch = Variable(torch.LongTensor([[vocab.word2index['<bos>']]]), volatile=True)
		dummy_input_sequence_lengths = Variable(torch.LongTensor([[0]]), volatile=True)
		# padded_outputwords_batch = Variable(torch.LongTensor(batch[2])) #batch_size*num_words
		# output_sequence_lengths = Variable(torch.LongTensor(batch[3])) #batch_size
		video_ids_list = batch[2]
		if USE_CUDA:
			padded_imageframes_batch = padded_imageframes_batch.cuda()
			frame_sequence_lengths = frame_sequence_lengths.cuda()

		#######Forward
		model.eval()
		outputword_log_probabilities, indexcaption = model(padded_imageframes_batch, frame_sequence_lengths, \
															padded_inputwords_batch, dummy_input_sequence_lengths)

		#######Calculate Loss
		# outputword_log_probabilities = outputword_log_probabilities.permute(0, 2, 1)
		# #outputword_log_probabilities: batch_size*vocab_size*num_words
		# #padded_outputwords_batch: batch_size*num_words
		# losses = criterion(outputword_log_probabilities, padded_outputwords_batch)
		# #loss: batch_size*num_words
		# losses = losses*captionwords_mask.float()
		# loss = losses.sum()

		#######Captions
		stringcaptions += [_caption(indexcaption, video_ids_list[0], vocab)]

	#######Write predicted captions
	with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, predcaptionsjson), 'w') as predsout:
		json.dump(stringcaptions, predsout)

	#for Variables use volatile=True
	if returntype == 'Bleu' or returntype == 'All':
		captionsjson = 'captions.json'
		captionsfile = os.path.join(cur_dir, input_dir, MSRVTT_dir, captionsjson)
		predsfile = os.path.join(cur_dir, output_dir, MSRVTT_dir, predcaptionsjson)
		coco = COCO(captionsfile)
		cocopreds = coco.loadRes(predsfile)
		cocoEval = COCOEvalCap(coco, cocopreds)
		cocoEval.evaluate()
		scores = ["{}: {:0.4f}".format(metric, score) for metric, score in cocoEval.eval.items()]
		with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, valscoresjson), 'w') as scoresout:
			json.dump(scores, scoresout)
		if returntype == 'Bleu':
			return scores
	elif returntype == 'Loss':
		return loss
	return loss, scores

if __name__=="__main__":

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	pred_captionsjson = 'predcaptions.json'

	captionsjson = 'captions.json'
	captionsfile = os.path.join(cur_dir, input_dir, MSRVTT_dir, captionsjson)
	predsfile = os.path.join(cur_dir, output_dir, MSRVTT_dir, predcaptionsjson)
	coco = COCO(captionsfile)
	cocopreds = coco.loadRes(predsfile)
	cocoEval = COCOEvalCap(coco, cocopreds)
	cocoEval.evaluate()
