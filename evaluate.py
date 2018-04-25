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

from evaluators.pycocotools.coco_video import COCO
from evaluators.pycocoevalcap.eval import COCOEvalCap

if torch.cuda.is_available():
	USE_CUDA = True
else:
	USE_CUDA = False

def _caption(hyp, videoid, vocab):
	generatedstring = ' '.join([str(vocab.index2word[index.data[0]]) for index in hyp])
	string_hyp = {'videoid': str(videoid), 'captions': [generatedstring]}
	return string_hyp

def evaluate(dataloader, model, vocab, epoch, model_name, returntype = 'ALL'):

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	predcaptionsjson = 'epoch{}_predcaptions.json'.format(epoch)
	valscoresjson = 'epoch{}_val_scores.json'.format(epoch)

	stringcaptions = []

	criterion = nn.NLLLoss(reduce = False)
	if USE_CUDA:
		model = model.cuda()
		criterion = criterion.cuda()

	for i,batch in enumerate(dataloader):

		#######Load Data
		padded_imageframes_batch = Variable(torch.stack(batch[0]), volatile=True) #batch_size*num_frames*3*224*224
		frame_sequence_lengths = Variable(torch.LongTensor(batch[1]), volatile=True) #batch_size
		padded_inputwords_batch = Variable(torch.LongTensor([[vocab.word2index['<bos>']]]), volatile=True) #batch_size*num_words
		dummy_input_sequence_lengths = Variable(torch.LongTensor([[0]]), volatile=True) #batch_size
		# padded_outputwords_batch = Variable(torch.LongTensor(batch[2])) #batch_size*num_words
		# output_sequence_lengths = Variable(torch.LongTensor(batch[3])) #batch_size
		video_ids_list = batch[2]
		if USE_CUDA:
			padded_imageframes_batch = padded_imageframes_batch.cuda()
			frame_sequence_lengths = frame_sequence_lengths.cuda()
			padded_inputwords_batch = padded_inputwords_batch.cuda()
			dummy_input_sequence_lengths = dummy_input_sequence_lengths.cuda()

		#######Forward
		model.eval()
		# outputword_log_probabilities, indexcaption = model(padded_imageframes_batch, frame_sequence_lengths, \
		# 													padded_inputwords_batch, dummy_input_sequence_lengths)
		indexcaption = model(padded_imageframes_batch, frame_sequence_lengths, \
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
	if not os.path.isdir(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name)):
		os.makedirs(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name))
	with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name, predcaptionsjson), 'w') as predsout:
		json.dump(stringcaptions, predsout)

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
	elif returntype == 'Loss':
		return loss
	return loss, scores

if __name__=="__main__":

	"""
	Usage: Example of command to use the previously trained model "baseline1"
	from "epoch0" to generate predictions and evaluate on validaition.

	python evaluate.py -p -m baseline1 -e 100
	"""

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--predict", action="store_true", default=False,
		required=False, help="Flag to load a pretrained model and predict.")
	ap.add_argument("-s", "--batch_size", default=1, required=False,
		help="If predict is True, the batch size to use during predictions.")
	ap.add_argument("-m", "--saved_model_dir", required=False,
		help="If predict True, the directory of the model you wish to load.")
	ap.add_argument("-e", "--saved_model_epoch", required=False,
		help="If predict True, the epoch you want to load e.g. 'epoch0'.")
	args = vars(ap.parse_args())

	PREDICT = args['predict']
	EVAL_BATCH_SIZE = int(args['batch_size'])
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

	if PREDICT == True:

		print("Loading previously trained model...")
		if USE_CUDA == True:
			maplocation = None
		else:
			maplocation = 'cpu'
		checkpoint = torch.load(model_filepath, map_location=maplocation)

		if not 'embeddings_requires_grad' in checkpoint['dict_args']:
			checkpoint['dict_args']['embeddings_requires_grad'] = False

		model = CSAL(checkpoint['dict_args'])
		model = nn.DataParallel(model)
		# print(checkpoint['state_dict'].keys())

		# if not 'module.sentence_decoder_layer.pretrained_words_layer.embeddings.weight' \
		# 	in checkpoint['state_dict']:
		#
		# 	pretrained_words_layer = model.module.pretrained_words_layer
		# 	checkpoint['state_dict']['module.sentence_decoder_layer.pretrained_words_layer.embeddings.weight'] \
		# 		= pretrained_words_layer.embeddings.weight

		model.load_state_dict(checkpoint['state_dict'])

		glovefile = open(glove_filepath, 'rb')
		glove = pickle.load(glovefile)
		glovefile.close()

		vocabfile = open(vocab_filepath, 'rb')
		vocab = pickle.load(vocabfile)
		vocabfile.close()

		data_parallel = True
		frame_trunc_length = 45
		val_num_workers = 0
		val_pretrained = True
		val_pklexist = True

		print("Get validation data...")
		val_pkl_file = 'MSRVTT/valvideo.pkl'
		file_names = [('MSRVTT/captions.json', 'MSRVTT/valvideo.json', 'MSRVTT/Frames')]
		files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
		val_pkl_path = os.path.join(cur_dir, input_dir, val_pkl_file)
		val_dataloader = loader.get_val_data(files, val_pkl_path, vocab, glove,
		 									batch_size=EVAL_BATCH_SIZE,
											num_workers=val_num_workers,
											pretrained=val_pretrained,
											pklexist= val_pklexist,
											data_parallel=data_parallel,
											frame_trunc_length=frame_trunc_length)

		#Get Validation Loss to stop overriding
		# val_loss, bleu = evaluator.evaluate(val_dataloader, csal, vocab)
		# print("Validation Loss: {}\tValidation Scores: {}".format(val_loss, bleu))
		# bleu = evaluate(val_dataloader, model, vocab, epoch=EPOCH, model_name=saved_model_dir, returntype='Bleu')
		evaluate(val_dataloader, model, vocab, epoch=EPOCH, model_name=saved_model_dir, returntype='Bleu')
		# print("Validation Scores: {}".format(bleu))

	else:

		coco = COCO(captions_filepath)
		cocopreds = coco.loadRes(preds_filepath)
		cocoEval = COCOEvalCap(coco, cocopreds)
		cocoEval.evaluate()
		scores = ["{}: {:0.4f}".format(metric, score) for metric, score in cocoEval.eval.items()]
		with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, saved_model_dir, valscoresjson), 'w') as scoresout:
			json.dump(scores, scoresout)
