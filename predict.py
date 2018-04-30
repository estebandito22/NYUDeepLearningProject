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
from stal import STAL

try:
	from layers.wordspretrained import PretrainedEmbeddings
except:
	from wordspretrained import PretrainedEmbeddings


USE_CUDA = False
if torch.cuda.is_available():
	USE_CUDA = True

def _caption(hyp, videoid, vocab):
	generatedstring = ' '.join([str(vocab.index2word[index.cpu().item()]) for index in hyp[1:-1]])
	string_hyp = {'videoid': str(videoid), 'captions': [generatedstring]}
	#print(string_hyp)
	return string_hyp

def evaluate(dataloader, model_filepath, vocab, epoch, model_name, spatial, map_location, returntype = 'ALL'):

        checkpoint = torch.load(model_filepath, map_location=map_location)
	print(checkpoint['dict_args'])
        if not spatial : model = CSAL(checkpoint['dict_args'])
        else : model = STAL(checkpoint['dict_args'])
        model = nn.DataParallel(model) if data_parallel else model

        model.load_state_dict(checkpoint['state_dict'])

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	predcaptionsjson = 'epoch{}_predcaptions.json'.format(epoch)
	valscoresjson = 'epoch{}_valscores.json'.format(epoch)

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
		video_ids_list = batch[2]
		if USE_CUDA:
			padded_imageframes_batch = padded_imageframes_batch.cuda()
			frame_sequence_lengths = frame_sequence_lengths.cuda()
			padded_inputwords_batch = padded_inputwords_batch.cuda()
			dummy_input_sequence_lengths = dummy_input_sequence_lengths.cuda()

		#######Forward
		model.eval()
		indexcaption = model(padded_imageframes_batch, frame_sequence_lengths, \
						padded_inputwords_batch, dummy_input_sequence_lengths)


		#######Captions
		#print(indexcaption)
		stringcaptions += [_caption(indexcaption, video_ids_list[0], vocab)]

	#######Write predicted captions
	if not os.path.isdir(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name)):
		os.makedirs(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name))
	with open(os.path.join(cur_dir, output_dir, MSRVTT_dir, model_name, predcaptionsjson), 'w') as predsout:
		json.dump(stringcaptions, predsout)

	return

if __name__=="__main__":

	"""python predict.py -m baseline1 -e 100 -vf Alexnet -sp True"""

	ap = argparse.ArgumentParser()
	ap.add_argument("-bs", "--batch_size", default=1, required=False,
		help="If predict is True, the batch size to use during predictions.")
	ap.add_argument("-m", "--saved_model_dir", required=False,
		help="If predict True, the directory of the model you wish to load.")
	ap.add_argument("-e", "--saved_model_epoch", required=False,
		help="If predict True, the epoch you want to load e.g. 'epoch0'.")
	ap.add_argument("-vf", "--video_features", required=True,
		help="Either Resnet or Alexnet")
	ap.add_argument("-sp", "--spatial_features", action='store_true',
		required=False, help="use spatial features")
	args = vars(ap.parse_args())

	EVAL_BATCH_SIZE = int(args['batch_size'])
	#EPOCH = int(args['saved_model_epoch'])
	SPATIAL = args['spatial_features']
	VID_FEATS = args['video_features']
	EPOCHS = args['saved_model_epoch'].split(',')
	print(EPOCHS)

	cur_dir = os.getcwd()
	input_dir = 'input'
	output_dir = 'output'
	MSRVTT_dir = 'MSRVTT'
	models_dir = 'models'
	saved_model_dir = args['saved_model_dir']
	#epoch_dir = 'epoch'+args['saved_model_epoch']
	#csalfile = 'csal.pth'
	csalfile = 'stal.pth'
	glovefile = 'glove.pkl'
	vocabfile = 'vocab.pkl'
	captionsjson = 'captions.json'
	#predcaptionsjson = 'epoch{}_predcaptions.json'.format(EPOCH)
	#valscoresjson = 'epoch{}_valscores.json'.format(EPOCH)
	glove_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, glovefile)
	vocab_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, vocabfile)
	#model_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, epoch_dir, csalfile)
	#captions_filepath = os.path.join(cur_dir, input_dir, MSRVTT_dir, captionsjson)
	#preds_filepath = os.path.join(cur_dir, output_dir, MSRVTT_dir, saved_model_dir, predcaptionsjson)


	print("Loading previously trained model...")
	if USE_CUDA == True:
		maplocation = None
	else:
		maplocation = 'cpu'

	data_parallel = False
	frame_trunc_length = 45
	val_num_workers = 0
	val_pretrained = True
	val_pklexist = True

	spatial = SPATIAL

	#checkpoint = torch.load(model_filepath, map_location=maplocation)

	'''if not spatial : model = CSAL(checkpoint['dict_args'])
	else : model = STAL(checkpoint['dict_args'])
	model = nn.DataParallel(model) if data_parallel else model

	model.load_state_dict(checkpoint['state_dict'])'''

	glovefile = open(glove_filepath, 'rb')
	glove = pickle.load(glovefile)
	glovefile.close()

	vocabfile = open(vocab_filepath, 'rb')
	vocab = pickle.load(vocabfile)
	vocabfile.close()

	print("Get validation data...")
	if VID_FEATS == "Resnet":
		val_pkl_file = 'MSRVTT/Pixel/Resnet51222/valvideo.pkl'
	elif VID_FEATS == "Alexnet":
		if not spatial : val_pkl_file = 'MSRVTT/Pixel/Alexnet1000/valvideo.pkl'
		else: val_pkl_file = 'MSRVTT/Pixel/Alexnet25622/valvideo.pkl'
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

	for EPOCH in EPOCHS:
		epoch_dir = 'epoch'+EPOCH
		EPOCH = int(EPOCH)
		model_filepath = os.path.join(cur_dir, models_dir, saved_model_dir, epoch_dir, csalfile)
		evaluate(val_dataloader, model_filepath, vocab, epoch=EPOCH, model_name=saved_model_dir, spatial=spatial, map_location=maplocation, returntype='Bleu')

#python predict.py -m SpatTempstal -e 12 -vf Alexnet -sp
