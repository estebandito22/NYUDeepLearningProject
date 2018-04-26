import os
import sys
import json
import pickle
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

import input.dataloader as loader
import layers.utils as utils
from csal import CSAL

import evaluate as evaluator
import test as tester

if torch.cuda.is_available():
	USE_CUDA = True
else:
	USE_CUDA = False

def train():
	cur_dir = os.getcwd()
	input_dir = 'input'
	glove_dir = 'glove/'
	glove_filename = 'glove.6B.300d.txt'
	glove_embdim = 300
	glove_filepath = os.path.join(glove_dir, glove_filename)

	data_parallel = True
	frame_trunc_length = 45

	train_batch_size = 128
	train_num_workers = 0
	train_pretrained = True
	train_pklexist = True
	eval_batch_size = 1

	print("Get train data...")
	#train_pkl_file = 'MSRVTT/Pixel/Resnet1000/trainvideo.pkl'
	train_pkl_file = 'MSRVTT/Pixel/Alexnet1000/trainvideo.pkl'
	file_names = [('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')]
	files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
	train_pkl_path = os.path.join(cur_dir, input_dir, train_pkl_file)

	train_dataloader, vocab, glove, train_data_size = loader.get_train_data(files, train_pkl_path, glove_filepath, glove_embdim, batch_size=train_batch_size, num_workers=train_num_workers, pretrained = train_pretrained, pklexist = train_pklexist, data_parallel=data_parallel, frame_trunc_length=frame_trunc_length)

	# print("Get validation data...")
	# file_names = [('MSRVTT/captions.json', 'MSRVTT/valvideo.json.sample', 'MSRVTT/Frames')]
	# files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
	# val_dataloader = loader.get_val_data(files, vocab, glove, eval_batch_size)

	modelname = 'test'
	save_dir = 'models/{}/'.format(modelname)
	save_dir_path = os.path.join(cur_dir, save_dir)
	if not os.path.exists(save_dir_path):
		os.makedirs(save_dir_path)

	glovefile = open(os.path.join(save_dir, 'glove.pkl'), 'wb')
	pickle.dump(glove, glovefile)
	glovefile.close()

	vocabfile = open(os.path.join(save_dir, 'vocab.pkl'), 'wb')
	pickle.dump(vocab, vocabfile)
	vocabfile.close()

	pretrained_wordvecs = glove.index2vec
	#model_name = MP, MPAttn, LSTM, LSTMAttn for CSAL
	hidden_dimension = 512 #glove_embdim
	dict_args = {
					"intermediate_layers" : ['layer4', 'fc'],
					"pretrained_feature_size" : 1000,
					#
					"word_embeddings" : pretrained_wordvecs,
					"word_embdim" : glove_embdim,
					"vocabulary_size" : len(pretrained_wordvecs),
					"use_pretrained_emb" : True,
					"backprop_embeddings" : False,
					#
					"encoder_configuration" : 'LSTMAttn',
					"encoder_input_dim" : hidden_dimension,
					"encoder_rnn_type" : 'LSTM',
					"encoder_rnn_hdim" : hidden_dimension,
					"encoder_num_layers" : 1,
					"encoder_dropout_rate" : 0.2,
					"encoderattn_projection_dim" : hidden_dimension/2,
					"encoderattn_query_dim" : hidden_dimension,
					#
					"decoder_rnn_input_dim" : glove_embdim + hidden_dimension,
					#"decoder_rnn_input_dim" : glove_embdim,
					"decoder_rnn_hidden_dim" : hidden_dimension,
					"decoder_tie_weights" : False,
					"decoder_rnn_type" : 'LSTM',
					"every_step": True,
					#"every_step": False
				}
	csal = CSAL(dict_args)
	print(dict_args)

	num_epochs = 500
	learning_rate = 1
	criterion = nn.NLLLoss(reduce = False)
	optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, csal.parameters()), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
	if USE_CUDA:
		if data_parallel: csal = nn.DataParallel(csal).cuda()
		else: csal = csal.cuda()
		criterion = criterion.cuda()

	print("Start training...")
	for epoch in range(num_epochs):

		start_time = time.time()
		for i,batch in enumerate(train_dataloader):

			load_time =  time.time()
			#######Load Data
			padded_imageframes_batch = Variable(torch.stack(batch[0])) #batch_size*num_frames*3*224*224
			frame_sequence_lengths = Variable(torch.LongTensor(batch[1])) #batch_size
			padded_inputwords_batch = Variable(torch.LongTensor(batch[2])) #batch_size*num_words
			input_sequence_lengths = Variable(torch.LongTensor(batch[3])) #batch_size
			padded_outputwords_batch = Variable(torch.LongTensor(batch[4])) #batch_size*num_words
			output_sequence_lengths = Variable(torch.LongTensor(batch[5])) #batch_size
			video_ids_list = batch[6]
			captionwords_mask = Variable(utils.sequence_mask(output_sequence_lengths)) #batch_size*num_words
			if USE_CUDA:
				async = data_parallel
				padded_imageframes_batch = padded_imageframes_batch.cuda(async=async)
				frame_sequence_lengths = frame_sequence_lengths.cuda(async=async)
				padded_inputwords_batch = padded_inputwords_batch.cuda(async=async)
				input_sequence_lengths = input_sequence_lengths.cuda(async=async)
				padded_outputwords_batch = padded_outputwords_batch.cuda(async=async)
				output_sequence_lengths = output_sequence_lengths.cuda(async=async)
				captionwords_mask = captionwords_mask.cuda(async=async)
			#print(padded_imageframes_batch.size())
			cuda_time = time.time()
			#######Forward
			csal = csal.train()
			optimizer.zero_grad()
			outputword_log_probabilities = csal(padded_imageframes_batch, frame_sequence_lengths, \
												padded_inputwords_batch, input_sequence_lengths)

			model_time = time.time()
			#######Calculate Loss
			outputword_log_probabilities = outputword_log_probabilities.permute(0, 2, 1)
			#outputword_log_probabilities: batch_size*vocab_size*num_words
			#padded_outputwords_batch: batch_size*num_words
			losses = criterion(outputword_log_probabilities, padded_outputwords_batch)
			#losses: batch_size*num_words
			losses = losses*captionwords_mask.float()
			#Divide by batch size and num_words
			losses = losses.sum(1)/(output_sequence_lengths.float())
			loss = losses.sum()/losses.size(0)

			loss_time = time.time()
			#######Backward
			loss.backward(retain_graph=False)
			optimizer.step()

			opt_time = time.time()
			#######Report
			if((i+1)%5 == 0):
				print('Epoch: [{0}/{1}], Step: [{2}/{3}], Test Loss: {4}'.format( \
							epoch+1, num_epochs, i+1, train_data_size//train_batch_size, loss.data[0]))


			#print("Load : {0}, Cuda : {1}, Model : {2}, Loss : {3}, Opt : {4}".format(start_time-load_time, load_time - cuda_time, cuda_time - model_time, model_time - loss_time, loss_time-opt_time))
			start_time = time.time()

		if(epoch%4 == 0): #After how many epochs
			#Get Validation Loss to stop overriding
			# val_loss, bleu = evaluator.evaluate(val_dataloader, csal, vocab)
			# print("Validation Loss: {}\tValidation Scores: {}".format(val_loss, bleu))
			# bleu = evaluator.evaluate(val_dataloader, csal, vocab, epoch=epoch, model_name = modelname, returntype='Bleu')
			# print("Validation Scores: {}".format(bleu))
			#Early Stopping not required
			if not os.path.isdir(os.path.join(save_dir, "epoch{}".format(epoch))):
				os.makedirs(os.path.join(save_dir, "epoch{}".format(epoch)))
			filename = 'csal' + '.pth'
			file = open(os.path.join(save_dir, "epoch{}".format(epoch), filename), 'wb')
			torch.save({'state_dict':csal.state_dict(), 'dict_args':dict_args}, file)
			print('Saving the model to {}'.format(save_dir+"epoch{}".format(epoch)))
			file.close()

if __name__=='__main__':
	train()
