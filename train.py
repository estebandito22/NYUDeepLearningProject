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
import test as tester

USE_CUDA = False
	 
def train():
	cur_dir = os.getcwd()
	input_dir = 'input'
	glove_dir = '../SQUAD/glove/'
	glove_filename = 'glove.6B.50d.txt'
	glove_embdim = 50
	glove_filepath = os.path.join(glove_dir, glove_filename)

	train_batch_size = 2
	eval_batch_size = 1

	file_names = [('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')]
	files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
	train_dataloader, vocab, glove, train_data_size = loader.get_train_data(files, glove_filepath, glove_embdim, train_batch_size)

	file_names = [('MSRVTT/captions.json', 'MSRVTT/valvideo.json', 'MSRVTT/Frames')]
	files = [[os.path.join(cur_dir, input_dir, filetype) for filetype in file] for file in file_names]
	val_dataloader = loader.get_val_data(files, vocab, glove, eval_batch_size)


	save_dir = 'models/baseline/'
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
	dict_args = {
					"intermediate_layers" : ['layer4', 'fc'],
					"word_embeddings" : pretrained_wordvecs,
					"pretrained_embdim" : glove_embdim,
					"vocabulary_size" : len(pretrained_wordvecs),
					"decoder_rnn_hidden_dim" : glove_embdim,
					"decoder_tie_weights" : True,
					"decoder_rnn_type" : 'LSTM',
					"pretrained_feature_size" : 1000
				}
	csal = CSAL(dict_args)

	num_epochs = 4
	learning_rate = 1.0
	criterion = nn.NLLLoss(reduce = False)  
	optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, csal.parameters()), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
	if USE_CUDA:
		csal = csal.cuda()
		criterion = criterion.cuda()

	for epoch in range(num_epochs):

		for i,batch in enumerate(train_dataloader):

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
				padded_imageframes_batch = padded_imageframes_batch.cuda()
				frame_sequence_lengths = frame_sequence_lengths.cuda()
				padded_inputwords_batch = padded_inputwords_batch.cuda()
				input_sequence_lengths = input_sequence_lengths.cuda()
				padded_outputwords_batch = padded_outputwords_batch.cuda()
				output_sequence_lengths = output_sequence_lengths.cuda()
				captionwords_mask = captionwords_mask.cuda()

			#######Forward
			csal = csal.train()
			optimizer.zero_grad()
			outputword_log_probabilities = csal(padded_imageframes_batch, frame_sequence_lengths, \
												padded_inputwords_batch, input_sequence_lengths)

			#######Calculate Loss
			outputword_log_probabilities = outputword_log_probabilities.permute(0, 2, 1)
			#outputword_log_probabilities: batch_size*vocab_size*num_words
			#padded_outputwords_batch: batch_size*num_words
			losses = criterion(outputword_log_probabilities, padded_outputwords_batch)
			#loss: batch_size*num_words
			losses = losses*captionwords_mask.float()
			loss = losses.sum()

			#######Backward
			loss.backward()
			optimizer.step()

			if((i+1)%2 == 0):
				print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}'.format( \
							epoch+1, num_epochs, i+1, train_data_size//train_batch_size, loss.data[0]))
			
				break

		if(epoch%1 == 0): #After how many epochs
			#Get Validation Loss to stop overriding
			val_loss, bleu = evaluator.evaluate(val_dataloader, csal)
			#print("val_loss")
			#Early Stopping not required
			filename = 'csal' + '.pth'
			file = open(os.path.join(save_dir, filename), 'wb')
			torch.save({'state_dict':csal.state_dict(), 'dict_args':dict_args}, file)
			print('Saving the model to {}'.format(save_dir))
			file.close()

if __name__=='__main__':
	train()
