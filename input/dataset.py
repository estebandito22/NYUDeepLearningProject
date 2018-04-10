import os
import sys

import torch
import numpy as numpy
import torch.utils.data as data

from input.vocab import Vocab
from input.glove import Glove
from input.dataiter import *
from input.datautils import *

#from vocab import Vocab
#from glove import Glove
#from dataiter import *
#from datautils import *


#Modify it for test to disable captions list

class Dataset(data.Dataset):
	def __init__(self, files):
		self.files = files
		self.data = [] #list of [frames, inputcaption, outputcaption]

		self.size = 0
		self.eos = 0
		self.bos = 0
		self.ipad = 0

		self.glove = None
		self.mode = 'train'

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		imagefiles = self.data[index][0]
		imageframes = [imagetotensor(imagefile) for imagefile in imagefiles]
		self.data[index][0] = imageframes
		return self.data[index]

	def set_pad_indices(self, vocab):
		self.bos = vocab.word2index['<bos>']
		self.eos = vocab.word2index['<eos>']
		self.ipad = torch.zeros([3,224,224]) #create zero tensor based on the resize value from dataiter

	def add_glove_vecs(self, glove):
		self.glove = glove

	def set_mode(self, mode):
		self.mode = mode

	def create(self, vocab):
		for captionsfilepath, trainvideoidspath, framesfolderpath in self.files:
			captions = json.load(open(captionsfilepath, 'r'))
			trainvideoids = json.load(open(trainvideoidspath, 'r')) 
			for videoid, captionslist in iterdatashallow(captions, trainvideoids):
				imagefiles = [imagefile for imagefile in iterimages(framesfolderpath, videoid)]
				if self.mode =='train':
					for caption in captionslist:
						inputcaptionwords = [self.bos] + vocab.get_word_indices(caption)
						outputcaptionwords = vocab.get_word_indices(caption) + [self.eos]
						self.data.append([imagefiles, inputcaptionwords, outputcaptionwords, videoid])
				else:
					self.data.append([imagefiles, videoid])
		self.size = len(self.data)

	def collate_fn(self, mini_batch):
		def get_padded_list_normal(sequence_list, padding_value, tensor = False):
			sequence_lengths = [len(sequence) for sequence in sequence_list]
			max_length = max(sequence_lengths)
			num_sequences = len(sequence_list)
			padded_sequence_list = [[padding_value for col in range(max_length)] for row in range(num_sequences)]
			for index, sequence in enumerate(sequence_list):
				padded_sequence_list[index][:sequence_lengths[index]] = sequence
				if tensor: padded_sequence_list[index] = torch.stack(padded_sequence_list[index])
			return padded_sequence_list, sequence_lengths

		def replace_indices_with_vecs(sequence_list):
			return [self.glove.get_index_vectors(sequence) for sequence in sequence_list]

		if self.mode =='train':
			imageframesbatch, inputwordsbatch, outputwordsbatch, videoidsbatch = zip(*mini_batch)
			padded_inputwords_batch, input_sequence_lengths = get_padded_list_normal(inputwordsbatch, self.eos)
			padded_outputwords_batch, output_sequence_lengths = get_padded_list_normal(outputwordsbatch, self.eos)
			#padded_inputwordsvecs_batch = replace_indices_with_vecs(padded_inputwords_batch)
			#padded_outputwordsvecs_batch = replace_indices_with_vecs(padded_outputwords_batch)

			padded_imageframes_batch, frame_sequence_lengths = get_padded_list_normal(imageframesbatch, self.ipad, tensor=True)

			return padded_imageframes_batch, frame_sequence_lengths, \
			   		padded_inputwords_batch, input_sequence_lengths, \
			   		padded_outputwords_batch, output_sequence_lengths, videoidsbatch

		else:
			imageframesbatch, videoidsbatch = zip(*mini_batch)
			padded_imageframes_batch, frame_sequence_lengths = get_padded_list_normal(imageframesbatch, self.ipad, tensor=True)

			return padded_imageframes_batch, frame_sequence_lengths, videoidsbatch
			   

if __name__=='__main__':

	vocab = Vocab([('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'Dummy')])
	vocab.add_begend_vocab()
	vocab.create()

	dataset = Dataset([('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')])	
	dataset.set_pad_indices(vocab)
	dataset.create(vocab)

	glove_dir = '../../SQUAD/glove/'
	glove_filename = 'glove.6B.50d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)
	embdim = 50
	glove = Glove(glove_filepath, embdim)
	glove.create(vocab)
	dataset.add_glove_vecs(glove)

	ibatch, ilens, inbatch, inlens, obatch, olens, videoids = \
		dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(27)])

	print(inbatch)

