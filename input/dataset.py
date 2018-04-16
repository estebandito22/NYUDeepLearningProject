import os
import sys

import torch
import numpy as numpy
import torch.utils.data as data

try:
	from input.vocab import Vocab
	from input.glove import Glove
	from input.pixel import Pixel
	from input.dataiter import *
	from input.datautils import *
except:
	from vocab import Vocab
	from glove import Glove
	from pixel import Pixel
	from dataiter import *
	from datautils import *


#Modify it for test to disable captions list

class Dataset(data.Dataset):
	def __init__(self, files):
		self.files = files
		self.data = [] #list of [videoid, inputcaption, outputcaption]

		self.size = 0
		self.eos = 0
		self.bos = 0
		self.ipad = 0

		self.glove = None
		self.pixel = None
		self.mode = 'train'
		self.pretrained = False
		self.data_parallel = False
		self.frame_trunc_length = 45

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		if self.pretrained:
			videoid = self.data[index][0]
			imageframes = self.pixel.get_pixel_vectors(videoid)
			return [imageframes] + self.data[index][1:] + [videoid]
		imagefiles = self.data[index][0]
		imageframes = [imagetotensor(imagefile) for imagefile in imagefiles]
		return [imageframes] + self.data[index][1:]
		
	def set_pad_indices(self, vocab):
		self.bos = vocab.word2index['<bos>']
		self.eos = vocab.word2index['<eos>']
		if self.pretrained: self.ipad = torch.zeros([1000,1,1])
		else: self.ipad = torch.zeros([3,224,224]) #create zero tensor based on the resize value from dataiter

	def add_video_vecs(self, pixel):
		self.pixel = pixel

	def add_glove_vecs(self, glove):
		self.glove = glove

	#def set_mode(self, mode):
		#self.mode = mode

	def set_flags(self, mode='train', data_parallel=False, frame_trunc_length=45, pretrained=False):
		self.mode = mode
		self.data_parallel = data_parallel
		self.frame_trunc_length = frame_trunc_length
		self.pretrained = pretrained

	def create(self, vocab):
		for captionsfilepath, trainvideoidspath, framesfolderpath in self.files:
			captions = json.load(open(captionsfilepath, 'r'))
			trainvideoids = json.load(open(trainvideoidspath, 'r'))
			for videoid, captionslist in iterdatashallow(captions, trainvideoids):
				if not self.pretrained:
					imagefiles = [imagefile for imagefile in iterimages(framesfolderpath, videoid)]
				if self.mode =='train':
					for caption in captionslist:
						inputcaptionwords = [self.bos] + vocab.get_word_indices(caption)
						outputcaptionwords = vocab.get_word_indices(caption) + [self.eos]
						if not self.pretrained:
							self.data.append([imagefiles, inputcaptionwords, outputcaptionwords, videoid])
						else:
							self.data.append([videoid, inputcaptionwords, outputcaptionwords])
				else:
					if not self.pretrained:
						self.data.append([imagefiles, videoid])
					else:
						self.data.append([videoid])
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

		def get_padded_list_truncated(sequence_list, padding_value, trunc_length, tensor = False, strict=False):
			sequence_lengths = [len(sequence) for sequence in sequence_list]
			num_sequences = len(sequence_list)
			max_length = max(sequence_lengths)
			if strict == True:
				max_length = trunc_length
			if max_length <= trunc_length:
				padded_sequence_list = [[padding_value for col in range(max_length)] for row in range(num_sequences)]
			else:
				padded_sequence_list = [[padding_value for col in range(trunc_length)] for row in range(num_sequences)]
			for index, sequence in enumerate(sequence_list):
				if sequence_lengths[index] <= trunc_length:
					padded_sequence_list[index][:sequence_lengths[index]] = sequence
				else:
					padded_sequence_list[index][:trunc_length] = sequence[:trunc_length]
					sequence_lengths[index] = trunc_length
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

			padded_imageframes_batch, frame_sequence_lengths = get_padded_list_truncated(imageframesbatch, self.ipad, self.frame_trunc_length, tensor=True, strict=self.data_parallel)

			return padded_imageframes_batch, frame_sequence_lengths, \
			   		padded_inputwords_batch, input_sequence_lengths, \
			   		padded_outputwords_batch, output_sequence_lengths, videoidsbatch

		else:
			imageframesbatch, videoidsbatch = zip(*mini_batch)
			padded_imageframes_batch, frame_sequence_lengths = get_padded_list_truncated(imageframesbatch, self.ipad, self.frame_trunc_length, tensor=True, strict=self.data_parallel)

			return padded_imageframes_batch, frame_sequence_lengths, videoidsbatch


if __name__=='__main__':

	vocab = Vocab([('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')])
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

	pixel = Pixel([('Dummy', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')])
	pixel.create()
	dataset.add_video_vecs(pixel)

	ibatch, ilens, inbatch, inlens, obatch, olens, videoids = \
		dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(27)])

	print(ibatch)
