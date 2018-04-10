import os
import sys
import json

from collections import Counter

from input.dataiter import *
#from dataiter import *

class Vocab():

	def __init__(self, files):
		self.files = files
		self.word2index = {}
		self.index2word = {}
		
		self.word_counter = Counter()

	def add_begend_vocab(self):
		self.index2word[len(self.word2index)] = '<eos>'
		self.word2index['<eos>'] = len(self.word2index)		
		self.index2word[len(self.word2index)] = '<bos>'
		self.word2index['<bos>'] = len(self.word2index)
		return

	def update_word_counter(self, tokens):
		if tokens is not None:
			tokens = [token.lower() for token in tokens]
			self.word_counter.update(tokens)
		return

	def create_vocab_map(self, counter, item2index, index2item):
		allcounter = list(counter.items())
		cur_length = len(item2index)
		for index in range(0,len(allcounter)):
			item2index[allcounter[index][0]] = cur_length + index
			index2item[cur_length + index] = allcounter[index][0]
		return

	def create(self):
		for captionsfilepath, trainvideoidspath, _ in self.files:
			captions = json.load(open(captionsfilepath, 'r'))
			trainvideoids = json.load(open(trainvideoidspath, 'r')) 
			for videoid, caption_tokens in iterdatadeep(captions, trainvideoids):
				self.update_word_counter(caption_tokens)
		self.create_vocab_map(self.word_counter, self.word2index, self.index2word)
		return

	def get_word_indices(self, tokens):
		indices = []
		for token in tokens:
			if token.lower() in self.word2index:
				indices.append(self.word2index[token.lower()])
			else:
				raise		
		return indices

	def get_index_words(self, indices):
		words = []
		for index in indices:
			if index in self.index2word:
				words.append(self.index2word[index])
			else:
				raise
		return words

if __name__ == '__main__':
	vocab = Vocab([('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'Dummy')])
	vocab.add_begend_vocab()
	vocab.create()
	print(vocab.index2word)

