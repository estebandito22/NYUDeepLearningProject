import os
import torch

from input.vocab import *
#from vocab import *

class Glove():

	def __init__(self, filepath, embdim):
		self.filepath = filepath
		self.embdim = embdim
		self.word2vec = {}
		self.index2vec = None

	def create(self, vocab):
		file = open(self.filepath, 'r')
		vector_dim = self.embdim
		self.index2vec = torch.randn(len(vocab.word2index) ,self.embdim)
		for line in file:
			words = line.strip().split(' ')
			word = words[0]
			vec = [float(value) for value in words[1:]]
			if word in vocab.word2index:
				self.word2vec[word] = vec
				self.index2vec[vocab.word2index[word]] = torch.FloatTensor(vec)
		for word in vocab.word2index.keys():
			if word not in self.word2vec.keys():
				random = [float('%.5f'%(torch.rand(1)[0])) for l in range(vector_dim)]
				self.word2vec[word] = random
				self.index2vec[vocab.word2index[word]] = torch.FloatTensor(random)
		return

	def get_word_vectors(self, tokens):
		wvecs = []
		for token in tokens:
			if token.lower() in self.word2vec:
				wvecs.append(self.word2vec[token.lower()])
			else:
				raise
		return wvecs

	def get_index_vectors(self, indices):
		wvecs = []
		for index in indices:
			if index < len(self.index2vec):
				wvecs.append(self.index2vec[index])		
			else:
				#print ("Not found in Vocab replacing with UNK  " + str(index))
				raise	
		return wvecs


if __name__ == '__main__':
	glove_dir = '../../SQUAD/glove/'
	glove_filename = 'glove.6B.50d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)
	embdim = 50

	glove = Glove(glove_filepath, embdim)
	vocab = Vocab([('MSRVTT/captions.json', 'MSRVTT/trainvideo.json', 'Dummy')])
	vocab.add_begend_vocab()

	vocab.create()
	glove.create(vocab)

	#print(glove.get_word_vectors(['the','wf','<eos>']))

	print(glove.get_index_vectors([1, 3, 5]))
