import torch
import torch.nn as nn
from torch.autograd import Variable


class PretrainedEmbeddings(nn.Module):

	def __init__(self, dict_args):
		super(PretrainedEmbeddings, self).__init__()


		self.use_pretrained_emb = dict_args["use_pretrained_emb"]

		if self.use_pretrained_emb:
			self.word_embeddings = dict_args["word_embeddings"]
			self.backprop_embeddings = dict_args["backprop_embeddings"]

		self.word_embdim = dict_args["word_embdim"]
		self.vocabulary_size = dict_args["vocabulary_size"]
		self.embeddings_requires_grad = dict_args["embeddings_requires_grad"]

		self.embeddings = nn.Embedding(self.vocabulary_size, self.word_embdim)

		if self.use_pretrained_emb:		
			self.embeddings.weight = nn.Parameter(self.word_embeddings)
			self.embeddings.weight.requires_grad = self.backprop_embeddings

	def forward(self, sequence):
		#sequence: batch_size*num_words
		return self.embeddings(sequence) #batch_size*num_words*wembed_dim


if __name__=='__main__':

	dict_args = {
					"use_pretrained_emb" : False,
					"backprop_embeddings" : False,
					"word_embeddings" : torch.randn(10,3), 
					"word_embdim" : 3, 
					"vocabulary_size":10
				}

	pretrainedEmbeddings = PretrainedEmbeddings(dict_args)
	print(pretrainedEmbeddings(Variable(torch.LongTensor([[1,2],[9,0]]))))
