import torch
import torch.nn as nn
from torch.autograd import Variable


class PretrainedEmbeddings(nn.Module):

	def __init__(self, dict_args):
		super(PretrainedEmbeddings, self).__init__()

		self.word_embeddings = dict_args["word_embeddings"]
		self.pretrained_embdim = dict_args["pretrained_embdim"]
		self.vocabulary_size = dict_args["vocabulary_size"]

		self.embeddings = nn.Embedding(self.vocabulary_size, self.pretrained_embdim)
		self.embeddings.weight = nn.Parameter(self.word_embeddings)
		self.embeddings.weight.requires_grad = False

	def forward(self, sequence):
		#sequence: batch_size*num_words
		return self.embeddings(sequence) #batch_size*num_words*wembed_dim


if __name__=='__main__':

	pretrainedEmbeddings = PretrainedEmbeddings({"word_embeddings" : torch.randn(10,3), "pretrained_embdim" : 3, "vocabulary_size":10})
	print(pretrainedEmbeddings(torch.LongTensor([[1,2],[9,0]])))

