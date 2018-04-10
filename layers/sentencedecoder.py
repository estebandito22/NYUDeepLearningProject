import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
from layers.wordspretrained import PretrainedEmbeddings

#import utils as utils
#from wordspretrained import PretrainedEmbeddings

class SentenceDecoder(nn.Module):

	def __init__(self, dict_args):
		super(SentenceDecoder, self).__init__()

		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.vocab_size = dict_args["vocabulary_size"]
		self.tie_weights = dict_args["tie_weights"]
		if self.tie_weights:
			#Remove <bos> from the vocabulary
			self.word_embeddings = dict_args["word_embeddings"]

		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim) #ToDO
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRUCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		self.linear = nn.Linear(self.hidden_dim, self.vocab_size)
		if self.tie_weights:
			self.linear.weight = self.word_embeddings

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			c_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return c_0
		elif self.rnn_type == 'GRU':
			pass
		elif self.rnn_type == 'RNN':
			pass


	def forward(self, isequence, hidden_t, sequence_mask=None):
		#isequence: batch_size*num_words*iembed
		#hidden_t: batch_size*hidden_dim
		#sequence_mask: batch_size*num_words

		batch_size, num_words, _ = isequence.size()
		isequence = isequence.permute(1,0,2) #isequence: num_words*batch_size*iembed

		h_t = hidden_t
		if self.rnn_type == 'LSTM': c_t = self.init_hidden(batch_size)

		osequence = Variable(isequence.data.new(num_words, batch_size, self.vocab_size).zero_())

		for step in range(num_words):
			input = isequence[step]
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(input, h_t) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			osequence[step] = self.linear(h_t) #batch_size*vocab_size

		osequence = osequence.permute(1,0,2)

		#redundant because we are masking the loss but who cares?
		#osequence = utils.mask_sequence(osequence, sequence_mask) 

		return osequence #batch_size*num_words*vocab_size


if __name__=='__main__':
	pretrainedEmbeddings = PretrainedEmbeddings({"word_embeddings" : torch.randn(10,3), "pretrained_embdim" : 3, "vocabulary_size":10})

	dict_args = {
					'input_dim' : 3, #pretrainedEmbeddings.pretrained_embdim
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'vocabulary_size' : pretrainedEmbeddings.vocabulary_size,
					'tie_weights' : True,
					'word_embeddings' : pretrainedEmbeddings.embeddings.weight
				}

	sentenceDecoder = SentenceDecoder(dict_args)
	osequence = sentenceDecoder(Variable(torch.randn(2,3,3)), Variable(torch.randn(2,3)), Variable(torch.LongTensor([[1,1,1],[1,0,0]])))
	print (osequence)





