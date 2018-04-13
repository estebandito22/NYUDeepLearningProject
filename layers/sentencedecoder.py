import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

try:
	import layers.utils as utils
	from layers.wordspretrained import PretrainedEmbeddings
except:
	import utils as utils
	from wordspretrained import PretrainedEmbeddings

from layers.beamsearch import BeamSearch


class SentenceDecoder(nn.Module):

	def __init__(self, dict_args):
		super(SentenceDecoder, self).__init__()

		self.pretrained_words_layer = dict_args['pretrained_words_layer']
		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.vocab_size = dict_args["vocabulary_size"]
		self.vocab_bosindex = dict_args["vocabulary_bosindex"]
		self.vocab_eosindex = dict_args["vocabulary_eosindex"]
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
			print(self.linear.weight.requires_grad)


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


		if self.training == True:

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

			osequence_probs = functional.log_softmax(osequence, dim=2)

			return osequence_probs #batch_size*num_words*vocab_size

		else:
			######Greedy inference
			beam = BeamSearch(1, self.vocab_bosindex, self.vocab_eosindex, cuda=False)

			MAX_WORDS = 50
			step = 0
			done = False
			while not done and step < MAX_WORDS:
				input = isequence[step]
				if self.rnn_type == 'LSTM':
					h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
				elif self.rnn_type == 'GRU':
					h_t = self.rnn(input, h_t) #h_t: batch_size*hidden_dim
				elif self.rnn_type == 'RNN':
					pass

				osequence[step] = self.linear(h_t)
				cur_osequence_probs = functional.log_softmax(osequence[step], dim=1)

				done = beam.advance(cur_osequence_probs.t())
				hyp_vector = self.pretrained_words_layer(beam.get_hyp(0)[-1])
				isequence = torch.cat([isequence, hyp_vector.unsqueeze(0)])

				if not done:
					osequence = torch.cat([osequence, osequence[step].unsqueeze(0)])
					step += 1

			osequence = osequence.permute(1,0,2)
			osequence_probs = functional.log_softmax(osequence, dim=2)

			return osequence_probs, beam.get_hyp(0)


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
