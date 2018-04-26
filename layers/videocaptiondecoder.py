import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

try: import layers.utils as utils
except: import utils as utils

try: from layers.wordspretrained import PretrainedEmbeddings
except: from wordspretrained import PretrainedEmbeddings

try: from layers.inference import BeamSearch
except: from inference import BeamSearch

USE_CUDA = False
if torch.cuda.is_available():
		USE_CUDA = True


class VideoCaptionDecoder(nn.Module):

	def __init__(self, dict_args):
		super(VideoCaptionDecoder, self).__init__()

		self.word_dim = dict_args['word_dim']
		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.vocab_size = dict_args["vocabulary_size"]

		self.every_step = dict_args["every_step"]

		if self.rnn_type == 'LSTM':
			self.toprnn = nn.LSTMCell(self.input_dim, self.hidden_dim)
			self.bottomrnn = nn.LSTMCell(self.word_dim, self.hidden_dim)
		elif self.rnn_type == 'GRU':
			self.toprnn = nn.GRUCell(self.input_dim, self.hidden_dim)
			self.bottomrnn = nn.GRUCell(self.word_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		self.linear = nn.Linear(self.hidden_dim, self.vocab_size)

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		return Variable(weight.new(batch_size, self.hidden_dim).zero_())

	def forward(self, isequence, encoderlayer, esequence, elengths, imask=None):
		#isequence: batch_size*num_words*iembed
		#esequence: batch_size*num_frames*C*H*W
		#elengths: batch_size
		#imask: batch_size*num_words

		batch_size, num_words, _ = isequence.size()
		isequence = isequence.permute(1,0,2) #isequence: num_words*batch_size*iembed

		#Initialize even if it is not every step?
		if not self.every_step: h_ttop = encoderlayer(esequence, elengths)
		else: h_ttop = self.init_hidden(batch_size)

		h_tbottom = self.init_hidden(batch_size)

		if self.rnn_type == 'LSTM':
			c_ttop = self.init_hidden(batch_size)
			c_tbottom = self.init_hidden(batch_size)

		osequence = Variable(isequence.data.new(num_words, batch_size, self.vocab_size).zero_())

		for step in range(num_words):
			word = isequence[step]

			if self.rnn_type == 'LSTM':
				h_tbottom, c_tbottom = self.bottomrnn(word, (h_tbottom, c_tbottom)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_tbottom = self.bottomrnn(word, h_tbottom) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			input = h_tbottom
			if self.every_step: input = torch.cat((encoderlayer(esequence, elengths, h_tbottom, h_ttop), input), dim = 1)
			if self.rnn_type == 'LSTM':
				h_ttop, c_ttop = self.toprnn(input, (h_ttop, c_ttop)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_ttop = self.toprnn(input, h_ttop) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			osequence[step] = self.linear(h_ttop) #batch_size*vocab_size

		osequence = osequence.permute(1,0,2)

		#redundant because we are masking the loss
		#osequence = utils.mask_sequence(osequence, sequence_mask)

		osequence_probs = functional.log_softmax(osequence, dim=2)
		encoderlayer.clear()
		return osequence_probs #batch_size*num_words*vocab_size

	#Works only with batch size of 1 as of now
	def inference(self, encoderlayer, esequence, elengths, embeddding_layer):

		dict_args = {
					'beamsize' : 2,
					'eosindex' : 0, #remove hardcoding
					'bosindex' : 1  #remove hardcoding
					} 

		beamsearch = BeamSearch(dict_args)

		word_ix = Variable(torch.LongTensor([dict_args['eosindex']]).expand(dict_args['beamsize']))
		#word_ix = Variable(torch.LongTensor([dict_args['eosindex']]))
		if USE_CUDA: word_ix = word_ix.cuda()
		word_t = embeddding_layer(word_ix) #beamsize*wemb_dim

		if not self.every_step:
			h_ttop = encoderlayer(esequence, elengths)
			h_ttop = h_ttop.expand(dict_args['beamsize'], h_ttop.size(1)) #beamsize*hidden_dim
		else: h_ttop = self.init_hidden(dict_args['beamsize'])

		h_tbottom = self.init_hidden(dict_args['beamsize'])

		if self.rnn_type == 'LSTM':
			c_ttop = self.init_hidden(dict_args['beamsize'])
			c_tbottom = self.init_hidden(dict_args['beamsize'])


		for i in range(15):
			if i!=0:
				word_ix, hidden_ix = beamsearch.get_inputs()
				word_t = embeddding_layer(word_ix) #beamsize*wemb_dim

				h_ttop = h_ttop.index_select(0, hidden_ix)
				h_tbottom = h_tbottom.index_select(0, hidden_ix)
				if self.rnn_type == 'LSTM':
					c_ttop = c_ttop.index_select(0, hidden_ix)
					c_tbottom = c_tbottom.index_select(0, hidden_ix)

			word = word_t

			if self.rnn_type == 'LSTM':
				h_tbottom, c_tbottom = self.bottomrnn(word, (h_tbottom, c_tbottom)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_tbottom = self.bottomrnn(word, h_tbottom) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			input = h_tbottom
			if self.every_step: input = torch.cat((encoderlayer(esequence, elengths, h_tbottom, h_ttop), input), dim = 1)
			if self.rnn_type == 'LSTM':
				h_ttop, c_ttop = self.toprnn(input, (h_ttop, c_ttop)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_ttop = self.toprnn(input, h_ttop) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			ovalues = self.linear(h_ttop) #beamsize*vocab_size
			oprobs = functional.log_softmax(ovalues, dim=1)
			stop = beamsearch.step(oprobs, i)
			if stop:
				break

		encoderlayer.clear()
		osequence = beamsearch.get_output(index=0)
		return osequence

