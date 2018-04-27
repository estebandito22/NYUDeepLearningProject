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

class SequenceDecoder(nn.Module):

	def __init__(self, dict_args):
		super(SequenceDecoder, self).__init__()

		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.vocab_size = dict_args["vocabulary_size"]
		self.tie_weights = dict_args["tie_weights"]
		if self.tie_weights:
			self.word_embeddings = dict_args["word_embeddings"]

		self.every_step = dict_args["every_step"]
		#Passed as argument to inference function instead
		#self.pretrained_words_layer = dict_args['pretrained_words_layer']


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
		return Variable(weight.new(batch_size, self.hidden_dim).zero_())

	def forward(self, isequence, encoderlayer, esequence, elengths, imask=None):
		#isequence: batch_size*num_words*iembed
		#esequence: batch_size*num_words*iembed
		#elengths: batch_size
		#imask: batch_size*num_words

		batch_size, num_words, _ = isequence.size()
		isequence = isequence.permute(1,0,2) #isequence: num_words*batch_size*iembed

		if not self.every_step: h_t = encoderlayer(esequence, elengths)
		else: h_t = self.init_hidden(batch_size)

		if self.rnn_type == 'LSTM': c_t = self.init_hidden(batch_size)

		osequence = Variable(isequence.data.new(num_words, batch_size, self.vocab_size).zero_())

		for step in range(num_words):
			input = isequence[step]
			if self.every_step: input = torch.cat((encoderlayer(esequence, elengths, h_t), input), dim = 1)
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

		batch_size, num_words, iembed = esequence.size()
		esequence =  esequence.expand(dict_args['beamsize'], num_words, iembed)
		elengths = elenghts.expand(dict_args['beamsize'])

		input_ix = Variable(torch.LongTensor([dict_args['eosindex']]).expand(dict_args['beamsize']))
		if USE_CUDA: input_ix = input_ix.cuda()
		input_t = embeddding_layer(input_ix) #beamsize*wemb_dim

		if not self.every_step:
			h_t = encoderlayer(esequence, elengths)
			#h_t = h_t.expand(dict_args['beamsize'], h_t.size(1)) #beamsize*hidden_dim
		else: h_t = self.init_hidden(dict_args['beamsize'])

		if self.rnn_type == 'LSTM': c_t = self.init_hidden(dict_args['beamsize'])

		for i in range(15):
			if i!=0:
				input_ix, hidden_ix = beamsearch.get_inputs()
				input_t = embeddding_layer(input_ix) #beamsize*wemb_dim
				#hidden_t = hidden_t.index_select(0, hidden_ix)
				h_t = h_t.index_select(0, hidden_ix)
				if self.rnn_type == 'LSTM': c_t = c_t.index_select(0, hidden_ix)
				#print(hidden_ix.view(1,-1))
			#h_t = hidden_t
			input = input_t
			if self.every_step: input = torch.cat((encoderlayer(esequence, elengths, h_t), input), dim = 1)
			
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: beamsize*hidden_dim
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(input, h_t) #h_t: beamsize*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			#hidden_t = h_t
			ovalues = self.linear(h_t) #beamsize*vocab_size
			oprobs = functional.log_softmax(ovalues, dim=1)
			stop = beamsearch.step(oprobs, i)
			if stop:
				break

		encoderlayer.clear()
		osequence = beamsearch.get_output(index=0)
		return osequence



if __name__=='__main__':


	dict_args = {
					"use_pretrained_emb" : False,
					"backprop_embeddings" : False,
					"word_embeddings" : torch.randn(10,3), 
					"word_embdim" : 3, 
					"vocabulary_size":10
				}

	pretrainedEmbeddings = PretrainedEmbeddings(dict_args)

	dict_args = {
					'input_dim' : 3, #pretrainedEmbeddings.pretrained_embdim
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'vocabulary_size' : pretrainedEmbeddings.vocabulary_size,
					'tie_weights' : True,
					'word_embeddings' : pretrainedEmbeddings.embeddings.weight,
					'pretrained_words_layer' : pretrainedEmbeddings
				}

	sentenceDecoder = SequenceDecoder(dict_args)
	osequence = sentenceDecoder(Variable(torch.randn(2,3,3)), Variable(torch.randn(2,3)), Variable(torch.LongTensor([[1,1,1],[1,0,0]])))
	#print (osequence)

	osequence = sentenceDecoder.inference(Variable(torch.randn(1,3)), pretrainedEmbeddings)
	print (osequence)
