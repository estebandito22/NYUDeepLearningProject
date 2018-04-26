import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

try: import layers.utils as utils
except: import utils as utils

try: import layers.similarity as similarity
except: import similarity

try: from layers.unidirattention import UniDirAttention
except: from unidirattention import UniDirAttention

try: from layers.bidirencoder import BiDirEncoder
except: from bidirencoder import BiDirEncoder

USE_CUDA = False
if torch.cuda.is_available():
        USE_CUDA = True

class SequenceEncoder(nn.Module):

	def __init__(self, dict_args):
		super(SequenceEncoder, self).__init__()

		self.configuration = dict_args['encoder_configuration']
		self.dropout_rate = dict_args['encoder_dropout_rate']
		self.input_dim = dict_args['encoder_input_dim']

		self.contextvector = None
		self.inputvector = None
		self.recurrent, self.attention = None, None

		if self.configuration == 'MP':
			self.recurrent, self.attention = False, False
		elif self.configuration == 'LSTM':
			self.recurrent, self.attention = True, False
		elif self.configuration == 'MPAttn':
			self.recurrent, self.attention = False, True
		elif self.configuration == 'LSTMAttn':
			self.recurrent, self.attention = True, True

		if self.recurrent:
			self.hidden_dim = dict_args['encoder_rnn_hdim']
			self.rnn_type = dict_args['encoder_rnn_type']
			self.num_layers = dict_args['encoder_num_layers']

			recurrent_layer_args = {
									'input_dim': self.input_dim,
									'rnn_hdim': self.hidden_dim, 
									'rnn_type': self.rnn_type, 
									'num_layers': self.num_layers, 
									'dropout_rate': self.dropout_rate
								   }
			self.recurrent_layer = BiDirEncoder(recurrent_layer_args)

		if self.attention:
			self.projection_dim = dict_args['encoderattn_projection_dim']

			if self.configuration == 'LSTMAttn': self.context_dim = self.hidden_dim
			elif self.configuration == 'MPAttn': self.context_dim = self.input_dim

			self.query_dim = dict_args['encoderattn_query_dim']
			similarity_function_args = {
											'sequence1_dim' : self.context_dim,
											'sequence2_dim' : self.query_dim,
											'projection_dim' : self.projection_dim
									   }
			self.similarity_function = similarity.LinearProjectionSimilarity(similarity_function_args)
			self.attention_function = UniDirAttention({'similarity_function': self.similarity_function})


	def forward(self, isequence, ilenghts, queryvector = None):
		#isequence : batch_size*num_steps*input_emb
		#ilenghts : batch_size

		contextvector = None
		imask = Variable(utils.sequence_mask(ilenghts))

		if self.configuration == 'MP':
			if self.contextvector is not None:
				contextvector =  self.contextvector
			else:
				#print("MP")
				contextvector = isequence.sum(dim = 1)
				contextvector = contextvector.div(ilenghts.unsqueeze(1).float())
				#contextvector : batch_size*input_emb
				self.contextvector = contextvector

		if self.recurrent:
			if self.contextvector is not None and self.inputvector is not None:
				contextvector  = self.contextvector
				isequence = self.inputvector
			else:
				#print("Recurrent")
				isequence, ihidden, _ = self.recurrent_layer(isequence, ilenghts)
				#isequence : batch_size*num_steps*hidden_dim
				#ihidden : batch_size*hidden_dim
				contextvector = ihidden
				self.contextvector = contextvector
				self.inputvector = isequence

		if self.attention:
			#print("Attention")
			vector_sequence_attention, sequence_attention_weights = self.attention_function(isequence, queryvector, imask, softmax=True)
			contextvector = vector_sequence_attention
			#contextvector : batch_size*input_emb/hidden_dim
		return contextvector

	def clear(self):
		del self.contextvector
		del self.inputvector
		self.contextvector = None
		self.inputvector = None

if __name__=='__main__':

	dict_args = {
					'encoder_configuration' : 'LSTMAttn',
					'encoder_input_dim' : 4,
					'encoder_dropout_rate' : 0.2,
					'encoder_rnn_hdim' : 3,
					'encoder_rnn_type' : 'LSTM',
					'encoder_num_layers' : 1,
					'encoderattn_projection_dim' : 2,
					'encoderattn_query_dim' : 3
				}

	isequence = Variable(torch.randn(2,3,4))
	ilenghts = Variable(torch.LongTensor([2,3]))
	
	sequenceencoder = SequenceEncoder(dict_args)

	for i in range(4):
		query = Variable(torch.randn(2,3))
		print(sequenceencoder(isequence, ilenghts, query))
		sequenceencoder.clear()







