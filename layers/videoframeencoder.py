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


class VideoFrameEncoder(nn.Module):

	def __init__(self, dict_args):
		super(VideoFrameEncoder, self).__init__()

		self.configuration = dict_args['encoder_configuration']
		self.dropout_rate = dict_args['encoder_dropout_rate']

		self.channel_dim = dict_args['frame_channel_dim']
		self.spatial_dim = dict_args['frame_spatial_dim']

		self.projection_dim = dict_args['encoderattn_projection_dim']
		self.use_linear = dict_args["encoder_linear"]

		if self.use_linear:
			self.channelred_dim = dict_args["frame_channelred_dim"]
			self.linear = nn.Linear(self.channel_dim, self.channelred_dim)
			self.channel_dim = self.channelred_dim
	
		self.recurrent = None
		self.spatialattention = None
		self.temporalattention = None

		self.dropout_layer = nn.Dropout(p=self.dropout_rate)

		if self.configuration == 'LSTMTrackSpatial':
			self.recurrent, self.spatialattention, self.temporalattention = True, True, False
		elif self.configuration == 'LSTMTrackSpatialTemporal':
			self.recurrent, self.spatialattention, self.temporalattention = True, True, True
		elif self.configuration == 'SpatialTemporal':
			self.recurrent, self.spatialattention, self.temporalattention = False, True, True


		if self.recurrent:
			self.hidden_dim = dict_args['encoder_rnn_hdim']
			self.rnn_type = dict_args['encoder_rnn_type']

			if self.rnn_type == 'LSTM':
				self.trackrnn = nn.LSTMCell(self.channel_dim, self.hidden_dim)
			elif self.rnn_type == 'GRU':
				self.trackrnn = nn.GRUCell(self.channel_dim, self.hidden_dim)
			elif self.rnn_type == 'RNN':
				pass


		if self.spatialattention:

			self.query_dim = dict_args['encoderattn_query_dim']

			spatial_similarity_function_args = {
												'sequence1_dim' : self.channel_dim,
												'sequence2_dim' : self.query_dim,
												'projection_dim' : self.projection_dim
									   		   }
			self.spatial_similarity_function = similarity.LinearProjectionSimilarity(spatial_similarity_function_args)
			self.spatial_attention_function = UniDirAttention({'similarity_function': self.spatial_similarity_function})

		if self.temporalattention:

			self.query_dim = dict_args['encoderattn_query_dim']
			self.context_dim = self.channel_dim
			if self.recurrent: self.context_dim = self.hidden_dim

			temporal_similarity_function_args = {
												'sequence1_dim' : self.context_dim,
												'sequence2_dim' : self.query_dim,
												'projection_dim' : self.projection_dim
									   		   }
			self.temporal_similarity_function = similarity.LinearProjectionSimilarity(temporal_similarity_function_args)
			self.temporal_attention_function = UniDirAttention({'similarity_function': self.temporal_similarity_function})


	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		return Variable(weight.new(batch_size, self.hidden_dim).zero_())


	def forward(self, isequence, ilenghts, spatialqueryvector = None, temporalqueryvector = None):
		#isequence : batch_size*num_frames*C*H*W
		#ilenghts : batch_size
		#spatialqueryvector Attention over the frames of the video
		#temporalqueryvector Attention over hidden states of tracking LSTM

		contextvector, temporalvectors, spatialvectors = None, None, None
		
		isequence = self.dropout_layer(isequence)		

		batch_size, num_frames, channel_dim, height, width = isequence.size()
		isequence = isequence.view(batch_size, num_frames, channel_dim, -1)
		#isequence : batch_size*num_frames*channel_dim*num_blocks
		isequence = isequence.permute(1, 0, 3, 2)
		#isequence : num_frames*batch_size*num_blocks*channel_dim
		num_blocks = isequence.size(2)

		if self.use_linear:
			isequence = functional.relu(self.linear(isequence))

		#Dropout?
	
		if self.temporalattention:
			temporalvectors = Variable(isequence.data.new(num_frames, batch_size, self.context_dim).zero_())

		if self.recurrent:
			h_t = spatialqueryvector
			if self.rnn_type == 'LSTM': c_t = self.init_hidden(batch_size)

			for step in range(num_frames):
				spatialvectors = isequence[step] #fsequence: batch_size*num_blocks*channel_dim
				slenghts = ilenghts.data.new(batch_size).zero_() + num_blocks
				smask = Variable(utils.sequence_mask(Variable(slenghts)))
				spatattnvector, spatattnweights = self.spatial_attention_function(spatialvectors, h_t, smask)
				#print(spatattnweights)

				if self.rnn_type == 'LSTM':
					h_t, c_t = self.trackrnn(spatattnvector, (h_t, c_t)) #h_t: batch_size*hidden_dim
				elif self.rnn_type == 'GRU':
					h_t = self.trackrnn(spatattnvector, h_t) #h_t: batch_size*hidden_dim
				elif self.rnn_type == 'RNN':
					pass

				if self.temporalattention: temporalvectors[step] = h_t
			contextvector = h_t		
		else:

			for step in range(num_frames):
				spatialvectors = isequence[step] #fsequence: batch_size*num_blocks*channel_dim
				slenghts = ilenghts.data.new(batch_size).zero_() + num_blocks
				smask = Variable(utils.sequence_mask(Variable(slenghts)))
				spatattnvector, spatattnweights = self.spatial_attention_function(spatialvectors, spatialqueryvector, smask)
				#print(spatattnweights)

				temporalvectors[step] = spatattnvector
			contextvector = None #Should undergo temporal attention


		if self.temporalattention:
			imask = Variable(utils.sequence_mask(ilenghts))
			temporalvectors = temporalvectors.permute(1, 0, 2)
			#temporalvectors: batch_size*num_frames*context_dim

			tempattnvector, tempattnweights = self.temporal_attention_function(temporalvectors, temporalqueryvector, imask)
			contextvector = tempattnvector
			#print(tempattnweights)

		return contextvector

	def clear(self):
		return



if __name__=='__main__':

	dict_args = {
					'encoder_configuration' : 'SpatialTemporal',
					'frame_channel_dim' : 6,
					'frame_spatial_dim' : 3,
					'encoder_dropout_rate' : 0.2,
					'encoder_rnn_hdim' : 4,
					'encoder_rnn_type' : 'LSTM',
					'encoderattn_projection_dim' : 2,
					'encoderattn_query_dim' : 4
				}

	isequence = Variable(torch.randn(2,3,6,3,3))
	ilenghts = Variable(torch.LongTensor([2,3]))
	
	sequenceencoder = VideoFrameEncoder(dict_args)

	for i in range(4):
		squery = Variable(torch.randn(2,4))
		tquery = Variable(torch.randn(2,4))
		print(sequenceencoder(isequence, ilenghts, squery, tquery))
		sequenceencoder.clear()


