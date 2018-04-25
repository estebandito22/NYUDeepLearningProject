import torch
import torch.nn as nn
from torch.autograd import Variable

try: import layers.utils as utils
except: import utils

try: import layers.similarity as similarity
except: import similarity

class UniDirAttention(nn.Module):

	def __init__(self, dict_args):
		super(UniDirAttention, self).__init__()
		self.similarity_function = dict_args['similarity_function']

	def forward(self, sequence, vector, sequence_mask, vector2=None, softmax=True):
		#vector: batch_size*iembed1
		#sequence: batch_size*num_words*iembed2
		vector_tiled = vector.unsqueeze(1).expand(vector.size(0), sequence.size(1), vector.size(1)) #vector_tiled: batch_size*num_words*iembed1

		if vector2 is None:
			similarity_vector = self.similarity_function(sequence, vector_tiled) #similarity_vector: batch_size*num_words
		else:
			#To be used only with 'ProjectionSimilaritySharedWeights' else concatenate vectors to form a single vector
			vector2_tiled = vector2.unsqueeze(1).expand(vector2.size(0), sequence.size(1), vector2.size(1)) #vector2_tiled: batch_size*num_words*iembed3
			similarity_vector = self.similarity_function(sequence, vector_tiled, vector2_tiled) #similarity_vector: batch_size*num_words

		sequence_attention_weights = utils.masked_softmax(similarity_vector, sequence_mask.float())

		vector_sequence_attention = utils.attention_pooling(sequence_attention_weights, sequence) #vector_sequence_attention: batch_size*iembed2

		#Will it save some memory?
		#if self.training:
			#sequence_attention_weights = None

		if softmax:
			return vector_sequence_attention, sequence_attention_weights
		else:
			return vector_sequence_attention, similarity_vector			



if __name__=='__main__':
	unidir = UniDirAttention({'similarity_function': 'WeightedSumProjection', 'sequence1_dim':6, 'sequence2_dim':20, 'projection_dim':10})
	vector_sequence_attention, sequence_attention_weights = unidir(Variable(torch.randn(2,5,6)), Variable(torch.randn(1,20).expand(2,20)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3]))), softmax=True)
	print(vector_sequence_attention)
	print(sequence_attention_weights)
