import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

from layers.wordspretrained import PretrainedEmbeddings
from layers.visionpretrained import PreTrainedResnet
from layers.sentencedecoder import SentenceDecoder

import layers.utils as utils

class CSAL(nn.Module):

	def __init__(self, dict_args):
		super(CSAL, self).__init__()

		#PretrainedVisionLayer
		#self.intermediate_layers = dict_args["intermediate_layers"]

		#VisionFeatureDimRedLayer
		self.pretrained_feature_size = dict_args["pretrained_feature_size"]

		#PretrainedWordsLayer
		self.word_embeddings = dict_args["word_embeddings"]
		self.pretrained_embdim = dict_args["pretrained_embdim"]
		self.vocabulary_size = dict_args["vocabulary_size"]

		#SentenceDecoderLayer
		self.decoder_rnn_hidden_dim = dict_args["decoder_rnn_hidden_dim"]
		self.decoder_tie_weights = dict_args["decoder_tie_weights"]
		self.decoder_rnn_type = dict_args["decoder_rnn_type"]


		#PretrainedVisionLayer
		pretrained_vision_layer_args = dict_args
		self.pretrained_vision_layer = PreTrainedResnet(pretrained_vision_layer_args)

		#VisionFeatureDimRedLayer
		self.vision_feature_dimred_layer = nn.Linear(self.pretrained_feature_size, self.pretrained_embdim)

		#PretrainedWordsLayer
		pretrained_words_layer_args = dict_args
		self.pretrained_words_layer = PretrainedEmbeddings(pretrained_words_layer_args)

		#SentenceDecoderLayer
		sentence_decoder_layer_args = {
										'input_dim' : self.pretrained_embdim, 
										'rnn_hdim' : self.decoder_rnn_hidden_dim,
										'rnn_type' : self.decoder_rnn_type,
										'vocabulary_size' : self.vocabulary_size,
										'tie_weights' : True,
										'word_embeddings' : self.pretrained_words_layer.embeddings.weight
									  }
		self.sentence_decoder_layer = SentenceDecoder(sentence_decoder_layer_args)


	def forward(self, videoframes, videoframes_lengths, inputwords, captionwords_lengths):
		#videoframes : batch_size*num_frames*3*224*224
		#videoframes_lengths : batch_size
		#inputwords : batch_size*num_words
		#outputwords : batch_size*num_words
		#captionwords_lengths : batch_size

		videoframes_mask = Variable(utils.sequence_mask(videoframes_lengths)) 
		#videoframes_mask: batch_size*num_frames

		#redundant because we are masking the loss
		#captionwords_mask = Variable(utils.sequence_mask(captionwords_lengths)) 
		#captionwords_mask: batch_size*num_words

		batch_size, num_frames, rgb, height, width = videoframes.size()
		videoframes = videoframes.view(-1,rgb,height,width)	
		#videoframes : batch_size.num_frames*3*224*224	
		videoframefeatures = self.pretrained_vision_layer(videoframes)
		videoframefeatures_fc = videoframefeatures[1] 
		#videoframefeatures_fc : batch_size.num_frames*1000
		videoframefeatures_fc = self.vision_feature_dimred_layer(videoframefeatures_fc)
		#videoframefeatures_fc : batch_size.num_frames*rnn_hdim
		_, feature_dim = videoframefeatures_fc.size()
		videoframefeatures_fc = videoframefeatures_fc.view(batch_size, num_frames, feature_dim) 
		#videoframefeatures_fc : batch_size*num_frames*1000
		videoframefeatures_fc = utils.mask_sequence(videoframefeatures_fc, videoframes_mask)

		videoframefeatures_fcmeanpooling = videoframefeatures_fc.sum(dim = 1)
		videoframefeatures_fcmeanpooling = videoframefeatures_fcmeanpooling.div(videoframes_lengths.unsqueeze(1).float())

		inputword_vectors = self.pretrained_words_layer(inputwords) 
		#inputword_vectors: batch_size*num_words*wembed_dim

		#outputword_values = self.sentence_decoder_layer(inputword_vectors, videoframefeatures_fcmeanpooling, captionwords_mask)
		outputword_values = self.sentence_decoder_layer(inputword_vectors, videoframefeatures_fcmeanpooling)
		#outputword_values = batch_size*num_words*vocab_size

		outputword_log_probabilities = functional.log_softmax(outputword_values, dim=2)
		#outputword_values = batch_size*num_words*vocab_size

		#outputword_log_probabilities = utils.mask_sequence(outputword_log_probabilities, captionwords_mask)

		return outputword_log_probabilities #batch_size*num_words*vocab_size


if __name__=='__main__':
	pretrained_wordvecs = torch.randn(10,3)
	glove_embdim = 3
	dict_args = {
					"intermediate_layers" : ['layer4', 'fc'],
					"word_embeddings" : pretrained_wordvecs,
					"pretrained_embdim" : glove_embdim,
					"vocabulary_size" : len(pretrained_wordvecs),
					"decoder_rnn_hidden_dim" : glove_embdim,
					"decoder_tie_weights" : True,
					"decoder_rnn_type" : 'LSTM',
					"pretrained_feature_size" : 1000
				}
	csal = CSAL(dict_args)
	videoframes = Variable(torch.randn(2, 4, 3, 224, 224))
	videoframes_lengths = Variable(torch.LongTensor([4,2]))
	inputwords = Variable(torch.LongTensor([[2,3,5], [7,9,1]]))
	#outputwords = Variable(torch.LongTensor([[3,5,2], [9,7,1]]))
	captionwords_lengths = Variable(torch.LongTensor([3,2]))
	outputword_log_probabilities = csal(videoframes, videoframes_lengths, inputwords, captionwords_lengths)
	print(outputword_log_probabilities)


