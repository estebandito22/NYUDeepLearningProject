import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

from layers.wordspretrained import PretrainedEmbeddings
from layers.visionpretrained import PreTrainedResnet
from layers.sequencedecoder import SequenceDecoder
from layers.sequenceencoder import SequenceEncoder

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
		self.pretrained_embdim = dict_args["word_embdim"]
		self.vocabulary_size = dict_args["vocabulary_size"]
		#self.vocabulary_bosindex = dict_args["vocabulary_bosindex"]
		#self.vocabulary_eosindex = dict_args["vocabulary_eosindex"]

		#FrameEncoderLayer
		self.encoder_configuration = dict_args["encoder_configuration"]

		#SentenceDecoderLayer
		self.decoder_rnn_input_dim = dict_args["decoder_rnn_input_dim"]
		self.decoder_rnn_hidden_dim = dict_args["decoder_rnn_hidden_dim"]
		self.decoder_tie_weights = dict_args["decoder_tie_weights"]
		self.decoder_rnn_type = dict_args["decoder_rnn_type"]
		self.every_step = dict_args["every_step"]
		self.decoder_dropout_rate = dict_args['decoder_dropout_rate']

		#PretrainedVisionLayer
		#pretrained_vision_layer_args = dict_args
		#self.pretrained_vision_layer = PreTrainedResnet(pretrained_vision_layer_args)

		#VisionFeatureDimRedLayer
		self.vision_feature_dimred_layer = nn.Linear(self.pretrained_feature_size, self.decoder_rnn_hidden_dim)

		#PretrainedWordsLayer
		pretrained_words_layer_args = dict_args
		self.pretrained_words_layer = PretrainedEmbeddings(pretrained_words_layer_args)

		#FrameEncoderLayer
		frame_encoder_layer_args = dict_args
		self.frame_encoder_layer = SequenceEncoder(frame_encoder_layer_args)

		#SentenceDecoderLayer
		sentence_decoder_layer_args = {
										'input_dim' : self.decoder_rnn_input_dim,
										'rnn_hdim' : self.decoder_rnn_hidden_dim,
										'rnn_type' : self.decoder_rnn_type,
										'vocabulary_size' : self.vocabulary_size,
										'tie_weights' : self.decoder_tie_weights,
										'word_embeddings' : self.pretrained_words_layer.embeddings.weight,
										#'pretrained_words_layer': self.pretrained_words_layer,
										'every_step': self.every_step,
										'dropout_rate' : self.decoder_dropout_rate
									  }
		self.sentence_decoder_layer = SequenceDecoder(sentence_decoder_layer_args)


	def forward(self, videoframes, videoframes_lengths, inputwords, captionwords_lengths):
		#videoframes : batch_size*num_frames*3*224*224
		#videoframes_lengths : batch_size
		#inputwords : batch_size*num_words
		#outputwords : batch_size*num_words
		#captionwords_lengths : batch_size

		videoframes = videoframes[:,0:videoframes_lengths.data.max()].contiguous()

		videoframes_mask = Variable(utils.sequence_mask(videoframes_lengths))
		#videoframes_mask: batch_size*num_frames

		#redundant because we are masking the loss
		#captionwords_mask = Variable(utils.sequence_mask(captionwords_lengths))
		#captionwords_mask: batch_size*num_words

		#batch_size, num_frames, rgb, height, width = videoframes.size()
		#videoframes = videoframes.view(-1,rgb,height,width).contiguous()
		#videoframes : batch_size.num_frames*3*224*224
		#videoframefeatures = self.pretrained_vision_layer(videoframes)
		#videoframefeatures_fc = videoframefeatures[1]
		#videoframefeatures_fc : batch_size.num_frames*1000

		
		videoframes = videoframes.contiguous()
		batch_size, num_frames, num_features, H, W = videoframes.size()

		if H == 1:
			videoframefeatures_fc = videoframes.view(-1, num_features).contiguous()
			videoframefeatures_fc = self.vision_feature_dimred_layer(videoframefeatures_fc)
			#videoframefeatures_fc : batch_size.num_frames*rnn_hdim
			_, feature_dim = videoframefeatures_fc.size()
			videoframefeatures_fc = videoframefeatures_fc.view(batch_size, num_frames, feature_dim)
			#videoframefeatures_fc : batch_size*num_frames*1000
			videoframefeatures_fc = utils.mask_sequence(videoframefeatures_fc, videoframes_mask)
		else:
			videoframefeatures_fc = videoframes


		inputword_vectors = self.pretrained_words_layer(inputwords)
		#inputword_vectors: batch_size*num_words*wembed_dim

		if not self.training:
			return self.sentence_decoder_layer.inference(self.frame_encoder_layer, videoframefeatures_fc, videoframes_lengths, self.pretrained_words_layer)


		#outputword_values = self.sentence_decoder_layer(inputword_vectors, videoframefeatures_fcmeanpooling, captionwords_mask)
		outputword_log_probabilities = self.sentence_decoder_layer(inputword_vectors, self.frame_encoder_layer, videoframefeatures_fc, videoframes_lengths)
		#outputword_values = batch_size*num_words*vocab_size

		# outputword_log_probabilities = functional.log_softmax(outputword_values, dim=2)
		#outputword_values = batch_size*num_words*vocab_size

		#outputword_log_probabilities = utils.mask_sequence(outputword_log_probabilities, captionwords_mask)

		return outputword_log_probabilities #batch_size*num_words*vocab_size


if __name__=='__main__':
	pretrained_wordvecs = torch.randn(10,3)
	glove_embdim = 3
	dict_args = {
					"intermediate_layers" : ['layer4', 'fc'],
					"pretrained_feature_size" : 1000,

					"word_embeddings" : pretrained_wordvecs,
					"word_embdim" : glove_embdim,
					"use_pretrained_emb" : True,
					"backprop_embeddings" : False,
					"vocabulary_size" : len(pretrained_wordvecs),

					"encoder_configuration" : 'LSTM',
					"encoder_input_dim" : glove_embdim,
					"encoder_rnn_type" : 'LSTM',
					"encoder_rnn_hdim" : glove_embdim,
					"encoder_num_layers" : 1,
					"encoder_dropout_rate" : 0.2,
					"encoderattn_projection_dim" : 2,
					"encoderattn_query_dim" : glove_embdim,

					"decoder_rnn_input_dim" : 2*glove_embdim, 
					"decoder_rnn_hidden_dim" : glove_embdim,
					"decoder_tie_weights" : True,
					"decoder_rnn_type" : 'LSTM',
					"decoder_dropout_rate" : 0.2,
					"every_step" : True
				}
	csal = CSAL(dict_args)
	videoframes = Variable(torch.randn(1, 4, 1000, 1, 1))
	videoframes_lengths = Variable(torch.LongTensor([4]))
	inputwords = Variable(torch.LongTensor([[2,3,5]]))
	#outputwords = Variable(torch.LongTensor([[3,5,2], [9,7,1]]))
	captionwords_lengths = Variable(torch.LongTensor([3]))
	csal.eval()
	outputword_log_probabilities = csal(videoframes, videoframes_lengths, inputwords, captionwords_lengths)
	print(outputword_log_probabilities)


