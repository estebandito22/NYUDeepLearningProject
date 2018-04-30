import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

from layers.wordspretrained import PretrainedEmbeddings
from layers.videocaptiondecoder import VideoCaptionDecoder
from layers.videoframeencoder import VideoFrameEncoder

import layers.utils as utils

class STAL(nn.Module):

	def __init__(self, dict_args):
		super(STAL, self).__init__()

		self.word_embeddings = dict_args["word_embeddings"]
		self.pretrained_embdim = dict_args["word_embdim"]
		self.vocabulary_size = dict_args["vocabulary_size"]

		self.encoder_configuration = dict_args["encoder_configuration"]

		self.decoder_rnn_word_dim = dict_args["decoder_rnn_word_dim"]
		self.decoder_rnn_input_dim = dict_args["decoder_rnn_input_dim"]
		self.decoder_rnn_hidden_dim = dict_args["decoder_rnn_hidden_dim"]
		self.decoder_rnn_type = dict_args["decoder_rnn_type"]
		self.every_step = dict_args["every_step"]
		self.decoder_top_dropout_rate = dict_args["decoder_top_dropout_rate"]
		self.decoder_bottom_dropout_rate = dict_args["decoder_bottom_dropout_rate"]
		self.decoder_residual_connection = dict_args["residual_connection"]	
	
		#PretrainedWordsLayer
		pretrained_words_layer_args = dict_args
		self.pretrained_words_layer = PretrainedEmbeddings(pretrained_words_layer_args)
		frame_encoder_layer_args = dict_args

		#FrameEncoderLayer
		self.frame_encoder_layer = VideoFrameEncoder(frame_encoder_layer_args)

		#SentenceDecoderLayer
		sentence_decoder_layer_args = {
										'word_dim' : self.decoder_rnn_word_dim,
										'input_dim' : self.decoder_rnn_input_dim,
										'rnn_hdim' : self.decoder_rnn_hidden_dim,
										'rnn_type' : self.decoder_rnn_type,
										'vocabulary_size' : self.vocabulary_size,
										'every_step': self.every_step,
										'top_dropout_rate' : self.decoder_top_dropout_rate,
										'bottom_dropout_rate' : self.decoder_bottom_dropout_rate,
										'residual_connection' : self.decoder_residual_connection
									  }
		self.sentence_decoder_layer = VideoCaptionDecoder(sentence_decoder_layer_args)


	def forward(self, videoframes, videoframes_lengths, inputwords=None, captionwords_lengths=None):
		#videoframes : batch_size*num_frames*256*3*3
		#videoframes_lengths : batch_size
		#inputwords : batch_size*num_words
		#outputwords : batch_size*num_words
		#captionwords_lengths : batch_size

		#Remove additional padding after truncation
		videoframes = videoframes[:,0:videoframes_lengths.data.max()].contiguous()

		if self.training: inputword_vectors = self.pretrained_words_layer(inputwords)
		#inputword_vectors: batch_size*num_words*wembed_dim

		if not self.training:
			return self.sentence_decoder_layer.inference(	
															self.frame_encoder_layer, 
															videoframes, 
															videoframes_lengths, 
															self.pretrained_words_layer
														)

		outputword_log_probabilities = self.sentence_decoder_layer(
																	inputword_vectors, 
																	self.frame_encoder_layer, 
																	videoframes, 
																	videoframes_lengths
																  )

		return outputword_log_probabilities #batch_size*num_words*vocab_size




if __name__=='__main__':
	pretrained_wordvecs = torch.randn(10,3)
	glove_embdim = 3
	hidden_dim = 12
	dict_args = {

					"word_embeddings" : pretrained_wordvecs,
					"word_embdim" : glove_embdim,
					"use_pretrained_emb" : True,
					"backprop_embeddings" : False,
					"vocabulary_size" : len(pretrained_wordvecs),

					"encoder_configuration" : 'LSTMTrackSpatialTemporal',
					"frame_channel_dim" : 256,
					"frame_spatial_dim" : 2,
					"encoder_rnn_type" : 'LSTM',
					"encoder_rnn_hdim" : hidden_dim,
					"encoder_dropout_rate" : 0.2,
					"encoderattn_projection_dim" : hidden_dim/2,
					"encoderattn_query_dim" : hidden_dim,

					"decoder_rnn_word_dim" : glove_embdim,
					"decoder_rnn_input_dim" : hidden_dim + hidden_dim, #channel_dim 
					"decoder_rnn_hidden_dim" : hidden_dim,
					"decoder_rnn_type" : 'LSTM',
					"every_step" : True,
					"decoder_top_dropout_rate" : 0.2,
                                        "decoder_bottom_dropout_rate" : 0.2,
                                        "residual_connection" : True
				}
	stal = STAL(dict_args)

	videoframes = Variable(torch.randn(1, 4, 256, 2, 2))
	videoframes_lengths = Variable(torch.LongTensor([3]))
	inputwords = Variable(torch.LongTensor([[2,3,5]]))
	#outputwords = Variable(torch.LongTensor([[3,5,2], [9,7,1]]))
	captionwords_lengths = Variable(torch.LongTensor([3]))
	outputword_log_probabilities = stal(videoframes, videoframes_lengths, inputwords, captionwords_lengths)
	#print(outputword_log_probabilities)

	stal.eval()
	outputword_log_probabilities = stal(videoframes, videoframes_lengths, inputwords, captionwords_lengths)
	print(outputword_log_probabilities)



