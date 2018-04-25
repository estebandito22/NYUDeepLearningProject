import torch
import torch.nn as nn
from torch.autograd import Variable

try: import layers.utils as utils
except: import utils

class BiDirEncoder(nn.Module):

	def __init__(self, dict_args):
		super(BiDirEncoder, self).__init__()
		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.num_layers = dict_args['num_layers']
		self.use_birnn = False
		if self.use_birnn == True:
			self.num_directions = 2
		else:
			self.num_directions = 1
		self.dropout_rate = dict_args['dropout_rate']

		#hidden layer
		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn, dropout=self.dropout_rate)
			#self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn)		
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn, dropout=self.dropout_rate)
			#self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn)
		elif self.rnn_type == 'RNN':
			pass
		#dropout layer
		if self.dropout_rate > 0: self.dropout_layer = nn.Dropout(p=self.dropout_rate)

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			h_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			c_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			return (h_0,c_0)
		elif self.rnn_type == 'GRU':
			h_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			return h_0
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, inputs, lengths):
		#inputs: batch_size*num_words*input_dim
		#lengths: batch_size
		batch_size, num_words, input_dim = inputs.size()

		input_sorted, sorted_lengths, original_indices = utils.sort_batch(inputs, lengths.data) #input_sorted: batch_size*num_words*input_dim
		iembeds = input_sorted.permute(1, 0, 2) #iembeds: num_words*batch_size*input_dim
		if self.dropout_rate > 0: iembeds = self.dropout_layer(iembeds) #iembeds: num_words*batch_size*input_dim
		iembeds = nn.utils.rnn.pack_padded_sequence(iembeds, list(sorted_lengths)) #iembeds: num_words*batch_size*input_dim
		hidden = self.init_hidden(batch_size) 
		rnn_out, h_n = self.rnn(iembeds, hidden) #rnn_out: num_words*batch_size*num_directions.hidden_dim
		if self.rnn_type == 'LSTM': h_n = h_n[0] #h_n: num_layers.num_directions*batch_size*hidden_dim
		if self.use_birnn == False: h_n = h_n[-1].squeeze() #h_n: batch_size*hidden_dim

		(rnn_out, lengths_new) = nn.utils.rnn.pad_packed_sequence(rnn_out) #rnn_out: num_words*batch_size*num_directions.hidden_dim
		rnn_out = rnn_out.permute(1,0,2) #rnn_out: batch_size*num_words*num_directions.hidden_dim
		rnn_out = utils.unsort_batch(rnn_out, original_indices) #rnn_out: batch_size*num_words*num_directions.hidden_dim
		h_n = utils.unsort_batch(h_n, original_indices) #h_n: batch_size*hidden_dim

		#lengths_new = utils.unsort_batch(torch.LongTensor(lengths_new), original_indices) #lengths: <=batch_size
		#return rnn_out, lengths_new #rnn_out: batch_size*num_words*num_directions.hidden_dim, #lengths: <=batch_size
		return rnn_out, h_n, None #rnn_out: batch_size*num_words*num_directions.hidden_dim, #h_n: batch_size*hidden_dim, #lengths: <=batch_size

if __name__=='__main__':
	dict_args = {
				 'input_dim':4, #input dimension
				 'rnn_hdim':3,  #size of the hidden dimension
				 'rnn_type':'LSTM', #RNN, LSTM, GRU
				 'num_layers':1, #number of layers
				 'dropout_rate':0
				}
	net = BiDirEncoder(dict_args)
	inputs = Variable(torch.FloatTensor([[[1.1,-0.2,-1.8,-0.8],[1.2,1.4,-1.5,1.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]], [[1.2,-0.3,0.7,0.9],[0.5,0.6,0.7,-0.3],[0.4,0.3,1.2,1.3],[0.0,0.0,0.0,0.0]]]))
	lengths = torch.LongTensor([2,3])
	rnn_out, h_n, _ = net(inputs, lengths)
	print (h_n)
	print (rnn_out)


