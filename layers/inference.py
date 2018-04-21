import torch
import torch.nn as nn
from torch.autograd import Variable

try: import layers.utils as utils
except: import utils as utils

USE_CUDA = False
if torch.cuda.is_available():
        USE_CUDA = True

class BeamSearch(object):
	#Works with batchsize of 1

	def __init__(self, dict_args):
		
		self.beamsize = dict_args["beamsize"]
		self.eosindex = dict_args["eosindex"]
		self.bosindex = dict_args["bosindex"]

		self.scores = Variable(torch.zeros(self.beamsize)) #beamsize
		self.scores[1:] = -float("Inf")

		self.children = []
		self.hidden = []

		initial = Variable(torch.LongTensor(self.beamsize).fill_(self.eosindex))
		initial[0] = self.bosindex

		if USE_CUDA:
			self.scores = self.scores.cuda()
			initial = initial.cuda()
	
		self.children.append(initial)

	def get_inputs(self):
		#return Variable(self.children[-1]), Variable(self.hidden[-1])
		return self.children[-1], self.hidden[-1]

	def get_output(self, index=0):
		osequence = []
		for i in range(len(self.children)):
			j = len(self.children) - i - 1
			osequence.append(self.children[j].data[index])
			index = self.hidden[j-1].data[index]
		osequence.reverse()
		return osequence	

	def step(self, currentprobs, i):
		#currentprobs: beamsize*vocab_size

		if i==0: currentprobs = currentprobs[0].view(1,-1)

		currentscore = self.scores.clone().unsqueeze(1)
		beamscores = currentprobs + currentscore #beamsize*vocab_size
		
		num_children, vocab_size = beamscores.size()

		beamscores = beamscores.view(-1)

		topscores, topindices = torch.topk(beamscores, self.beamsize, 0, True, True)

		hiddenindices = topindices/vocab_size
		childrenindices = topindices - hiddenindices*vocab_size

		'''print(currentprobs)
		print(currentscore)
		print(self.scores)
		print(parentindices)
		print(hiddenindices)
		print(childrenindices)'''
		
		self.scores = topscores
		self.hidden.append(hiddenindices)
		self.children.append(childrenindices)

		if childrenindices.data[0] == self.eosindex:
			return True
		return False


if __name__=='__main__':

	dict_args = {
					'beamsize' : 2,
					'eosindex' : 0,
					'bosindex' : 1
				} 

	beamsearch = BeamSearch(dict_args)

	for i in range(3):
		currentprobs = Variable(torch.randn(2,5))
		if USE_CUDA: currentprobs = currentprobs.cuda()
		beamsearch.step(currentprobs, i)
		c, h = beamsearch.get_inputs()

	print(beamsearch.children)
	print(beamsearch.get_output(1))



