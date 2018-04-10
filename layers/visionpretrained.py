import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class PreTrainedResnet(nn.Module):

	def __init__(self, dict_args):
		super(PreTrainedResnet, self).__init__()

		self.intermediate_layers = dict_args['intermediate_layers']
		self.pretrained_model = models.resnet18(pretrained=True)

	def forward(self, x):
		intermediate_features = []
		for name, module in self.pretrained_model._modules.items():
			x = module(x).squeeze()
			if name in self.intermediate_layers:
				intermediate_features += [x]
		return intermediate_features

if __name__=='__main__':
	dict_args = {
				 'intermediate_layers':['layer4', 'fc']
				}
	resnet = PreTrainedResnet(dict_args)

	input = Variable(torch.randn(2,3,224,224))
	output = resnet(input)

	print(output[0].data.shape)
	print(output[1].data.shape)

