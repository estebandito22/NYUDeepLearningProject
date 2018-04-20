import os
import sys
import time
import pickle

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
torchvision.set_image_backend('accimage')

try:
	from input.dataiter import *
	from input.datautils import *
except:
	from dataiter import *
	from datautils import *

class PreTrainedResnet(nn.Module):

	def __init__(self, dict_args):
		super(PreTrainedResnet, self).__init__()

		self.intermediate_layers = dict_args['intermediate_layers']
		self.pretrained_model = models.resnet18(pretrained=True).eval()
		if torch.cuda.is_available():
			self.pretrained_model = self.pretrained_model.cuda()
		for param in self.pretrained_model.parameters():
			param.requires_grad = False

	def forward(self, x):
		intermediate_features = []
		for name, module in self.pretrained_model._modules.items():
			if name=='fc':
				x = x.squeeze().contiguous()
			x = module(x)
			if name in self.intermediate_layers:
				intermediate_features += [x]
		return intermediate_features

#Class name to be changed
class Pixel():

	def __init__(self, files, pklfilepath):
		self.files = files
		self.pklfilepath = pklfilepath
		self.video2vec = {}
		self.pretrained = PreTrainedResnet({'intermediate_layers':['layer4', 'fc']})

	def create(self):
		for _, videoidspath, framesfolderpath in self.files:
			videoids = json.load(open(videoidspath, 'r'))
			for videoid in itervideos(videoids):
				start_time = time.time()
				imagefiles = [imagefile for imagefile in iterimages(framesfolderpath, videoid)]
				file_time = time.time()
				imageframes = [imagetotensor(imagefile) for imagefile in imagefiles]
				frame_time = time.time()
				imagefeatures = [self.pretrained(Variable(frame.unsqueeze(0).cuda()))[1].view(-1).unsqueeze(-1).unsqueeze(-1).data.cpu() for frame in imageframes]
				#imagefeatures = self.pretrained(Variable(torch.stack(imageframes).cuda()))[1].data.cpu()
				#imagefeatures = torch.randn(45, 1000)
				#imagefeatures = imagefeatures.unsqueeze(-1).unsqueeze(-1)
				#imagefeatures = list(imagefeatures)
				#imagefeatures = [torch.randn(1000) for i in range(45)]
				self.video2vec[videoid] = imagefeatures

				feat_time = time.time()
				print(videoid, len(imagefeatures), imagefeatures[0].shape)
				print('File : {0}, Frame : {1}, Feat : {2}'.format(file_time-start_time, frame_time-file_time, feat_time-frame_time))
		return

	def get_pixel_vectors(self, videoid):
		#index error if video not found
		return self.video2vec[videoid]

	def save(self):
		pklfile = open(self.pklfilepath, 'wb')
		pickle.dump(self.video2vec, pklfile)
		pklfile.close()
		print("saving the pklfile to {0}".format(self.pklfilepath))

	def load(self):
		pklfile = open(self.pklfilepath, 'rb')
		self.video2vec = pickle.load(pklfile)
		pklfile.close()


if __name__ == '__main__':

	pklfilepath = 'MSRVTT/trainvideo.pkl'
	pixel = Pixel([('Dummy', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')], pklfilepath)

	#pixel.create()
	#pixel.save()
	pixel.load()
	print(len(pixel.get_pixel_vectors('video0')))
	print(pixel.get_pixel_vectors('video1')[0])
