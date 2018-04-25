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


class PreTrainedAlexnet(nn.Module):

	def __init__(self, dict_args):
		super(PreTrainedAlexnet, self).__init__()

		self.spatial_boolean = dict_args['spatial_boolean']
		self.pretrained_model = models.alexnet(pretrained=True).eval()
		if torch.cuda.is_available():
			self.pretrained_model = self.pretrained_model.cuda()

		if self.spatial_boolean:
			self.features = nn.Sequential(*list(self.pretrained_model.features.children()))
		else:
			self.features = self.pretrained_model

	def forward(self, x):
		x = self.features(x)
		x = nn.functional.avg_pool2d(x, 2, 2)
		#x = nn.functional.max_pool2d(x, 4, 1)
		return x


#Class name to be changed
class Pixel():

	def __init__(self, files, pklfilepath):
		self.files = files
		self.pklfilepath = pklfilepath
		self.video2vec = {}
		#self.pretrained = PreTrainedResnet({'intermediate_layers':['layer4', 'fc']})
		self.pretrained = PreTrainedAlexnet({'spatial_boolean' : True})

	def create(self):
		for _, videoidspath, framesfolderpath in self.files:
			videoids = json.load(open(videoidspath, 'r'))
			for videoid in itervideos(videoids):
				start_time = time.time()
				imagefiles = [imagefile for imagefile in iterimages(framesfolderpath, videoid)]
				file_time = time.time()
				imageframes = [imagetotensor(imagefile) for imagefile in imagefiles]
				frame_time = time.time()
				#imagefeatures = [self.pretrained(Variable(frame.unsqueeze(0).cuda())).view(-1).unsqueeze(-1).unsqueeze(-1).data.cpu() for frame in imageframes]
				imagefeatures = [self.pretrained(Variable(frame.unsqueeze(0).cuda())).squeeze().data.cpu() for frame in imageframes]

				self.video2vec[videoid] = imagefeatures

				feat_time = time.time()
				print(videoid, len(imagefeatures), imagefeatures[0].shape)
				print('File : {0}, Frame : {1}, Feat : {2}'.format(file_time-start_time, frame_time-file_time, feat_time-frame_time))

				'''imagefeatures = [Variable(torch.randn(256, 3, 3)).data.cpu() for frame in imagefiles]
                                
                                self.video2vec[videoid] = imagefeatures
				print(videoid, len(imagefeatures), imagefeatures[0].shape)'''
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

	pklfilepath = 'MSRVTT/Pixel/Alexnet25633/trainvideo.pkl'
	#pklfilepath = 'MSRVTT/valvideo.pkl'
	pixel = Pixel([('Dummy', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')], pklfilepath)
	#pixel = Pixel([('Dummy', 'MSRVTT/valvideo.json', 'MSRVTT/Frames')], pklfilepath)

	pixel.create()
	pixel.save()
	pixel.load()
	#print(len(pixel.get_pixel_vectors('video0')))
	#print(pixel.get_pixel_vectors('video1')[0])
	
	print(len(pixel.get_pixel_vectors(pixel.video2vec.keys()[0])))
	print(pixel.get_pixel_vectors(pixel.video2vec.keys()[0])[0])

