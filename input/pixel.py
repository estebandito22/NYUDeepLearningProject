import os
import time
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
			x = module(x).squeeze().contiguous()
                        if name in self.intermediate_layers:
                                intermediate_features += [x]
                return intermediate_features

#Class name to be changed 
class Pixel():

	def __init__(self, files):
		self.files = files
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
				#imagefeatures = [self.pretrained(Variable(frame.cuda()))[1].view(-1).data.cpu() for frame in imageframes]
				imagefeatures = self.pretrained(Variable(torch.stack(imageframes).cuda()))[1].data.cpu()
				print(imagefeatures.shape)
				imagefeatures = list(imagefeatures)
				self.video2vec[videoid] = imagefeatures
				feat_time = time.time()
				print(videoid, len(imagefeatures), imagefeatures[0].shape)
				print('File : {0}, Frame : {1}, Feat : {2}'.format(file_time-start_time, frame_time-file_time, feat_time-frame_time))
		return

	def get_pixel_vectors(self, videoid):
		#index error if video not found
		return self.video2vec[videoid]


if __name__ == '__main__':

	pixel = Pixel([('Dummy', 'MSRVTT/trainvideo.json', 'MSRVTT/Frames')])

	pixel.create()



