import os
import torch

from PIL import Image
from torchvision import transforms

def imagetotensor(imagefile):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
	image = Image.open(imagefile)
	transform = transforms.Compose([
        #transforms.Resize(256),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
	return transform(image)


if __name__=='__main__':
	imagefile = os.path.join('MSRVTT', 'Frames', 'video3', 'video3000.png')
	print(imagetotensor(imagefile))