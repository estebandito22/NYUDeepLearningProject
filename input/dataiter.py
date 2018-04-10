import os
import json


def iterdatashallow(captionsjson, videosjson):
	videoids = videosjson["data"]
	captions = captionsjson["data"]
	for video in captions:
		videoid = video["videoid"]
		captionslist = video["captions"]
		if videoid in videoids:
			yield videoid, captionslist

def iterdatadeep(captionsjson, videosjson):
	videoids = videosjson["data"]
	captions = captionsjson["data"]
	for video in captions:
		videoid = video["videoid"]
		captionslist = video["captions"]
		if videoid in videoids:
			for caption in captionslist:
				yield videoid, caption

def iterimages(framesfolder, imagefolder):
	framespath = os.path.join(framesfolder, imagefolder)
	for filename in os.listdir(framespath):
		if filename.endswith(".png"):
			filepath = os.path.join(framesfolder, imagefolder, filename) 
			yield filepath


if __name__=='__main__':
	
	'''captionsfilepath = 'MSRVTT/captions.json'
	trainvideoidspath = 'MSRVTT/trainvideo.json'
	captions = json.load(open(captionsfilepath, 'r'))
	trainvideoids = json.load(open(trainvideoidspath, 'r')) 
	for videoid, caption_tokens in iterdatadeep(captions, trainvideoids):
		print(caption_tokens)'''

	framesfolder = 'MSRVTT/Frames'
	imagefolder = 'video0'
	for imagefile in iterimages(framesfolder, imagefolder):
		print (imagefile)
