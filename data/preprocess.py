import os
import logging
import json
import glob

from preprocessvideo import *
from preprocesstext import *
from preprocessutils import *


def processvideomsrvtt(jsonfilepath, videofolderpath, framefolderpath, clipfolderpath, challenge):
	trainvideo = []
	valvideo = []
	testvideo = []
	downloadedvideos = []

	jsonfile = open(jsonfilepath, 'r')
	data = json.load(jsonfile )
	count = 0
	
	for file in glob.glob(os.path.join(clipfolderpath, "*.mp4")):
		videoid = file.split('/')[-1]
		videoid = videoid[:-4]
		downloadedvideos.append(videoid)

	for video in data["videos"]:
		url = video["url"]
		videoid = video["video_id"]
		start_time = video["start time"]
		end_time = video["end time"]
		split = getsplit(video["split"], challenge)
	
		if videoid not in downloadedvideos:
			try:
				downloadvideo(url, videoid, videofolderpath)
				logging.info("Downloaded Video %s" % (videoid))
				clipvideo(videofolderpath, clipfolderpath, videoid, start_time, end_time)
				logging.info("Extracted Clips %s" % (videoid))
			except:
				count = count + 1
				logging.warn("Error Occured %s Train Count = %d" % (videoid, traincount))
		getframes(clipfolderpath, framefolderpath, videoid, 3)
		logging.info("Processed Frames %s" % (videoid))
		if split == 'train':
			trainvideo.append(videoid)
		elif split == 'validate':
			valvideo.append(videoid)
		elif split == 'test':
			testvideo.append(videoid)

	return trainvideo, valvideo, testvideo


def processtextmsrvtt(jsonfilepath):
	sentences = []

	jsonfile = open(jsonfilepath, 'r')
	data = json.load(jsonfile )
	currentvideo = ''
	videoid = ''
	captionlist = []

	for captions in data["sentences"]:
		caption = captions['caption']
		videoid = captions['video_id']
		if videoid != currentvideo:
			if currentvideo != '':
				sentences.append({'videoid' : currentvideo, 'captions' : captionlist})
				captionlist = []
			currentvideo = videoid
		captionlist.append(tokenize(caption))
	sentences.append({'videoid' : currentvideo, 'captions' : captionlist})
	return sentences

if __name__=='__main__':
	logging.basicConfig(filename="dataprep.log", level=logging.INFO)

	cur_dir = os.getcwd()
	outputfolder = "../input"
	datafolder = "MSRVTT"
	videofolder = "Videos"
	clipfolder = "Clips"
	framefolder = "Frames"
	datafolderpath = os.path.join(cur_dir, datafolder)
	videofolderpath = os.path.join(cur_dir, datafolder, videofolder)
	clipfolderpath = os.path.join(cur_dir, datafolder, clipfolder)
	framefolderpath = os.path.join(cur_dir, outputfolder, datafolder, framefolder)

	#jsonfilename = "train_val_videodatainfo.json"
	jsonfilename = "examp.json"
	challenge = "msrvtt_2016"	
	jsonfilepath = os.path.join(datafolderpath, jsonfilename)

	'''trainvideo, valvideo, testvideo = \
		processvideomsrvtt(jsonfilepath, videofolderpath, framefolderpath, clipfolderpath, challenge)

	trainjson = os.path.join(cur_dir, outputfolder, datafolder, 'trainvideo.json')
	valjson = os.path.join(cur_dir, outputfolder, datafolder, 'valvideo.json')
	testjson = os.path.join(cur_dir, outputfolder, datafolder, 'testvideo.json')

	trainfile = open(trainjson, 'w')
	valfile = open(valjson, 'w')
	testfile = open(testjson, 'w')

	trainfile.write(json.dumps({'data':trainvideo})) 
	valfile.write(json.dumps({'data':valvideo})) 
	testfile.write(json.dumps({'data':testvideo}))'''

	captionjson = os.path.join(cur_dir, outputfolder, datafolder, 'captions.json')
	captionfile = open(captionjson, 'w')
	sentences = processtextmsrvtt(jsonfilepath)
	captionfile.write(json.dumps({'data':sentences}))

