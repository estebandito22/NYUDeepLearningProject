import os
import json

def showsplits(jsonfilepath):
	jsonfile = open(jsonfilepath, 'r')
	data = json.load(jsonfile)
	splits = []

	for video in data["videos"]:
		split = video["split"]	
		if split not in splits:
			splits.append(split)

	print (splits)

if __name__=='__main__':
	cur_dir = os.getcwd()
	datafolder = "MSRVTT"
	datafolderpath = os.path.join(cur_dir, datafolder)
	jsonfilename = "train_val_videodatainfo.json"	
	jsonfilepath = os.path.join(datafolderpath, jsonfilename)
	showsplits(jsonfilepath)