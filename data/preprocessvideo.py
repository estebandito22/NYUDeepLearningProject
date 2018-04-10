import os
import shutil
import youtube_dl
import glob
from moviepy.editor import VideoFileClip

def downloadvideo(url, videoid, videofolderpath):
	if not os.path.exists(videofolderpath):
		os.makedirs(videofolderpath)
	ydl_opts = {'format': '(mp4)[height<=360]',
				'download_archive': 'downloads.log',
				'outtmpl': os.path.join(videofolderpath, videoid) + '.%(ext)s'}
	ydl = youtube_dl.YoutubeDL(ydl_opts)
	ydl.download([url])
	return

def clipvideo(videofolderpath, clipfolderpath, videoid, start_time, end_time):
	if not os.path.exists(clipfolderpath):
		os.makedirs(clipfolderpath)	
	clip = VideoFileClip(os.path.join(videofolderpath, videoid) + ".mp4").subclip(start_time, end_time)
	clip.write_videofile(os.path.join(clipfolderpath, videoid)+ ".mp4", audio = False)	
	return

def getframes(clipfolderpath, framefolderpath, videoid, fps):
	clip = VideoFileClip(os.path.join(clipfolderpath, videoid) + ".mp4")
	print(clip.fps)
	videoframefolderpath = os.path.join(framefolderpath, videoid)
	if not os.path.exists(videoframefolderpath):
		os.makedirs(videoframefolderpath)
	clip.write_images_sequence(os.path.join(videoframefolderpath, videoid) + "%03d.png", fps=fps)
	return

if __name__=='__main__':
	cur_dir = os.getcwd()
	datafolder = "MSRVTT"
	videofolder = "Videos"
	clipfolder = "Clips"
	framefolder = "Frames"
	folderpath = os.path.join(cur_dir, datafolder)
	videofolderpath = os.path.join(cur_dir, datafolder, videofolder)
	clipfolderpath = os.path.join(cur_dir, datafolder, clipfolder)
	framefolderpath = os.path.join(cur_dir, datafolder, framefolder)


	url = "https://www.youtube.com/watch?v=JCqfncpIjJM"
	videoid = 'sundari'
	start_time = 10.23
	end_time = 25.24

	try:
		downloadvideo(url, videoid, videofolderpath)
		clipvideo(videofolderpath, clipfolderpath, videoid, start_time, end_time)
		getframes(clipfolderpath, framefolderpath, videoid, 10)
	except:
		print("An Error Occured for video " + videoid)
