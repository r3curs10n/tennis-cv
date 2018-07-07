import numpy as np
import cv2
import json

# Utility to play a video (VIDEO_NAME) and label interesting frames
# to generate data.
#
# When the video is playing, you can:
# 1. Press 'm' to pause.
# 2. Press 'o'/'p' to give label 0/1 to the current frame.
# 3. Press 'q' to quit the video.
#
# The captured frame and label will be saved in DATA_PATH.

DATA_PATH = 'data'
VIDEO_NAME = 'test_video.mp4'

def get_metadata_path():
	return DATA_PATH + '/metadata.json'

def get_frame_path(id):
	return DATA_PATH + '/test_' + str(id) + '.png'

def get_info_path(id):
	return DATA_PATH + '/test_' + str(id) + '.json'

def read_metadata():
	with open(get_metadata_path()) as data:
		return json.load(data)

def save_metadata(metadata):
	with open(get_metadata_path(), 'w') as data:
		json.dump(metadata, data)

def save_frame(id, img, label):
	cv2.imwrite(get_frame_path(id), img)
	with open(get_info_path(id), 'w') as data:
		json.dump({'is_court': label}, data)

def main():
	metadata = read_metadata()
	next_id = metadata['next_id']

	cap = cv2.VideoCapture(VIDEO_NAME)

	paused = False
	frame = None
	while (cap.isOpened()):
		if not paused:
			ret, frame = cap.read()
		cv2.imshow('frame', frame)
		key = cv2.waitKey(25)
		if key & 0xFF == ord('q'):
			break
		if key & 0xFF == ord('m'):
			paused = True
		if key & 0xFF == ord('o'):
			save_frame(next_id, frame, 0)
			next_id+=1
			paused = False
		if key & 0xFF == ord('p'):
			save_frame(next_id, frame, 1)
			next_id+=1
			paused = False

	metadata['next_id'] = next_id
	save_metadata(metadata)

main()