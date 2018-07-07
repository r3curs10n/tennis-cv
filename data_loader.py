import numpy as np
import cv2
import os
import json

def get_info_for_frame(frame_filename):
	info_filename = frame_filename.replace('.png', '.json')
	with open(info_filename) as data:
		return json.load(data)

def preprocess_for_classification(frame):
	# print frame
	resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	normalized = gray.astype('float32')
	normalized /= 255
	return normalized

def get_data_for_classification(dataset):
	files = [dataset + '/' + x for x in os.listdir(dataset)]
	png_files = [x for x in files if x[-4:] == '.png']
	xs = []
	ys = []
	for f in png_files:
		x = preprocess_for_classification(cv2.imread(f))
		y = get_info_for_frame(f)['is_court']
		xs.append(x)
		ys.append(y)
	return (np.array(xs), np.array(ys))

def get_train_and_test_data():
	return (get_data_for_classification('data'), get_data_for_classification('test_data'))