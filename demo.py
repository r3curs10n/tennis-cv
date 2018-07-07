from keras.models import load_model

import numpy as np
import cv2
import data_loader

GREEN = (0,255,0)
RED = (0,0,255)
MODEL_NAME = 'models/court_detector.h5'

# Plays a video and draws a GREEN circle on it when the court is fully
# visible and RED when it is not. Uses MODEL_NAME for classification.
def main():
	model = load_model(MODEL_NAME)
	cap = cv2.VideoCapture('test_video.mp4')

	while (cap.isOpened()):
		# Skip alternate frame because classification is slow.
		ret, frame = cap.read()
		ret, frame = cap.read()

		tr = data_loader.preprocess_for_classification(frame)
		tr = tr.reshape(1, 1, tr.shape[0], tr.shape[1])
		ans = model.predict(tr, batch_size=1)
		
		color = GREEN
		if ans[0][0] < 0.5:
			color = RED

		cv2.circle(frame, (50,50), 10, color, -1)

		cv2.imshow('frame', frame)

		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break

main()