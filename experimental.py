import numpy as np
import cv2
import pprint

pp = pprint.PrettyPrinter(indent=4)

GREEN = (0, 255, 0)

# This is experimental stuff. Doesn't work well.

def segment_by_angle_rt(lines, delta):
    lines = sorted(lines, key=lambda x: x[0][1])
    lines = [(x[0][0], x[0][1]) for x in lines]

    prev = -9999999
    segmented_lines = []
    for rho, theta in lines:
    	if theta - prev > delta:
    		segmented_lines.append([(rho, theta)])
    		prev = theta
    	else:
    		segmented_lines[-1].append((rho, theta))
    return segmented_lines

def get_theta(line):
	if (line[0][0]-line[1][0]) == 0:
		return np.pi/2
	return np.arctan(float(line[0][1]-line[1][1])/(line[0][0]-line[1][0]))

def rad2deg(x):
	return x*180/np.pi

def segment_by_angle(lines, delta):
	nl = [(x[0], x[1], get_theta(x)) for x in lines]
	nl = sorted(nl, key=lambda x: x[2])
	nl = [x for x in nl if abs(90-rad2deg(x[2])) > 10]

	prev = -99999
	sl = []
	for p1, p2, theta in nl:
		if theta - prev > delta:
			sl.append([(p1, p2)])
			prev = theta
		else:
			sl[-1].append((p1, p2))
	return sl

def get_cross_segment_intersections(segmented_lines):
	intersections = []
	for i, segment in enumerate(segmented_lines[:-1]):
		for other_segment in segmented_lines[i+1:]:
			get_intersections(segment, other_segment, intersections)
	return intersections

def get_intersections(lines1, lines2, intersections):
	for line1 in lines1:
		for line2 in lines2:
			intersections.append(get_intersection(line1, line2))

def get_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_intersection_rt(line1, line2):
	rho1, theta1 = line1
	rho2, theta2 = line2
	A = np.array([
		[np.cos(theta1), np.sin(theta1)],
		[np.cos(theta2), np.sin(theta2)]
	])
	b = np.array([[rho1], [rho2]])
	x0, y0 = np.linalg.solve(A, b)
	x0, y0 = int(np.round(x0)), int(np.round(y0))
	return (x0, y0)

def get_condensed_points(points, delta):
	clusters = []
	for pt in points:
		found = False
		for i, c in enumerate(clusters):
			if distance(c[0], pt) < delta:
				clusters[i].append(pt)
				found = True
				break
		if not found:
			clusters.append([pt])
	return [avg_point(c) for c in clusters]

def distance(p1, p2):
	return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2

def avg_point(cluster):
	x = sum([p[0] for p in cluster])/len(cluster)
	y = sum([p[1] for p in cluster])/len(cluster)
	return (int(x), int(y))

def histogram(img, box):
	top_left, bottom_right = box
	roi = box[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def draw_lines(img, lines):
	for line in lines:
		cv2.line(img, line[0], line[1], (255, 0, 0), 2)

def get_polar_line(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	g = abs(x2*y1-y2*x1)
	h =  cv2.norm((float(x2 - x1), float(y2 - y1)))
	rho = g / h
	theta = -np.arctan2(x2-x1, y2-y1)
	return rho, theta

def find_quad(roi, tag):
	height, width = roi.shape
	lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=75, minLineLength=width/3, maxLineGap=width/6)
	roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
	draw_lines(roi, lines)

	lines = [((x[0][0], x[0][1]), (x[0][2], x[0][3])) for x in lines]
	sl = segment_by_angle(lines, 0.17)
	ins = get_cross_segment_intersections(sl)
	ins = get_condensed_points(ins, 400)

	for i in ins:
		cv2.circle(roi, i, 5, (255,255,0), -1)
	cv2.imshow(tag, roi)

	return ins

def get_key_points(img):
	height = img.shape[0]
	width = img.shape[1]
	img = img[height/2:, :]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.Canny(gray,50,150,apertureSize = 3)

	roi_width = int(width/2.5)
	roi_height = int(height/2)
	left_roi = gray[:, :roi_width]
	right_roi = gray[:, -roi_width:width]

	lp = find_quad(left_roi, 'left')
	rp = find_quad(right_roi, 'right')

	def lp_transform(pt):
		x, y = pt
		return (x, y+roi_height)

	def rp_transform(pt):
		x, y = pt
		return (x + width - roi_width, y+roi_height)

	return map(lp_transform, lp) + map(rp_transform, rp)

def in_range(x, low, high):
	return x >= low and x <= high

def filter_lines(lines, width, height, keepAll=False):
	fl = []
	for line in lines:
		x1, y1, x2, y2 = line[0]
		line_f = ((x1,y1),(x2,y2))
		theta = rad2deg(get_theta(line_f))
		if keepAll:
			fl.append(line_f)
			continue
		if abs(theta) < 10:
			xl = min(x1, x2)
			xh = max(x1, x2)
			if xl < width/4 and xh > 3*width/4 or y1 < height/2:
				fl.append(line_f)
		elif in_range(abs(theta), 50, 80):
			yl = min(y1, y2)
			yh = max(y1, y2)
			if yl < 10:
				fl.append(line_f)
	return fl


def get_key_points_2(img, verbose=False):
	height = img.shape[0]
	width = img.shape[1]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.blur(gray, (3,3))
	gray = cv2.Canny(gray,50,150,apertureSize = 3)

	lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=75, minLineLength=height/2, maxLineGap=height/4)
	lines = filter_lines(lines, width, height)
	
	draw_lines(img, lines)

	sl = segment_by_angle(lines, 0.17)
	ins = get_cross_segment_intersections(sl)
	ins = get_condensed_points(ins, 400)

	if verbose:
		for i in ins:
			cv2.circle(img, i, 5, (0,255,0), -1)
		cv2.imshow('img', img)
	return ins

def get_key_points_3(imgo, verbose=True):
	height = imgo.shape[0]
	img = imgo[height/3:, :, :]
	old_height = height
	height, width, _ = img.shape
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	min_color = np.array([0,50,190])
	max_color = np.array([360,255,255])
	mask = cv2.inRange(hsv, min_color, max_color)

	kernel = np.ones((3,3), np.uint8)
	mask = cv2.dilate(mask, kernel, iterations=1)

	lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
	lines = filter_lines(lines, width, height, keepAll=False)
	
	#if verbose:
	#	draw_lines(img, lines)

	#sl = segment_by_angle(lines, 0.17)
	#ins = get_cross_segment_intersections(sl)
	#ins = get_condensed_points(ins, 400)

	if verbose:
		#for i in ins:
		#	cv2.circle(img, i, 5, (0,255,0), -1)
		cv2.imshow('img', img)
		cv2.imshow('mask', mask)
	#return [(x, y+old_height/3) for (x,y) in ins]

def main():
	img = cv2.imread('test_img2.png')
	
	get_key_points_3(img, verbose=True)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

	segmented_lines = segment_by_angle(lines, 0.7)
	intersections = get_cross_segment_intersections(segmented_lines)
	intersections = get_condensed_points(intersections, 200)

	for pt in intersections:
		cv2.circle(img, pt, 5, GREEN, -1)

	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()