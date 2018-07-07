import numpy as np
import cv2

GREEN = (0, 255, 0)

# This is experimental stuff. Doesn't work well.

def segment_by_angle(lines, delta):
    lines = sorted(lines, key=lambda x: x[0][1])
    lines = [(x[0][0], x[0][1]) for x in lines]

    prev = -2*delta
    segmented_lines = []
    for rho, theta in lines:
    	if theta - prev > delta:
    		segmented_lines.append([(rho, theta)])
    		prev = theta
    	else:
    		segmented_lines[-1].append((rho, theta))
    return segmented_lines

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


def main():
	img = cv2.imread('test_4.png')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
	cv2.imshow('img', gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main1():
	img = cv2.imread('test_2.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,11,2)

	# blurred = cv2.medianBlur(gray, 1)
	# edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
	cv2.imshow('img', th3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

	lines = cv2.HoughLines(edges, 50, np.pi/18, 200)

	for line in lines:
		for rho,theta in line:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))

	    	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

	cv2.imshow('ff', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

	segmented_lines = segment_by_angle(lines, 0.17)
	intersections = get_cross_segment_intersections(segmented_lines)
	intersections = get_condensed_points(intersections, 200)

	for pt in intersections:
		cv2.circle(img, pt, 5, GREEN, -1)

	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()