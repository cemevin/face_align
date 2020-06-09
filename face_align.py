import cv2
import sys
import numpy as np
import math
from PIL import Image
import os
import imageio

# Get user supplied values
# imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

def rotatePoint(point, radians, origin=(0, 0)):
	x, y = point
	ox, oy = origin

	qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
	qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

	return qx, qy

def rotate_image(image, angle, image_center = False):
	if image_center == False:
		image_center = tuple(np.array(image.shape[1::-1]) / 2)

	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1])#, flags=cv2.INTER_LINEAR)
	return result

def euclidean_distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectEyes(img):
	eyes = eye_detector.detectMultiScale(img,
		scaleFactor=1.1,
		minNeighbors = 5,
		maxSize = (150,150),
		minSize = (50, 50)
	)

	# get biggest two eyes in case there are false positives
	def myfn(x):
		return -1 * x[:,2] * x[:,3]
	temp = myfn(eyes)
	order = np.argsort(temp)
	eyes = eyes[order]

	# get eye 1 and 2, draw rectangles
	index = 0
	for (eye_x, eye_y, eye_w, eye_h) in eyes:
		if index == 0:
			eye_1 = (eye_x, eye_y, eye_w, eye_h)
		elif index == 1:
			eye_2 = (eye_x, eye_y, eye_w, eye_h)
		else:
			break

		index = index + 1

	# detect left and right eyes
	if eye_1[0] < eye_2[0]:
	   left_eye = eye_1
	   right_eye = eye_2
	else:
	   left_eye = eye_2
	   right_eye = eye_1

	# detect eye centers
	left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
	right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))

	return (left_eye_center, right_eye_center)

def detectFace(img, scaleFactor = 1.3):
	faces = face_detector.detectMultiScale(img, scaleFactor, 5)
	face_x, face_y, face_w, face_h = faces[0]

	maxx = (0,0,0,0)
	for (x,y,w,h) in faces:
		if w*h > maxx[3] * maxx[2]:
			maxx = (x,y,w,h)

	x,y,w,h = maxx
	face_x, face_y, face_w, face_h = maxx

	return face_x, face_y, face_w, face_h

def processImage(path):
	img = cv2.imread(path)
	img = cv2.resize(img,(720, 960))
	img_raw = img.copy()

	# detect face
	face_x, face_y, face_w, face_h = detectFace(img_raw)

	# crop
	img_eyes = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
	img_gray = cv2.cvtColor(img_eyes, cv2.COLOR_BGR2GRAY)

	# detect eye
	left_eye_center, right_eye_center = detectEyes(img_gray)
	left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
	right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]

	# draw a triangle between eyes
	if left_eye_y > right_eye_y:
	   point_3rd = (right_eye_x, left_eye_y)
	   direction = -1 #rotate ccw
	else:
	   point_3rd = (left_eye_x, right_eye_y)
	   direction = 1 #rotate anti ccw

	# find angle of rotation
	a = euclidean_distance(left_eye_center, point_3rd)
	b = euclidean_distance(right_eye_center, left_eye_center)
	c = euclidean_distance(right_eye_center, point_3rd)

	cos_a = (b*b + c*c - a*a)/(2*b*c)
	angle = np.arccos(cos_a)
	angle = (angle * 180) / math.pi

	if direction == -1:
	   angle = 90 - angle

	# global eye coordinates
	leftx = left_eye_x + face_x
	lefty = left_eye_y + face_y
	rightx = right_eye_x + face_x
	righty = right_eye_y + face_y
	eyesCenter = ((rightx+leftx)/2, (righty+lefty)/2)

	# rotate image around the eyes
	img_raw = rotate_image(img_raw, direction * angle, eyesCenter)

	# rotate eyes
	leftx, lefty = rotatePoint((leftx, lefty), direction * angle * math.pi / 180, eyesCenter)
	rightx, righty = rotatePoint((rightx, righty), direction * angle * math.pi / 180, eyesCenter)

	# center the eyes
	height = img_raw.shape[0]
	width = img_raw.shape[1]
	center = (width/2,height*7/16)
	eyesCenter = ((rightx+leftx)/2, (righty+lefty)/2)
	offset = (eyesCenter[0]-center[0], eyesCenter[1]-center[1])
	dx = offset[0]
	dy = offset[1]
	translation_matrix = np.float32([ [1,0,-dx], [0,1,-dy] ])
	num_rows, num_cols = img_raw.shape[:2]
	img_raw = cv2.warpAffine(img_raw, translation_matrix, (num_cols, num_rows))

	# resize image
	idealEyeDistance = 180
	eyeDistance = rightx - leftx
	resizeFactor = idealEyeDistance/eyeDistance
	img_raw = cv2.resize(img_raw,None, fx=resizeFactor, fy=resizeFactor)

	# crop image
	newHeight = img_raw.shape[0]
	newWidth = img_raw.shape[1]
	blank_image = np.zeros((height,width,3), np.uint8)

	if newWidth > width:
		startX = int((newWidth - width)/2)
		endX = startX + width
		startY = int((newHeight - height)/2)
		endY = startY + height

		blank_image[:,:] = img_raw[startY:endY,startX:endX]
	else:
		startX = -int((newWidth - width)/2)
		endX = startX + width
		startY = -int((newHeight - height)/2)
		endY = startY + height

		blank_image[startY:endY,startX:endX] = img_raw[:,:]

	new_img = Image.fromarray(blank_image)

	return np.array(new_img)

print('--- processing images ---')
for path in os.listdir('./images'):
	print('processing ' + path)
	newImg = processImage("images/" + path)
	cv2.imwrite("cropped/" + path, newImg)

# video
print('--- creating video ---')
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), 15,(720,960))
for i in range(1, 253):
    frame = cv2.imread('./cropped/' + str(i) + '.jpg')
    writer.write(frame)
writer.release()
