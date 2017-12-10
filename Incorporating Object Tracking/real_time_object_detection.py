# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
'''
NOTE: NEEDS https://pypi.python.org/pypi/opencv-contrib-python/3.3.0.10#downloads
		and https://pypi.python.org/pypi/opencv-python/3.3.0.10
		That is, it needs an opencv with compiled trackers.
TODO (MAX):

'''


# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys


# Object class.
class DetectedObject:

	def __init__(self, startX, startY, endX, endY, labelIDX):
		global objIDCnt
		global CLASSES
		self.start = [startX,startY]
		self.end = [endX,endY]
		self.labelIDX = labelIDX
		self.id = objIDCnt
		print(self.id)
		objIDCnt = objIDCnt + 1

	# Returns false if labels aren't the same. 
	#(Note, not comparing two of the same object, just with the detection) 
	def compare(self, startX,startY,endX,endY,label, margin):
		#print(label)
		if label != CLASSES[self.labelIDX]:
			return False
		else:
			# Compress these if trying to be codespace efficient.
			delta_x_top = self.start[0] - startX
			delta_y_top = self.start[1] - startY
			delta_x_bottom = self.end[0] - endX
			delta_y_bottom = self.end[1] - endY 
			#print(delta_x_top)
			#max_off = max(delta_x_top,delta_y_top,delta_x_bottom, delta_y_bottom)
			max_off = max(abs(delta_x_top),abs(delta_y_top),abs(delta_x_bottom),abs(delta_y_bottom))
			if max_off > margin:
				return False
			else:
				return True



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-f", "--trackframes", default=10,
	help="Number of frames to track between detection events")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# MAX: Set frame count variable.
cntFrame = 0
objBuffer = [] # Buffer to store identified
margin = 20
objIDCnt = 0
trackerHasInit = 0

# Options are Boosting, MIL, KCF, TLD, MedianFlow, and GOTURN.
tracker = cv2.TrackerKCF_create()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Run detection algorithm.	
	if cntFrame == 0:
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				
				### MAX: INCORPORATE OBJECT IDENTIFICATION HERE (EG MAKE SURE THAT A DETECTED OBJECT KEEPS IT'S ID)
				
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				label_in = "{}".format(CLASSES[idx])

				# Check if object that is detected already exists in detection buffer.
				### NOTE CURRENT METHOD DOESNT ACCOUNT FOR DETECTOR CHANGING ITS MIND
				###  INCORPORATE REMOVAL OF OBJECTS FROM objBuffer
				doesExist = False
				for elt in objBuffer:
					if elt.compare(startX,startY,endX,endY,label_in,margin):
						elt.start = [startX, startY]
						elt.end = [endX, endY]
						doesExist = True
						tracker.clear()
						tracker = cv2.TrackerKCF_create()
						tracker.init(frame, (elt.start[0],elt.start[1],elt.end[0],elt.end[1]))

				if not doesExist:
					objBuffer.append(DetectedObject(startX,startY, endX, endY, idx))
					if objIDCnt == 1: # That is, if the just added object is 0
						if trackerHasInit == 0:
							tracker.init(frame,(startX,startY,endX,endY))
							trackerHasInit = 1

				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	else: ### MAX: Still to implement, tracking frames goes here.
		### Don't forget to call fps.update(), redraw, etc
		for elt in objBuffer:
			### NOTE: ONLY TRACKING FIRST BOX, NEED TO FIGURE OUT HOW TO ADD TRACKERS
			if elt.id == 0: 
				ok,box = tracker.update(frame)
				startX = int(box[0])
				startY = int(box[1])
				endX = int(box[2])
				endY = int(box[3])
				elt.start=[startX,startY]
				elt.end = [endX, endY]
				### Draw box, make sure color is kept.
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[elt.labelIDX], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, CLASSES[elt.labelIDX], (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[elt.labelIDX], 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	cntFrame = cntFrame + 1
	if cntFrame >= int(args["trackframes"]):
		cntFrame = 0


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
raise SystemExit