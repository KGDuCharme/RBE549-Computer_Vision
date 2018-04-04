
# coding: utf-8

# # Object Detection and Tracking
# 
# Authors: Kevin DuCharme and Max Li
#
# # Description: Main body of RBE549 Final Project


import numpy as np
import cv2
import os
import time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

import boxConvert

# ## Define Object class
# Object class.
class DetectedObject:
	def __init__(self, box, label,score, tracker_type):
		global objIDCnt
		global category_index
		self.box = box
		self.label = label
		self.score = score
		self.visited = False
		self.id = objIDCnt
		print(self.id)
		objIDCnt = objIDCnt + 1
		self.createTracker(tracker_type)


	def createTracker(self, tracker_type):
		if tracker_type == 'BOOSTING':
			self.tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			self.tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			self.tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			self.tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			self.tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			self.tracker = cv2.TrackerGOTURN_create()

	# Returns false if labels aren't the same. 
	#(Note, not comparing two of the same object, just with the detection) 
	def compare(self, box_cmp,label, margin):
		#print(label)
		if label != self.label:
			return False
		else:
			delta_x_top = self.box[0] - box_cmp[0]
			delta_y_top = self.box[1] - box_cmp[1]
			delta_x_bottom = self.box[2] - box_cmp[2]
			delta_y_bottom = self.box[3] - box_cmp[3]
			
			max_off = max(abs(delta_x_top),abs(delta_y_top),abs(delta_x_bottom),abs(delta_y_bottom))
			if max_off > margin:
				return False
			else:
				return True






cap = cv2.VideoCapture(0)
if tf.__version__ != '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


# ## Env setup


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'tool_graph'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 3


# ## Tracker config
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[4] # KCF takes some absurd amount of time it seems.
								# Currently using Medianflow

								
objIDCnt = 0 
FRAMES_TRACK = 30
SIZE_FPS_BUF = 100 # size of moving average FPS buffer.
frame_mod_count = 0 # Keeps track of when detection should occur.
val_threshold = 0.5 # Threshold of detection
track_margin = 0.1 # Margin the box is allowed to be off before counting as a different one.The whole screen is normalized 0 to 1

objBuffer = []
avgFPSBuf = np.zeros(SIZE_FPS_BUF)
ind_FPSBuf = 0 #index for circular FPS buffer.

# ## Graph import 
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# ## Helper code

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


# # Detection

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		while(True):
			timer = cv2.getTickCount()
			ret, image_np = cap.read()

			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)

			if frame_mod_count == 0:
				# Actual detection.
				(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

				scores_past_thresh = scores > val_threshold
				boxes_valid = boxes[scores_past_thresh]
				classes_valid = classes[scores_past_thresh]
				scores_valid = scores[scores_past_thresh]

				for ind_new,elt in enumerate(boxes_valid):
					# compare box to existing list of boxes
					# if existing list of boxes has box, update box.
					doesExist = False

					for ind_obj,x in enumerate(objBuffer):
						if doesExist == True:
							break
						else:
						# Compare.
							if x.compare(elt,classes_valid[ind_new],track_margin):
								ind_save = ind_obj
								x.visited = True
								doesExist = True

					if doesExist == False:
						objBuffer.append(DetectedObject(elt,classes_valid[ind_new],scores_valid[ind_new],tracker_type))
						elt_in = boxConvert.convertToTracking(elt,image_np.shape)
						objBuffer[-1].tracker.init(image_np,elt_in)
						objBuffer[-1].visited = True
					# Add box and class to list.
					# Initialize tracker.
					
					else:
						objBuffer[ind_save].tracker.clear()
						objBuffer[ind_save].box = elt
						objBuffer[ind_save].score = scores_valid[ind_new]
						elt_in = boxConvert.convertToTracking(elt,image_np.shape)
						objBuffer[ind_save].createTracker(tracker_type)
						objBuffer[ind_save].tracker.init(image_np,elt_in)

				for ind,elt in enumerate(objBuffer):
					if elt.visited == False:
						print('NOT VISITED')
						elt.tracker.clear()
						del objBuffer[ind]
					else:
						elt.visited = False # Reset.

			else: # Use tracking instead.
			
				for elt in objBuffer:
					det,box_normal = elt.tracker.update(image_np)
					elt.box = boxConvert.convertToDetecting(box_normal,image_np.shape)
			
			# Pre-initialize buffers that would contain all the objects.
			size = len(objBuffer)
			boxes_vis = np.zeros([max(30,size),4])
			classes_vis = np.zeros([max(30,size),1])
			scores_vis = np.zeros([max(30,size),1])
			ids_vis = np.zeros([max(30,size),1])

			# Concatenate elements in a way that the visualization tool works.
			for ind,elt in enumerate(objBuffer):
				boxes_vis[ind] = elt.box
				classes_vis[ind] = elt.label
				scores_vis[ind] = elt.score
				ids_vis[ind] = elt.id

			# Keep track of FPS and average FPS
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
			if fps < 500: #Above this just means that it passed through the tracking frames quickly.
				avgFPSBuf[ind_FPSBuf] = fps
				fps_avg = np.mean(avgFPSBuf)
				ind_FPSBuf = (ind_FPSBuf + 1) % SIZE_FPS_BUF	
				#print(fps_avg)

			# If there are any items in object buffer, print them.	
			if np.any(boxes_vis != 0):
				vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes_vis),
				np.squeeze(classes_vis).astype(np.int32),
				np.squeeze(scores_vis),
				np.squeeze(ids_vis),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)
			else:	
				vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				np.squeeze(ids_vis),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)
			
			cv2.imshow('object detection',image_np) 
			frame_mod_count = (frame_mod_count+1)% FRAMES_TRACK

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break






