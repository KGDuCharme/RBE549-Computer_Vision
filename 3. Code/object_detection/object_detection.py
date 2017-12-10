
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import cv2
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

# ## Define Object class
# Object class.
class DetectedObject:
	def __init__(self, startX, startY, endX, endY, labelIDX, tracker_type):
		global objIDCnt
		global category_index
		self.start = [startX,startY]
		self.end = [endX,endY]
		self.labelIDX = labelIDX
		self.id = objIDCnt
		print(self.id)
		objIDCnt = objIDCnt + 1

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
	def compare(self, startX,startY,endX,endY,label, margin):
		#print(label)
		if label != category_index[self.labelIDX]:
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

# ## Tracker config
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2] # Usinc KCF
objIDCnt = 0 
 	


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

FRAMES_TRACK = 1
frame_mod_count = 0

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
# DEBUG
print(category_index)
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
      ret, image_np = cap.read()
    
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      ### MAX: THIS IS WHERE TRACKING VS DETECTION WOULD ACTUALLY GO.

      if frame_mod_count == 0:
	      # Actual detection.
	      (boxes, scores, classes, num) = sess.run(
	          [detection_boxes, detection_scores, detection_classes, num_detections],
	          feed_dict={image_tensor: image_np_expanded})

	      ### WORKFLOW
	      '''
	      	if scores > threshold:
	      		compare box to existing list of boxes
	      		if existing list of boxes has box, update box.
	      		Else, add box
	  	  	After search:
				If box in list wasn't picked, decrease some internal counter
	  	  		if counter reaches zero, remove from list.
			For each box in list:
				if tracker exists:
					close previous tracker.
				else:
					clear tracker.
				Initialize tracker.
	      '''

      else: # Use tracking instead.

      	''' 
      	TRACKING GOES HERE
      	'''

  	## Visualization
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      ### END DETECTION STUFF.
      cv2.imshow('object detection',image_np) 
      frame_mod_count = (frame_mod_count+1)% FRAMES_TRACK

      if cv2.waitKey(1) & 0xFF == ord('q'):
           break


cap.release()
cv2.destroyAllWindows()



