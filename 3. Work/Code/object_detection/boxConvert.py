# coding: utf-8

# # Object Detection and Tracking, Utilities
# 
# Authors: Kevin DuCharme and Max Li
#
# # Description: Utility functions that convert between the normalized bounding boxes that TF uses and 
# 	the pixel-based bounding boxes that opencv uses.


import numpy as np
def convertToTracking(input, size):
	size_mult = size[0:2] + size[0:2]
	resized = np.multiply(input, size_mult)
	ret = tuple(abs(np.round(resized)))
	return ret

def convertToDetecting(input,size):
	size_mult = size[0:2] + size[0:2]
	resized = np.divide(input,size_mult)
	ret = list(resized)
	return ret
