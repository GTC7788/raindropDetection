#####################################################################

# Example : Detect raindrops within an image by using sliding window region
# proposal algorithm and classify the ROI by using AlexNet CNN.

# Copyright (c) 2017/18 - Tiancheng Guo / Toby Breckon, Durham University, UK

# License : https://github.com/GTC7788/raindropDetection/LICENSE

#####################################################################

# The green rectangles represents the detected raindrops.
# The red rectangles represents the ground truth raindrops in the image.


# This script takes 1 argument indicating the image to process.
# e.g. 
# > python raindrop_detection_sliding_window.py 3 
# will process image 3 in the in the raindrop_detection_images folder and use 
# the associated ground truth xml file in ground_truth_labels folder for 
# image 3 as well.  

# This program will output 1 detection result image in the same folder.

#####################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
import cv2
import os
import time
import xml.etree.ElementTree as ET
from math import sqrt
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_image_dataset_from_dir
import argparse


#####################################################################
# Use a command line parser to read command line argument
# The integer number represents the number of the image to process
parser = argparse.ArgumentParser()
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer represents the number of image')
args = parser.parse_args()

number = args.integers[0]

# Manually set the number of image to process.
# number = 1


#######################################################################

# Image path
image_path = "raindrop_detection_images/%s.jpg" % number

# Path to output the result image after raindrop detection 
result_path = "img_%s_sliding_window_result.jpg" % number

# Path to the xml file that contains the ground truth data
ground_truth_xml_path = "ground_truth_labels/%s.xml" % number

# Path of the trained model for AlexNet
model_path = 'Model/alexRainApr06.tfl'

# Turn on ground truth detections on image 
ground_truth = True


#######################################################################

"""
Convert the PIL Image object into array.
Args:
	pil_image: PIL image object
Returns:
	result: an array ready for the CNN to predict
"""
def img_to_array(pil_image):

    pil_image.load()
    result = np.asarray(pil_image, dtype="float32")
    result /= 255
    return result


"""
Set up the structure of AlexNet CNN by using TFLearn.
Returns:
	network: a CNN which follows the structure of AlexNet.
"""
def create_basic_alexnet():

	# Building network as per architecture in [Guo/Breckon, 2018]

	network = input_data(shape=[None, 30, 30, 3])
	network = conv_2d(network, 64, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 128, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='momentum',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)
	return network




"""
Calculates all the windows that will slide through an image.

Args:
	image: the image to apply sliding window.
	stepSize: step size (in pixel) between each window.
	windowSize: size of each window.
Return:
	All of the sliding windows for an image, each element represents 
	the coordinates of top left corner of the window and its size. 
"""
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


"""
Sliding window algorithm will generates too many rectangles.
We can use the groupRectangles method to reduce overlapping rectangles.
Args:
	rectangleList_before: list of detected regions (rectangles).
	threshold: Minimum possible number of rectangles minus 1. 
			   The threshold is used in a group of rectangles to retain it. 
	eps: Relative difference between sides of the rectangles to merge them into a group.
Return:
	rectangleList_after: list of optimized detected regions.
"""
# Regularise the format of the proposed result list. 
def utilize_rectangle_list(rectangleList_before, threshold, eps):
	# Using the groupRectangles() function to shrink the rectangle list
	rectangleList_after = []

	for element in rectangleList_before:
		full_rectangle_list = []
		full_rectangle_list.append(element[0])
		full_rectangle_list.append(element[1])
		full_rectangle_list.append(element[0]+30)
		full_rectangle_list.append(element[1]+30)
		rectangleList_after.append(full_rectangle_list)

	# group the proposed overlapping regions into one region, 
	# decrese the recall but increase the precision. 
	rectangleList_after, weight = cv2.groupRectangles(rectangleList_after, threshold, eps)

	return rectangleList_after


"""
Parse the xml file that stores the ground truth raindrop locations in the image

Args:
	fileName: the xml file name
Returns: 
	list that each element contains the location of a ground truth raindrop

"""
def parse_xml_file(fileName):
	xml_file = ET.parse(fileName)
	# XML_path to retrieve the x, y coordinates 
	xIndex = xml_file.findall('object/polygon/pt/x')
	yIndex = xml_file.findall('object/polygon/pt/y')
	xList = []
	yList = []
	for x in xIndex:
		xList.append(int(x.text))

	for y in yIndex:
		yList.append(int(y.text))

	combinedList = zip(xList,yList)

	subList = []
	finalList = []
	counter = 1
	for element in combinedList:
		switch = counter % 4
		if switch == 0:
			subList.append(element)
			finalList.append(subList)
			subList = []
		else:
			subList.append(element)
		counter += 1

	return finalList

"""
Retrieve the coordinates of each ground truth raindrop locations
Args:
	xml_golden: a list that each element contains the location of a ground truth raindrop
Returns:
	a list of coordinates for each ground truth raindrops that ready for drawing. 	
"""
def xml_transform(xml_golden):
	xml_result = []
	for element in xml_golden:
		sub_list = []
		sub_list = [element[0][0], element[0][1], 
		element[2][0], element[2][1]]

		xml_result.append(sub_list)
	return xml_result



"""
Slide the window across the image, pass each window (region of interest) into the trained AlexNet. 
If the region is classified as a raindrop, store the region's coordinates in a list and return 
the list.

Args:
	image: the image to process
	winW: width of the sliding window
	winH: height of the sliding window
Return:
	rectangle_result: a list of region of interest that classified as raindrop by the AlexNet
"""
def cnn_find_raindrop(image, winW, winH):
	rectangle_result = []

	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet the desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		roi = image[y:y + winH, x:x + winW]
		
		# Convert array into PIL Image.
		im = Image.fromarray(roi)
		tensor_image = img_to_array(im)
		imgs = [] # must be in the 2d list format, no additional usage.
		imgs.append(tensor_image)

		# predict the region.
		predict_result = model.predict(imgs)
		final_result = np.argmax(predict_result[0]) # transfer the result to 0 or 1
		
		if final_result == 1:
			rectangle_result.append((x, y))
	return rectangle_result



# Initialise the AlexNet and load the trained model for the CNN.
alex_net = create_basic_alexnet()
model = tflearn.DNN(alex_net)
model.load(model_path, weights_only = True)


# Set the height and width of the sliding window
(winW, winH) = (30, 30)

# Read the image 
image = cv2.imread(image_path)

# Get the proposed regions
rectangle_result = cnn_find_raindrop(image, winW, winH)

# Remove overlapping rectangles
new_rectangle_list = utilize_rectangle_list(rectangle_result, 1, 0.1)



# # **************** Draw Optimized Rectangles *******************
# We don't want to draw the detection rectangles directly on the original image,
# we copy the image and draws the rectangels on the copied image. 
clone = image.copy() 

for element in new_rectangle_list:
	cv2.rectangle(clone,(element[0], element[1]),(element[2],element[3] ),(0, 255, 0),2)
## *************************************************************



# ********** Draw the rectangles that contains ground truth raindrops ********
if ground_truth:
	# Parse the xml file that contains raindrop locations of the image
	xml_golden = parse_xml_file(ground_truth_xml_path)
	# Read the coordinates of the raindrops
	xml_reformat = xml_transform(xml_golden)
	# *************** Draw the XML Result ********************
	for element in xml_reformat:
		cv2.rectangle(clone,(element[0], element[1]),(element[2],element[3] ),(0, 0, 255),2)
	# ********************************************************


# Save the result image into a folder.
cv2.imwrite(result_path, clone)
