#####################################################################

# Classify raindrop by using a trained CNN with structure 
# of classic AlexNet.

# How to run:
 
# In command line: "python raindrop_classification.py number_of_image"  

# e.g. 
# > python raindrop_classification.py 3 
# will process image 3 in the raindrop_classification_images folder.  

# This program will output the result in the commandline.

# the result is a list of two numbers, the first number 
# indicates non-raindrop, the seconde number indicates raindrop.
# e.g. [ not raindrop,  raindrop ]
#####################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
import cv2
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_image_dataset_from_dir
import os
from tflearn.layers.merge_ops import merge
import argparse


########################################################################
# Use a command line parser to read command line argument
# The integer number represents the number of the image to process
parser = argparse.ArgumentParser()
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer represents the number of image')
args = parser.parse_args()


number = args.integers[0]
# number = 6

img_name = 'raindrop_classification_images/%s.jpg' %number
########################################################################



def load_img(img_path):
	img = Image.open(img_path)
	return img

"""
resize the loaded image into uniform size.
"""
def resize_img(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

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
	# Building 'AlexNet'
	network = input_data(shape=[None, 30, 30, 3])
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='momentum', 
		loss='categorical_crossentropy', learning_rate=0.001)
		
	return network


original= cv2.imread(img_name)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 30,30)
cv2.imshow('image',original)


original_img = load_img(img_name)
resize_img = resize_img(original_img, 30, 30)
tensor_image = img_to_array(resize_img)
imgs = []
imgs.append(tensor_image)


# *************************************************
# Set up the trained AlexNet
alex_net = create_basic_alexnet()
model = tflearn.DNN(alex_net)
model.load('Model/alexRainApr12.tfl', weights_only = True)


# *************************************************
# pass the image into AlexNet
predict_result = model.predict(imgs)
final_result = np.argmax(predict_result[0]) # return the index of the max number in a list


classes = {1 : 'Raindrop', 0  : 'Not Raindrop'}

print("For image %s.jpg" %number)
print("Numerical Result Data is: ")
print(predict_result)

print("AlexNet predict this picture is " + str(classes[int(final_result)]))


cv2.waitKey(0)





