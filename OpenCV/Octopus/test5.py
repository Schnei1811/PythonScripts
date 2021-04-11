from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2
import numpy as np
from random import shuffle
import os
import matplotlib.pyplot as plt
import tqdm
from sklearn import metrics

def label_img(dir):
    if dir == 'Octopus': return [1, 0, 0, 0, 0, 0]
    elif dir == 'Kelp': return [0, 1, 0, 0, 0, 0]
    elif dir == 'Fish': return [0, 0, 1, 0, 0, 0]
    elif dir == 'Environment': return [0, 0, 0, 1, 0, 0]
    elif dir == 'Diver': return [0, 0, 0, 0, 1, 0]
    elif dir == 'MultiClass': return [0, 0, 0, 0, 0, 1]


IMG_SIZE = 150
LR = 0.0001
NUM_CLASSIFICATIONS = 6
NumEpochs = 5
MODEL_NAME = 'Models/6ClassVGG16.model'


# Building 'VGG Network'
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, NUM_CLASSIFICATIONS, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='targets')
model = tflearn.DNN(network, tensorboard_dir='log')



orig = cv2.imread('octotest3.jpg', cv2.IMREAD_GRAYSCALE)
orig = cv2.resize(orig, (150,150))
data = orig.reshape(IMG_SIZE, IMG_SIZE, 1)
model_out = model.predict([data])[0]

if np.argmax(model_out) == 0: str_label = 'Octopus'
elif np.argmax(model_out) == 1: str_label = 'Kelp'
elif np.argmax(model_out) == 2: str_label = 'Fish'
elif np.argmax(model_out) == 3: str_label = 'Environment'
elif np.argmax(model_out) == 4: str_label = 'Diver'
elif np.argmax(model_out) == 5: str_label = 'Multiclass'

