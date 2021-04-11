# -*- coding: utf-8 -*-

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 16-layers convolutional network to Oxford's 17 Category Flower
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

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

# cv2.imshow('image', X[0])
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('image', X[1])
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('image', X[2])
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('image', X[3])
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('image', X[4])
# cv2.waitKey()
# cv2.destroyAllWindows()

def label_img(dir):
    if dir == 'Diver': return [1, 0, 0, 0]
    elif dir == 'Fish': return [0, 1, 0, 0]
    elif dir == 'Kelp': return [0, 0, 1, 0]
    elif dir == 'Octopus': return [0, 0, 0, 1]

def create_train_data():
    training_data = []
    for dir in tqdm.tqdm(os.listdir(TRAIN_DIR)):
        CURRENT_DIR = os.path.join(TRAIN_DIR, dir)
        for img in os.listdir('Files/Classifications/CreatureClassification/{}'.format(dir)):
            label = label_img(dir)
            path = os.path.join(CURRENT_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('ConvNetData.npy', training_data)
    return training_data


IMG_SIZE = 200
LR = 0.0001
MODEL_NAME = '{}-{}.model'.format(LR, 'VGG16')
TRAIN_DIR = 'Files/Classifications/CreatureClassification/'

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
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='targets')
model = tflearn.DNN(network, tensorboard_dir='log')

#Data = create_train_data()
data = np.load('ConvNetData.npy')

shuffle(data)

train = data[:-500]
test = data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')
else:
    model.fit({'input': X}, {'targets': Y}, n_epoch=5, batch_size=32, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

fig = plt.figure()

for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0: str_label = 'Diver'
    elif np.argmax(model_out) == 1: str_label = 'Fish'
    elif np.argmax(model_out) == 2: str_label = 'Kelp'
    else: str_label = 'Octopus'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()