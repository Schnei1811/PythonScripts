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
from sklearn import metrics

# def label_img(dir):
#     if dir == 'TrueOctopus': return [1, 0, 0, 0, 0, 0, 0, 0]
#     elif dir == 'OctoScan': return [0, 1, 0, 0, 0, 0, 0, 0]
#     elif dir == 'PartOctopus': return [0, 0, 1, 0, 0, 0, 0, 0]
#     elif dir == 'MultiFish': return [0, 0, 0, 1, 0, 0, 0, 0]
#     elif dir == 'Kelp': return [0, 0, 0, 0, 1, 0, 0, 0]
#     elif dir == 'Fish': return [0, 0, 0, 0, 0, 1, 0, 0]
#     elif dir == 'Environment': return [0, 0, 0, 0, 0, 0, 1, 0]
#     elif dir == 'Diver': return [0, 0, 0, 0, 0, 0, 0, 1]

def label_img(dir):
    if dir == 'Octopus': return [1, 0, 0, 0, 0, 0]
    elif dir == 'Kelp': return [0, 1, 0, 0, 0, 0]
    elif dir == 'Fish': return [0, 0, 1, 0, 0, 0]
    elif dir == 'Environment': return [0, 0, 0, 1, 0, 0]
    elif dir == 'Diver': return [0, 0, 0, 0, 1, 0]
    elif dir == 'MultiClass': return [0, 0, 0, 0, 0, 1]

# def label_img(dir):
#     if dir == 'Octopus': return [1, 0, 0, 0]
#     elif dir == 'NotOcto': return [0, 1, 0, 0]
#     elif dir == 'Diver': return [0, 0, 1, 0]
#     elif dir == 'MultiClass': return [0, 0, 0, 1]

# def label_img(dir):
#     if dir == 'Octopus': return [1, 0]
#     elif dir == 'Kelp': return [0, 1]
#     elif dir == 'Fish': return [0, 1]
#     elif dir == 'Environment': return [0, 1]
#     elif dir == 'Diver': return [0, 1]

def create_train_data():
    training_data = []
    for dir in tqdm.tqdm(os.listdir(TRAIN_DIR)):
        CURRENT_DIR = os.path.join(TRAIN_DIR, dir)
        for img in os.listdir(TRAIN_DIR + '/{}'.format(dir)):
            label = label_img(dir)
            path = os.path.join(CURRENT_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('ConvNetTESTData.npy', training_data)
    return training_data


IMG_SIZE = 150
LR = 0.0001
NUM_CLASSIFICATIONS = 8
NumEpochs = 5
MODEL_NAME = '{}-{}.model'.format(LR, 'VGG16')

TRAIN_DIR = 'Files/Classifications/ConciseCreatureClassification/'

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

data = create_train_data()
#Data = np.load('ConvNetData.npy')

lentrainset = int(len(data)*.8)
lentestset = len(data) - lentrainset
np.random.shuffle(data)

print('Len Train Set', lentrainset)
print('Len Test Set', len(data) - lentrainset)

train = data[:-lentestset]
test = data[-lentestset:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')
else:
    model.fit({'input': X}, {'targets': Y}, n_epoch=NumEpochs, batch_size=32, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

fig = plt.figure()

print(test_x.shape)
print(test_y)
print(test_y.shape)
print(type(test_y))

y_pred = np.array([])

for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    y_pred.append(np.argmax(model.predict([data])[0]))


print(y_pred)
print(y_pred.shape)
print(type(y_pred))



print(metrics.f1_score(test[:12], y_pred, average='weighted'))
print(metrics.confusion_matrix(test[:12], y_pred))

# for num, Data in enumerate(test):
#     y_pred = model.predict([Data])[0]
#     print(y_pred)
#
# f1 = metrics.f1_score(test_y, Y, average='weighted')
# print(f1)
# metrics.confusion_matrix(test_y, Y)


for num, data in enumerate(test[:12]):
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0: str_label = 'TrueOctopus'
    elif np.argmax(model_out) == 1: str_label = 'OctoScan'
    elif np.argmax(model_out) == 2: str_label = 'PartOcto'
    elif np.argmax(model_out) == 3: str_label = 'MultiFish'
    elif np.argmax(model_out) == 4: str_label = 'Kelp'
    elif np.argmax(model_out) == 5: str_label = 'Fish'
    elif np.argmax(model_out) == 6: str_label = 'Environment'
    elif np.argmax(model_out) == 7: str_label = 'Diver'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

