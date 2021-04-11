import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.helpers.regularizer import add_weights_regularizer
import matplotlib.pyplot as plt
import json
import keras



def AlexNet():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'AlexNet')
    AlexNet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)
    AlexNet = conv_2d(AlexNet, 96, 11, strides=4, activation='relu', regularizer='L2', weight_decay=0.0001)
    AlexNet = max_pool_2d(AlexNet, 3, strides=2)
    AlexNet = local_response_normalization(AlexNet)
    AlexNet = conv_2d(AlexNet, 256, 5, activation='relu')
    AlexNet = max_pool_2d(AlexNet, 3, strides=2)
    AlexNet = local_response_normalization(AlexNet)
    AlexNet = conv_2d(AlexNet, 384, 3, activation='relu')
    AlexNet = conv_2d(AlexNet, 384, 3, activation='relu')
    AlexNet = conv_2d(AlexNet, 256, 3, activation='relu')
    AlexNet = max_pool_2d(AlexNet, 3, strides=2)
    AlexNet = local_response_normalization(AlexNet)
    AlexNet = fully_connected(AlexNet, 4096, activation='tanh')
    AlexNet = dropout(AlexNet, 0.5)
    AlexNet = fully_connected(AlexNet, 4096, activation='tanh')
    AlexNet = dropout(AlexNet, 0.5)
    AlexNet = fully_connected(AlexNet, 2, activation='softmax')
    AlexNet = regression(AlexNet, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='targets')
    model = tflearn.DNN(AlexNet, best_checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                        tensorboard_dir='log')
    return MODEL_NAME, model

def VGG16():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'VGG16')
    VGG16 = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)

    VGG16 = conv_2d(VGG16, 64, 3, activation='relu', scope='conv1_1', regularizer='L2', weight_decay=0.0001)
    VGG16 = conv_2d(VGG16, 64, 3, activation='relu', scope='conv1_2')
    VGG16 = max_pool_2d(VGG16, 2, strides=2, name='maxpool1')

    VGG16 = conv_2d(VGG16, 128, 3, activation='relu', scope='conv2_1')
    VGG16 = conv_2d(VGG16, 128, 3, activation='relu', scope='conv2_2')
    VGG16 = max_pool_2d(VGG16, 2, strides=2, name='maxpool2')

    VGG16 = conv_2d(VGG16, 256, 3, activation='relu', scope='conv3_1')
    VGG16 = conv_2d(VGG16, 256, 3, activation='relu', scope='conv3_2')
    VGG16 = conv_2d(VGG16, 256, 3, activation='relu', scope='conv3_3')
    VGG16 = max_pool_2d(VGG16, 2, strides=2, name='maxpool3')

    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv4_1')
    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv4_2')
    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv4_3')
    VGG16 = max_pool_2d(VGG16, 2, strides=2, name='maxpool4')

    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv5_1')
    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv5_2')
    VGG16 = conv_2d(VGG16, 512, 3, activation='relu', scope='conv5_3')
    VGG16 = max_pool_2d(VGG16, 2, strides=2, name='maxpool5')

    VGG16 = fully_connected(VGG16, 4096, activation='relu', scope='fc6')
    VGG16 = dropout(VGG16, 0.5, name='dropout1')

    VGG16 = fully_connected(VGG16, 4096, activation='relu', scope='fc7')
    VGG16 = dropout(VGG16, 0.5, name='dropout2')

    VGG16 = fully_connected(VGG16, 2, activation='softmax', scope='fc8')
    VGG16 = regression(VGG16, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(VGG16, best_checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                    tensorboard_dir='log')

    return MODEL_NAME, model

def VGG19():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'VGG19')
    VGG19 = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)

    VGG19 = conv_2d(VGG19, 64, 3, activation='relu', scope='conv1_1', regularizer='L2', weight_decay=0.0001)
    VGG19 = conv_2d(VGG19, 64, 3, activation='relu', scope='conv1_2')
    VGG19 = max_pool_2d(VGG19, 2, strides=2, name='maxpool1')

    VGG19 = conv_2d(VGG19, 128, 3, activation='relu', scope='conv2_1')
    VGG19 = conv_2d(VGG19, 128, 3, activation='relu', scope='conv2_2')
    VGG19 = max_pool_2d(VGG19, 2, strides=2, name='maxpool2')

    VGG19 = conv_2d(VGG19, 256, 3, activation='relu', scope='conv3_1')
    VGG19 = conv_2d(VGG19, 256, 3, activation='relu', scope='conv3_2')
    VGG19 = conv_2d(VGG19, 256, 3, activation='relu', scope='conv3_3')
    VGG19 = conv_2d(VGG19, 256, 3, activation='relu', scope='conv3_4')
    VGG19 = max_pool_2d(VGG19, 2, strides=2, name='maxpool3')

    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv4_1')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv4_2')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv4_3')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv4_4')
    VGG19 = max_pool_2d(VGG19, 2, strides=2, name='maxpool4')

    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv5_1')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv5_2')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv5_3')
    VGG19 = conv_2d(VGG19, 512, 3, activation='relu', scope='conv5_4')
    VGG19 = max_pool_2d(VGG19, 2, strides=2, name='maxpool5')
    VGG19 = tflearn.layers.core.flatten(VGG19, name='Flatten')

    VGG19 = fully_connected(VGG19, 4096, activation='relu', scope='fc6')
    VGG19 = dropout(VGG19, 0.5, name='dropout1')

    VGG19 = fully_connected(VGG19, 4096, activation='relu', scope='fc7')
    VGG19 = dropout(VGG19, 0.5, name='dropout2')

    VGG19 = fully_connected(VGG19, 2, activation='softmax', scope='fc8')
    VGG19 = regression(VGG19, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(VGG19, best_checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                        tensorboard_dir='log')
    return MODEL_NAME, model

def ResNext():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'ResNet')
    n=5
    net = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.resnext_block(net, n, 16, 32)
    net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 32, 32)
    net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 64, 32)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', name='targets')
    # Training
    model = tflearn.DNN(net, checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                        tensorboard_dir='log/')
    return MODEL_NAME, model

def DenseNet():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'DenseNet')
    # Growth Rate (12, 16, 32, ...)
    k = 12
    # Depth (40, 1xdif-100, ...)
    L = 40
    nb_layers = int((L - 4) / 3)
    net = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.densenet_block(net, nb_layers, k)
    net = tflearn.densenet_block(net, nb_layers, k)
    net = tflearn.densenet_block(net, nb_layers, k)
    net = tflearn.global_avg_pool(net)

    # Regression
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', name='targets')
    # Training
    model = tflearn.DNN(net, checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                        tensorboard_dir='log/')
    return MODEL_NAME, model


def GoogLeNet():
    MODEL_NAME = 'iWildCam-{}-{}-{}.model'.format(LR, IMG_SIZE, 'GoogLeNet')
    network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input', data_augmentation=img_aug)
    conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    # 3a
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu', name='inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                    name='inception_3a_pool_1_1')
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                mode='concat', axis=3)

    # 3b
    inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                      name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                      name='inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                    name='inception_3b_pool_1_1')
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                mode='concat', axis=3, name='inception_3b_output')
    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

    # 4a
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4a_pool_1_1')
    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=3, name='inception_4a_output')

    # 4b
    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                      name='inception_4b_3_3_reduce')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                      name='inception_4b_5_5_reduce')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4b_5_5')
    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4b_pool_1_1')
    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                mode='concat', axis=3, name='inception_4b_output')

    # 4c
    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                      name='inception_4c_3_3_reduce')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                      name='inception_4c_5_5_reduce')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4c_5_5')
    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4c_pool_1_1')
    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                mode='concat', axis=3, name='inception_4c_output')

    # 4d
    inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                      name='inception_4d_3_3_reduce')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                      name='inception_4d_5_5_reduce')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4d_5_5')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4d_pool_1_1')
    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                mode='concat', axis=3, name='inception_4d_output')

    # 4e
    inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                      name='inception_4e_3_3_reduce')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                      name='inception_4e_5_5_reduce')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                    name='inception_4e_pool_1_1')
    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3,
                                mode='concat')
    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

    # 5a
    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5a_5_5')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                    name='inception_5a_pool_1_1')
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,
                                mode='concat')

    # 5b
    inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                      name='inception_5b_3_3_reduce')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                      name='inception_5b_5_5_reduce')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                    name='inception_5b_pool_1_1')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3,
                                mode='concat')
    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    # fc
    inception = fully_connected(pool5_7_7, 2, activation='softmax')
    inception = regression(inception, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR,
                           name='targets')

    model = tflearn.DNN(inception, best_checkpoint_path='{}checkpoints/'.format(MODEL_NAME), best_val_accuracy=0.75,
                        tensorboard_dir='log')
    return MODEL_NAME, model

os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680

anno_dir = 'D:PythonData/iWildCam/Annotations/'
TRAIN_DIR = 'D:PythonData/iWildCam/LowRes/Train/'
VAL_DIR = 'D:PythonData/iWildCam/LowRes/Val/'
IMG_SIZE = 50
LR = 0.00001

# img_prep = tflearn.ImagePreprocessing()
# img_prep.add_featurewise_zero_center()

img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_blur(sigma_max=3.0)
# img_aug.add_random_crop((IMG_SIZE, IMG_SIZE), int(IMG_SIZE/20))
# img_aug.add_random_rotation(max_angle=10.0)


print('Loading Train Data...')
#train_data = np.load('../data/{}_pixel_train_data.npy'.format(IMG_SIZE))
train_data = np.load('../data/{}_pixel_train_multitask_data.npy'.format(IMG_SIZE))

#train_data = np.load('50_pixel_train_data2000.npy')
#train_data = np.load('100_pixel_train_data2000.npy')
# train_data = np.load('150_pixel_train_data2000.npy')
print('Train ready')


print('Loading Val Data...')
#val_data = np.load('../data/{}_pixel_val_data.npy'.format(IMG_SIZE))
val_data = np.load('../data/{}_pixel_val_multitask_data.npy'.format(IMG_SIZE))

#val_data = np.load('50_pixel_val_data200.npy')
#val_data = np.load('100_pixel_val_data200.npy')
# val_data = np.load('150_pixel_val_data200.npy')
print('Val ready')


X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train_data]

val_x = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_y = [i[1] for i in val_data]


#MODEL_NAME, model = AlexNet()
#MODEL_NAME, model = VGG16()
#MODEL_NAME, model = VGG19()
#MODEL_NAME, model = DenseNet()
#MODEL_NAME, model = ResNext()
MODEL_NAME, model = GoogLeNet()

#model.load('C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\TFlearn\\{}checkpoints\\8481'.format(MODEL_NAME))

model.fit({'input': X}, {'targets': Y}, n_epoch=200, validation_set=({'input': val_x}, {'targets': val_y}),
          snapshot_epoch=True, show_metric=True, run_id=MODEL_NAME, shuffle=True)

model.save(MODEL_NAME)

#tensorboard --logdir=log:C:\Users\Stefan\Desktop\PythonScripts\iWildCam\log

fig = plt.figure()

for num, data in enumerate(val_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1: str_label = 'Animal'
    else: str_label = 'No Animal'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()









