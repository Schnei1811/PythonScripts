import os
import cv2
import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def VGG16(IMG_SIZE):
    VGG16 = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    VGG16 = conv_2d(VGG16, 64, 3, activation='relu', scope='conv1_1')
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
    VGG16 = regression(VGG16)

    model = tflearn.DNN(VGG16)
    return model


def Create_Ouput_List(IMG_SIZE, checkpoint_path):
    model = VGG16(IMG_SIZE)
    model.load(checkpoint_path)

    for file in tqdm(IMAGE_LIST):
        try:
            img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            model_img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
            model_out = np.argmax(model.predict([model_img])[0])
            IMAGE_DICT[file].append(model_out)
        except:
            print('File Found that is not an Image: {}'.format(file))

    tf.reset_default_graph()

model1 = 'Models/VGG16-150/9274'
model2 = 'Models/VGG16-150/9200'
model3 = 'Models/VGG16-150/9148'
model4 = 'Models/VGG16-100/9040'
model5 = 'Models/VGG16-50/9060'

IMAGE_LIST = []
OUTPUT_LIST = []
IMAGE_DICT = {}

rootDir = 'Photos/'
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        IMAGE_LIST.append(dirName + '/' + fname)

for file in tqdm(IMAGE_LIST):
    IMAGE_DICT[file] = []

Create_Ouput_List(150, model1)
Create_Ouput_List(150, model2)
Create_Ouput_List(150, model3)
Create_Ouput_List(100, model4)
Create_Ouput_List(50, model5)

for file in IMAGE_DICT:
    if sum(IMAGE_DICT[file]) >= 3:
        animal = 1
    else:
        animal = 0
    OUTPUT_LIST.append([file, animal])

my_df = pd.DataFrame(OUTPUT_LIST)

my_df.to_csv('BinaryAnimalOutput.csv', index=False, header=False)














