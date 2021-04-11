import numpy as np
import tflearn
import cv2
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def confidence_gradient(img):
    SCAN_SIZE = 20
    STEP_SIZE = 5
    rx1, ry1, rx2, ry2 = 0, 0, SCAN_SIZE, SCAN_SIZE
    # cv2.imshow('orgimg', orgimg)
    # cv2.imwrite('Files/Classifications/VideoImages2/{}{}{}{}{}.jpg'.format(x1, y1, x2, y2, framenum), roi)
    while SCAN_SIZE > 19:
        while ry2 < 150:
            roi = img[ry1:ry2, rx1:rx2]
            roi = cv2.resize(roi, (CONV_IMG_SIZE, CONV_IMG_SIZE)).reshape(CONV_IMG_SIZE, CONV_IMG_SIZE, 1)
            model_out = model.predict([roi])[0]
            print(model_out)
            if np.argmax(model_out) == 0:
                if rx1 < minx: minx = rx1
                if rx2 > maxx: maxx = rx2
                if ry1 < miny: miny = ry1
                if ry2 > maxy: maxy = ry2
                cv2.imshow('roi', roi)
                cv2.waitKey()
                cv2.destroyAllWindows
            rx1 += STEP_SIZE
            rx2 += STEP_SIZE
            if rx2 > 150:
                rx1, rx2 = 0, SCAN_SIZE
                ry1 += STEP_SIZE
                ry2 += STEP_SIZE
        SCAN_SIZE -= 50
        rx1, ry1, rx2, ry2 = 0, 0, SCAN_SIZE, SCAN_SIZE






CONV_IMG_SIZE = 150
IMG_SIZE = 150
LR = 0.0001
MODEL_NAME = '0.0001-VGG16.model'
NUM_CLASSIFICATIONS = 6

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




img = cv2.imread('octotest13.jpg', cv2.IMREAD_GRAYSCALE)
confidence_gradient(img)