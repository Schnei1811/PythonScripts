import numpy as np
import os
import cv2
import tflearn
import scipy
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
np.set_printoptions(threshold=np.nan)


CONV_IMG_SIZE = 150
NUM_CLASSIFICATIONS = 8
LR = 0.0001
MODEL_NAME = '0.0001-VGG16.model'
#MODEL_NAME = 'Models/6ClassVGG16.model'

network = input_data(shape=[None, CONV_IMG_SIZE, CONV_IMG_SIZE, 1], name='input')
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

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')
else: print('Error Loading Model')

imgname = 'octotest11'
img = cv2.imread('{}.jpg'.format(imgname), cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]
print(height, width)
SCAN_SIZE = 150
STEP_SIZE = 20

minx, miny, maxx, maxy = width, height, 0, 0
x1, y1, x2, y2 = 0, 0, SCAN_SIZE, SCAN_SIZE

framenum = 0

while SCAN_SIZE > 149:
    while y2 < height:
        roi = img[y1:y2, x1:x2]
        roi = cv2.resize(roi, (CONV_IMG_SIZE, CONV_IMG_SIZE)).reshape(CONV_IMG_SIZE, CONV_IMG_SIZE, 1)
        model_out = model.predict([roi])[0]

        # if np.argmax(model_out) == 0:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/Octopus/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[0]), roi)
        # elif np.argmax(model_out) == 1:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/Kelp/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[1]), roi)
        # elif np.argmax(model_out) == 2:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/Fish/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[2]), roi)
        # elif np.argmax(model_out) == 3:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/Environment/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[3]), roi)
        # elif np.argmax(model_out) == 4:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/Diver/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[4]), roi)
        # elif np.argmax(model_out) == 5:
        #     cv2.imwrite('Files/Classifications/SegmentedImages/MultiClass/{}{}{}{}{}a{}.jpg'.format(rx1, ry1, rx2, ry2, framenum, model_out[5]), roi)

        if model_out[0] > 0.99:
            roioctoinfo = np.array([x1, y1, x2, y2])
            if 'octopusscansquares' not in locals(): octopusscansquares = roioctoinfo
            else: octopusscansquares = np.vstack((roioctoinfo, octopusscansquares))
            cv2.imshow('roi{}{}'.format(x1,y1), roi)
            cv2.imwrite('Files/Classifications/testimages/roi{}{}.jpg'.format(x1,y1), roi)
        x1 += STEP_SIZE
        x2 += STEP_SIZE
        if x2 > width:
            roi = img[y1:y2, width-SCAN_SIZE:width]
            roi = cv2.resize(roi, (CONV_IMG_SIZE, CONV_IMG_SIZE)).reshape(CONV_IMG_SIZE, CONV_IMG_SIZE, 1)
            model_out = model.predict([roi])[0]
            if model_out[0] > 0.99:
                roioctoinfo = np.array([x1, y1, x2, y2])
                if 'octopusscansquares' not in locals(): octopusscansquares = roioctoinfo
                else: octopusscansquares = np.vstack((roioctoinfo, octopusscansquares))
                cv2.imshow('roi{}{}'.format(x1, y1), roi)
                cv2.imwrite('Files/Classifications/testimages/roi{}{}.jpg'.format(x1, y1), roi)
            x1, x2 = 0, SCAN_SIZE
            y1 += STEP_SIZE
            y2 += STEP_SIZE
    SCAN_SIZE -= 50
    x1, y1, x2, y2 = 0, 0, SCAN_SIZE, SCAN_SIZE

intersectiongraph = np.zeros((len(octopusscansquares), len(octopusscansquares)))

for q in range(0, len(octopusscansquares)):
    for r in range(0, len(octopusscansquares)):
        if q == r: intersectiongraph[q, r] = 0
        elif (octopusscansquares[q, 2] < octopusscansquares[r, 0] or octopusscansquares[r, 2] < octopusscansquares[q, 0]
              or octopusscansquares[q, 3] < octopusscansquares[r, 1] or octopusscansquares[r, 3] < octopusscansquares[q, 1]):
            intersectiongraph[q, r] = 0
        else: intersectiongraph[q, r] = 1

uniquesquares = np.array([scipy.sparse.csgraph.connected_components(intersectiongraph, directed=False, connection='weak', return_labels=True)[1]])
consideredsquares = np.concatenate((uniquesquares.T, octopusscansquares), axis=1)
consideredsquares = consideredsquares[np.argsort(consideredsquares[:, 0])]

squarecount = 0
minx, miny, maxx, maxy = width, height, 0, 0
for q, r in enumerate(consideredsquares):
    # print(squarecount)
    if consideredsquares[q, 1] < minx: minx = consideredsquares[q, 1]
    if consideredsquares[q, 2] < miny: miny = consideredsquares[q, 2]
    if consideredsquares[q, 3] > maxx: maxx = consideredsquares[q, 3]
    if consideredsquares[q, 4] > maxy: maxy = consideredsquares[q, 4]
    try:
        if consideredsquares[q + 1, 0] - consideredsquares[q, 0] > 0:
            if squarecount == 0: finalsquares = np.array([framenum, 0, minx, miny, maxx, maxy])
            else: finalsquares = np.vstack((np.array([framenum, 0, minx, miny, maxx, maxy]), finalsquares))
            squarecount += 1
            minx, miny, maxx, maxy = width, height, 0, 0
    except:
        if squarecount == 0: finalsquares = np.array([framenum, 0, minx, miny, maxx, maxy])
        else: finalsquares = np.vstack((np.array([framenum, 0, minx, miny, maxx, maxy]), finalsquares))
        squarecount += 1
        minx, miny, maxx, maxy = width, height, 0, 0

print(finalsquares)

if finalsquares.ndim == 1: cv2.rectangle(img, (finalsquares[2], finalsquares[3]), (finalsquares[4], finalsquares[5]), 255, 2)
else:
    for i, j in enumerate(finalsquares):
        cv2.rectangle(img, (finalsquares[i, 2], finalsquares[i, 3]), (finalsquares[i, 4], finalsquares[i, 5]), 255, 2)

cv2.imwrite('{}TEST.jpg'.format(imgname), img)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows