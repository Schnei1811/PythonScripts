import cv2
from tqdm import tqdm
import os
import numpy as np


imagenum = 6800
pathlen = len('Data/rawimage/blank/IMG_' + str(imagenum) + '.jpg')
lennum = len(str(imagenum))
startnum = 1 + pathlen - 4 - lennum             # len .jpg

IMG_WIDTH, IMG_HEIGHT = 980, 1308           #smallest image dimensions


while imagenum < 7296:
    try:
        path = 'Data/rawimages/print/IMG_' + str(imagenum) + '.jpg'
        path2 = path[0:startnum] + str(imagenum+1) + path[startnum+lennum:]
        print(path2)
        path3 = path[0:startnum] + str(imagenum+2) + path[startnum+lennum:]
        img1 = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),  (IMG_WIDTH, IMG_HEIGHT))
        img2 = cv2.resize(cv2.imread(path2, cv2.IMREAD_COLOR), (IMG_WIDTH, IMG_HEIGHT))
        img3 = cv2.resize(cv2.imread(path3, cv2.IMREAD_COLOR), (IMG_WIDTH, IMG_HEIGHT))
        img1 = img1[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img2 = img2[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img3 = img3[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img = np.concatenate((img1, img2), axis=0)
        img = np.concatenate((img, img3), axis=0)
        cv2.imwrite('Data/joinedimages/print/{}.jpg'.format(imagenum), img)
        imagenum += 3
    except:
        imagenum += 1

imagenum = 6782

while imagenum < 7296:
    try:
        path = 'Data/rawimages/blank/IMG_' + str(imagenum) + '.jpg'
        path2 = path[0:startnum] + str(imagenum+1) + path[startnum+lennum:]
        print(path2)
        path3 = path[0:startnum] + str(imagenum+2) + path[startnum+lennum:]
        img1 = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),  (IMG_WIDTH, IMG_HEIGHT))
        img2 = cv2.resize(cv2.imread(path2, cv2.IMREAD_COLOR), (IMG_WIDTH, IMG_HEIGHT))
        img3 = cv2.resize(cv2.imread(path3, cv2.IMREAD_COLOR), (IMG_WIDTH, IMG_HEIGHT))
        img1 = img1[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img2 = img2[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img3 = img3[int(0.35 * IMG_HEIGHT):int(0.55 * IMG_HEIGHT), int(0.15 * IMG_WIDTH):int(0.85 * IMG_WIDTH)]
        img = np.concatenate((img1, img2), axis=0)
        img = np.concatenate((img, img3), axis=0)
        cv2.imwrite('Data/joinedimages/blank/{}.jpg'.format(imagenum), img)
        imagenum += 3
    except:
        imagenum += 1
