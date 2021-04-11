import cv2
import numpy as np
import tqdm
import os
from random import shuffle
from sklearn.cluster import KMeans
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def create_resized_images():
    training_data = []
    for imgname in os.listdir(ORIGINAL_IMAGES_DIR):
        path = os.path.join(ORIGINAL_IMAGES_DIR, imgname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #cv2.imwrite(MODIFIED_DIR + imgname, img)
    return training_data

def save_image_dot_list(COLOUR, LABEL):
    xydict = {}
    imgcounter = 0
    for imgname in os.listdir(LABELLED_IMAGES_DIR):
        coloutcount = 0
        path = os.path.join(LABELLED_IMAGES_DIR, imgname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        for y in range(0, IMG_SIZE):
            for x in range(0, IMG_SIZE):
                if img[y, x][0] == COLOUR[0] and img[y, x][1] == COLOUR[1] and img[y, x][2] == COLOUR[2]:
                    if coloutcount == 0:
                        COLOUR_dot_array = [x, y]
                        coloutcount += 1
                    else: COLOUR_dot_array = np.vstack(([x, y], COLOUR_dot_array))
        kmeans = KMeans(n_clusters=len(COLOUR_dot_array))
        kmeans.fit(COLOUR_dot_array)
        xy = kmeans.cluster_centers_.astype(int)
        xydict[imgcounter] = xy
        imgcounter += 1
    print(COLOUR, xydict)
    np.save('Files/OctopusImages/DotCoordinates/' + LABEL + 'Coordinates.npy', xydict)

def segment_mutiple_dot_images(COORDINATES, SAVEDIR):
    dotsdict = np.load(COORDINATES).item()
    imgcounter = 0
    frameaddition = int(ROI_SIZE / 2)
    for imgname in os.listdir(ORIGINAL_IMAGES_DIR):
        path = os.path.join(ORIGINAL_IMAGES_DIR, imgname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        points = dotsdict[imgcounter]
        print('image', imgcounter)
        for point_counter in range(len(points)):
            x, y = int(points[point_counter][0]), int(points[point_counter][1])

            if y <= frameaddition: y += frameaddition - y
            elif y >= IMG_SIZE - frameaddition: y -= frameaddition - (IMG_SIZE - y)
            if x <= frameaddition: x += frameaddition - x
            elif x >= IMG_SIZE - frameaddition: x -= frameaddition - (IMG_SIZE - x)

            ROI = img[y - frameaddition:y + frameaddition, x - frameaddition:x + frameaddition]

            cv2.imwrite(SAVEDIR + imgname + 'PD' + str(point_counter) + '.png', ROI)
        imgcounter += 1

def assign_sections_from_ConvNet():
    model.load(MODEL_DIR)
    i = 1
    while True:
        img = cv2.imread(ORIGINAL_IMAGES_DIR + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mantletipdict = {}
        mantlecenterdict = {}
        tentacledict = {}
        bodydict = {}
        nooctodict = {}
        imgdict = {}
        sectioncounter = 0

        for x in range(0, IMG_SIZE, PIXEL_SKIP):
            for y in range(0, IMG_SIZE, PIXEL_SKIP):
                roi = img[x:ROI_SIZE + x, y:ROI_SIZE + y]
                roi = roi.reshape(ROI_SIZE, ROI_SIZE, 1)
                model_out = model.predict([roi])[0]

                mantlecenterdict[sectioncounter] = model_out[0]
                mantletipdict[sectioncounter] = model_out[1]
                tentacledict[sectioncounter] = model_out[2]
                bodydict[sectioncounter] = model_out[3]
                nooctodict[sectioncounter] = model_out[4]

                print(model_out)

                result = np.argmax(model_out)
                if result == 0: print('MantleTip')
                elif result == 1: print('MantleCenter')
                elif result == 2: print('Tentacle')
                elif result == 3: print('Body')
                else: print('NoOctopus')
                imgdict[sectioncounter] = [result, model_out[result]]
                sectioncounter += 1

        img = cv2.imread(ORIGINAL_IMAGES_DIR + str(i) + '.png', cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sectioncounter = 0

        alpha = 0.2
        output = img.copy()
        overlay = img.copy()

        print(max(mantletipdict, key=mantletipdict.get))
        print(max(mantlecenterdict, key=mantlecenterdict.get))

        for x in range(0, IMG_SIZE, PIXEL_SKIP):
            for y in range(0, IMG_SIZE, PIXEL_SKIP):
                if max(mantletipdict, key=mantletipdict.get) == sectioncounter:
                    cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), RED, -1)
                elif max(mantlecenterdict, key=mantlecenterdict.get) == sectioncounter:
                    cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), ORANGE, -1)
                elif max(bodydict, key=bodydict.get) == sectioncounter:
                    cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), BLUE, -1)
                else:
                    if imgdict[sectioncounter][0] == 0:
                        cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), RED, -1)
                    elif imgdict[sectioncounter][0] == 1:
                        cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), ORANGE, -1)
                    elif imgdict[sectioncounter][0] == 2:
                        cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), GREEN, -1)
                    elif imgdict[sectioncounter][0] == 3:
                        cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), BLUE, -1)
                    else:
                        cv2.rectangle(overlay, (x, y), (ROI_SIZE + x, ROI_SIZE + y), PURPLE, -1)

                sectioncounter += 1

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        cv2.imshow('img', output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        i += 1

if not os.path.exists('Files/Classifications/OctopusSegmentClassification/MantleTip'):
    os.makedirs('Files/Classifications/OctopusSegmentClassification/MantleTip')
if not os.path.exists('Files/Classifications/OctopusSegmentClassification/MantleCenter'):
    os.makedirs('Files/Classifications/OctopusSegmentClassification/MantleCenter')
if not os.path.exists('Files/Classifications/OctopusSegmentClassification/Tentacle'):
    os.makedirs('Files/Classifications/OctopusSegmentClassification/Tentacle')
if not os.path.exists('Files/Classifications/OctopusSegmentClassification/Body/'):
     os.makedirs('Files/Classifications/OctopusSegmentClassification/Body/')
if not os.path.exists('Files/Classifications/OctopusSegmentClassification/NoOctopus'):
    os.makedirs('Files/Classifications/OctopusSegmentClassification/NoOctopus')

IMG_SIZE = 200
RED = [36, 28, 237]                         #MantleTip
ORANGE = [39, 127, 255]                     #Center of Two Eyes
GREEN = [76, 177, 34]                       #Tentacle
BLUE = [232, 162, 0]                        #Body
PURPLE = [164, 73, 163]                     #No Octopus

#tensorboard --logdir=foo:C:\Users\Stefan\Dropbox\PythonScripts\PythonScripts\opencv\Octopus\Files\log
LR = 1e-3
IMG_SIZE = 200
ROI_SIZE = 40
PIXEL_SKIP = 40
#MODEL_NAME = 'mantletipvscentervsnoocto-{}-{}.model'.format(LR, '8conv-tree')
#MODEL_NAME = 'mantletipvscentervstentaclevsnoocto-{}-{}.model'.format(LR, '8conv-tree')
MODEL_NAME = 'FullModel-{}-{}-{}.model'.format(ROI_SIZE, LR, '8conv-tree')

ORIGINAL_IMAGES_DIR = 'Files/OctopusImages/OctopusOriginals200x200/'
LABELLED_IMAGES_DIR = 'Files/OctopusImages/OctopusLabels200x200/'
MANTLETIP_DIR = 'Files/Classifications/OctopusSegmentClassification/MantleTip/'
MANTLECENTER_DIR = 'Files/Classifications/OctopusSegmentClassification/MantleCenter/'
TENTACLE_DIR = 'Files/Classifications/OctopusSegmentClassification/Tentacle/'
BODY_DIR = 'Files/Classifications/OctopusSegmentClassification/Body/'
NOOCTOPUS_DIR = 'Files/Classifications/OctopusSegmentClassification/NoOctopus/'
RED_COORDINATES = 'Files/OctopusImages/DotCoordinates/MantleTipCoordinates.npy'
ORANGE_COORDINATES = 'Files/OctopusImages/DotCoordinates/MantleCenterCoordinates.npy'
GREEN_COORDINATES = 'Files/OctopusImages/DotCoordinates/TentacleCoordinates.npy'
BLUE_COORDINATES = 'Files/OctopusImages/DotCoordinates/BodyCoordinates.npy'
PURPLE_COORDINATES = 'Files/OctopusImages/DotCoordinates/NoOctopusCoordinates.npy'
MODEL_DIR = 'Files/Classifications/Models/' + MODEL_NAME

convnet = input_data(shape=[None, ROI_SIZE, ROI_SIZE, 1], name='input')
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 512, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')


#create_resized_images()
# save_image_dot_list(RED, 'MantleTip')
# save_image_dot_list(ORANGE, 'MantleCenter')
# save_image_dot_list(GREEN, 'Tentacle')
# save_image_dot_list(BLUE, 'Body')
# save_image_dot_list(PURPLE, 'NoOctopus')
# segment_mutiple_dot_images(RED_COORDINATES, MANTLETIP_DIR)
# segment_mutiple_dot_images(ORANGE_COORDINATES, MANTLECENTER_DIR)
# segment_mutiple_dot_images(GREEN_COORDINATES, TENTACLE_DIR)
# segment_mutiple_dot_images(BLUE_COORDINATES, BODY_DIR)
# segment_mutiple_dot_images(PURPLE_COORDINATES, NOOCTOPUS_DIR)
assign_sections_from_ConvNet()
