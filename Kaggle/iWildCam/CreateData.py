from random import shuffle
from tqdm import tqdm
import cv2
import os
import numpy as np
import json

def create_train_data():
    with open(anno_dir + 'train_annotations.json') as json_data: anno_train = json.load(json_data)
    training_data = []
    counter, counterbreak = 0, 2000
    for filename in tqdm(anno_train['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(TRAIN_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        counter += 1
        if counter == counterbreak: break
    np.save('data/{}_pixel_train_data{}.npy'.format(IMG_SIZE, counterbreak), training_data)
    shuffle(training_data)
    if counter == 0: np.save('data/{}_pixel_train_data.npy'.format(IMG_SIZE), training_data)
    return training_data

def create_val_data():
    with open(anno_dir + 'val_annotations.json') as json_data: anno_val = json.load(json_data)
    validation_data = []
    counter, counterbreak = 0, 200
    for filename in tqdm(anno_val['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(VAL_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        validation_data.append([np.array(img), label])
        counter += 1
        if counter == counterbreak: break
    np.save('data/{}_pixel_val_data{}.npy'.format(IMG_SIZE, counterbreak), validation_data)
    shuffle(validation_data)
    if counter == 0: np.save('data/{}_pixel_val_data.npy'.format(IMG_SIZE), validation_data)
    return validation_data


def create_multitask_train_data():
    with open(anno_dir + 'train_annotations.json') as json_data: anno_train = json.load(json_data)
    training_data = []
    counter, counterbreak = 0, 2000
    for filename in tqdm(anno_train['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(TRAIN_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        for filename2 in (anno_train['images']):
            if filename2['id'] == filename['image_id']:
                training_data.append([np.array(img), np.array(label), filename2['location']])
                break
    #     counter += 1
    #     if counter == counterbreak: break
    # np.save('data/{}_pixel_train_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), training_data)
    shuffle(training_data)
    if counter == 0: np.save('data/{}_pixel_train_multitask_data.npy'.format(IMG_SIZE), training_data)
    return training_data

def create_multitask_val_data():
    with open(anno_dir + 'val_annotations.json') as json_data: anno_val = json.load(json_data)
    validation_data = []
    counter, counterbreak = 0, 200
    for filename in tqdm(anno_val['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(VAL_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        for filename2 in (anno_val['images']):
            if filename2['id'] == filename['image_id']:
                validation_data.append([np.array(img), np.array(label), filename2['location']])
                break
    #     counter += 1
    #     if counter == counterbreak: break
    # np.save('data/{}_pixel_val_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), validation_data)
    shuffle(validation_data)
    if counter == 0: np.save('data/{}_pixel_val_multitask_data.npy'.format(IMG_SIZE), validation_data)
    return validation_data


def create_mock_train_val_multitask_data():
    with open(anno_dir + 'train_annotations.json') as json_data: anno_train = json.load(json_data)
    training_data = []
    val_data = []
    counter, counterbreak = 0, 7500
    for filename in tqdm(anno_train['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(TRAIN_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        for filename2 in (anno_train['images']):
            if filename2['id'] == filename['image_id']:
                if np.random.random() <= 0.1: val_data.append([np.array(img), np.array(label), filename2['location']])
                else: training_data.append([np.array(img), np.array(label), filename2['location']])
                break
        # counter += 1
        # if counter == counterbreak: break
    counter = 0
    with open(anno_dir + 'val_annotations.json') as json_data: anno_val = json.load(json_data)
    for filename in tqdm(anno_val['annotations']):
        if filename['category_id'] == 0: label = [1, 0]
        elif filename['category_id'] == 1: label = [0, 1]
        path = os.path.join(VAL_DIR, filename['image_id']+'.jpg')
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        for filename2 in (anno_val['images']):
            if filename2['id'] == filename['image_id']:
                if np.random.random() <= 0.1: val_data.append([np.array(img), np.array(label), filename2['location']])
                else: training_data.append([np.array(img), np.array(label), filename2['location']])
                break
    #     counter += 1
    #     if counter == counterbreak: break
    # np.save('data/{}_pixel_mock_train_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), training_data)
    # np.save('data/{}_pixel_mock_val_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), val_data)
    if counter == 0:
        np.save('data/{}_pixel_mock_train_multitask_data.npy'.format(IMG_SIZE), training_data)
        np.save('data/{}_pixel_mock_val_multitask_data.npy'.format(IMG_SIZE), val_data)
    return training_data

anno_dir = 'D:PythonData/iWildCam/Annotations/'
TRAIN_DIR = 'D:PythonData/iWildCam/LowRes/Train/'
VAL_DIR = 'D:PythonData/iWildCam/LowRes/Val/'
IMG_SIZE = 100


# create_train_data()
# create_val_data()
#
# create_multitask_train_data()
# create_multitask_val_data()

create_mock_train_val_multitask_data()
