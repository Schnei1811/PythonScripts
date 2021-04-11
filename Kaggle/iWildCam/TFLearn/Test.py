import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import json
import os.path
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pandas as pd


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
    VGG16 = regression(VGG16, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(VGG16)
    return model


def Create_Ouput_List(IMG_SIZE, checkpoint_path):
    model = VGG16(IMG_SIZE)
    model.load(checkpoint_path)

    for file in tqdm(TEST_LIST):
        path = os.path.join(TEST_DIR, file)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))

        model_img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = np.argmax(model.predict([model_img])[0])
        TEST_DICT[file].append(model_out)

        # if model_out == 1: print('ANIMAL')
        # else: print('NO ANIMAL')

        # cv2.imshow('org_img', org_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows

    tf.reset_default_graph()


def Build_Output_Val():
    anno_dir = 'D:PythonData/iWildCam/Annotations/'
    with open(anno_dir + 'val_annotations.json') as json_data: anno_val = json.load(json_data)
    validation_data = []
    print(anno_val['annotations'])
    print(TEST_DICT)
    # for filename in tqdm(anno_val['annotations']):
    #     if filename['category_id'] == 0: label = [1, 0]
    #     elif filename['category_id'] == 1: label = [0, 1]

    for file in TEST_DICT:
        print(TEST_DICT[file])
        if sum(TEST_DICT[file]) == NUM_MODELS:
            animal = 1
        else:
            animal = 0
        id = file.split('.')[0]
        Output_List.append([id, animal])

    my_df = pd.DataFrame(Output_List)

    my_df.to_csv('iWildSubmissionOutput.csv', index=False, header=False)

def Build_Output_Test():
    for file in TEST_DICT:
        print(TEST_DICT[file])
        if sum(TEST_DICT[file]) >= 3:
            animal = 1
        else:
            animal = 0
        id = file.split('.')[0]
        Output_List.append([id, animal])

    my_df = pd.DataFrame(Output_List)

    my_df.to_csv('iWildSubmissionOutput.csv', index=False, header=False)


os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680
#TEST_DIR = 'D:PythonData/iWildCam/LowRes/Test/'
TEST_DIR = 'D:PythonData/iWildCam/LowRes/Val/'
LR = 1e-4
NUM_MODELS = 5

# VGG_100_8952_model_checkpoint_path = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
#                                      'iWildCam-0.0001-1xdif-100-VGG16.modelcheckpoints\\8952'
# VGG_100_9040_model_checkpoint_path = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
#                                      'iWildCam-0.0001-1xdif-100-VGG16.modelcheckpoints\\9040'
# VGG_50_9060_model_checkpoint_path = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
#                                     'iWildCam-0.0001-1xdif-50-VGG16.modelcheckpoints\\9060'


model1 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
                                    'iWildCam-1e-05-150-VGG16.modelcheckpoints\\9274'
model2 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
                                     'iWildCam-0.0001-150-VGG16.modelcheckpoints\\9148'
model3 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
                                     'iWildCam-0.0001-1xdif-50-VGG16.modelcheckpoints\\9060'
model4 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
                                    'iWildCam-1e-05-150-VGG16.modelcheckpoints\\9200'
model5 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\' \
                                     'iWildCam-0.0001-1xdif-100-VGG16.modelcheckpoints\\9040'

print(os.path.isfile(model1 + '.meta'))
print(os.path.isfile(model2 + '.meta'))
print(os.path.isfile(model3 + '.meta'))
print(os.path.isfile(model4 + '.meta'))
print(os.path.isfile(model5 + '.meta'))

TEST_LIST = os.listdir(TEST_DIR)
TEST_LIST = TEST_LIST[:10]
TEST_DICT = {}

for file in tqdm(TEST_LIST):
    TEST_DICT[file] = []



print(TEST_LIST)

print(TEST_DICT)

# 0 = No Animal
# 1 = Animal

Create_Ouput_List(150, model1)
Create_Ouput_List(150, model2)
Create_Ouput_List(50, model3)
Create_Ouput_List(150, model4)
Create_Ouput_List(100, model5)




Build_Output_Val()

#Build_Output_Test()


























