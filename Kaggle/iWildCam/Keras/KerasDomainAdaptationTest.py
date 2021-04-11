import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import pandas as pd
from keras.preprocessing import image
from keras.backend import clear_session
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.imagenet_utils import preprocess_input


def VGG16():

    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

    first_location_output = Dense(65, activation='softmax', name='first_location_out')(x)
    first_animal_output = Dense(2, activation='softmax', name='first_animal_out')(x)

    x1 = Dense(4096, activation='relu', name='x1fc1')(x)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(4096, activation='relu', name='x1fc2')(x1)
    x1 = Dropout(0.5)(x1)
    second_location_output = Dense(65, activation='softmax', name='second_location_out')(x1)

    x2 = Dense(4096, activation='relu', name='x2fc1')(x)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(4096, activation='relu', name='x2fc2')(x2)
    x2 = Dropout(0.5)(x2)
    second_animal_output = Dense(2, activation='softmax', name='second_animal_out')(x2)

    # Create model.
    model = Model(inputs=img_input, outputs=[first_location_output, first_animal_output,
                                             second_location_output, second_animal_output], name='vgg16')
    return model


def Create_Ouput_List(IMG_SIZE, checkpoint_path):
    model = VGG16()
    model.load_weights(checkpoint_path)

    for file in tqdm(TEST_LIST):
        path = os.path.join(TEST_DIR, file)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        input_img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
        input_img /= 255.0
        model_out = np.argmax(model.predict(input_img)[3][0])
        TEST_DICT[file].append(model_out)

        # if model_out == 1: print('ANIMAL')
        # else: print('NO ANIMAL')
        #
        # cv2.imshow('org_img', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows

    clear_session()


def Build_Output_Val():
    # anno_dir = 'D:PythonData/iWildCam/Annotations/'
    # with open(anno_dir + 'val_annotations.json') as json_data:
    #     anno_val = json.load(json_data)
    validation_data = []
    #print(anno_val['annotations'])
    #print(TEST_DICT)
    # for filename in tqdm(anno_val['annotations']):
    #     if filename['category_id'] == 0: label = [1, 0]
    #     elif filename['category_id'] == 1: label = [0, 1]

    for file in TEST_DICT:
        if sum(TEST_DICT[file]) == NUM_MODELS: animal = 1
        else: animal = 0
        id = file.split('.')[0]
        Output_List.append([id, animal])

    my_df = pd.DataFrame(Output_List)
    my_df.to_csv('iWildSubmissionOutput.csv', index=False, header=False)


#os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680

TEST_DIR = 'D:PythonData/iWildCam/LowRes/Test/'
#TEST_DIR = 'D:PythonData/iWildCam/LowRes/Val/'
#TEST_DIR = 'D:PythonData/iWildCam/LowRes/Train/'
LR = 1e-4
IMG_SIZE = 100
NUM_MODELS = 1

#model1 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\Keras\\VGG16_saved_models\\1xdif-50\\VGG16_50'
model1 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\iWildCam\\Keras\\VGG16_saved_models\\1xdif-100\\weights.27-13.88.hdf5'

TEST_LIST = os.listdir(TEST_DIR)
TEST_LIST = TEST_LIST[:50]
print(TEST_LIST)
TEST_DICT = {}

for file in tqdm(TEST_LIST):
    TEST_DICT[file] = []

# 0 = No Animal
# 1 = Animal

Create_Ouput_List(IMG_SIZE, model1)

Output_List = []

Build_Output_Val()




















