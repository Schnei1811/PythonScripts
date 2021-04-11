from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
import numpy as np
import tqdm
import cv2
import os



def VGG19():

    img_input1 = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    img_input2 = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Block 1a
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv1', kernel_initializer='glorot_normal')(img_input1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv2', kernel_initializer='glorot_normal')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1a-1_pool')(x1)

    # Block 2a
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv1', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv2', kernel_initializer='glorot_normal')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block2a_pool')(x1)

    # Block 3a
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv1', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv2', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv3', kernel_initializer='glorot_normal')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block3a_pool')(x1)

    # Block 4a
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv1', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv2', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv3', kernel_initializer='glorot_normal')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block4a_pool')(x1)

    # Block 5a
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv1', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv2', kernel_initializer='glorot_normal')(x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv3', kernel_initializer='glorot_normal')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block5a_pool')(x1)

    x1 = Flatten(name='flattena')(x1)

    # Block 1b
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1b_conv1', kernel_initializer='glorot_normal')(img_input2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1b_conv2', kernel_initializer='glorot_normal')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1b_pool')(x2)

    # Block 2b
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2b_conv1', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2b_conv2', kernel_initializer='glorot_normal')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2b_pool')(x2)

    # Block 3a
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3b_conv1', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3b_conv2', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3b_conv3', kernel_initializer='glorot_normal')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block3b_pool')(x2)

    # Block 4a
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4b_conv1', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4b_conv2', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4b_conv3', kernel_initializer='glorot_normal')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block4b_pool')(x2)

    # Block 5a
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5b_conv1', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5b_conv2', kernel_initializer='glorot_normal')(x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5b_conv3', kernel_initializer='glorot_normal')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block5b_pool')(x2)

    x2 = Flatten(name='flattenb')(x2)

    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', name='fc1', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.6)(x)
    x = Dense(1024, activation='relu', name='fc2', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.6)(x)

    similarity_output = Dense(2, activation='softmax', name='similarity_out')(x)

    # Create model.
    model = Model(inputs=[img_input1, img_input2], outputs=[similarity_output], name='vgg16')

    return model








# videoname = '1chargingbehaviour'
#videoname = '2corrallingbehaviour'
#videoname = '3corrallingfromanotherangle'
#videoname = '4successfulescape'
#videoname = '5shortclip'
#videoname = 'GO010141'
#videoname = 'GO020142'
#videoname = 'GO040170'
#videoname = 'GO040141'
#videoname = 'test'


#Full Length Videos
# videoname = 'GO011675'
#
# capture = cv2.VideoCapture('D:PythonData/Octopus/RawVideos/{}.mp4'.format(videoname))
# maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
# width, height = int(capture.get(3)), int(capture.get(4))
# OctoDatabase = {}
# OctoCoordinate = {}
# octonum = 0


import glob

IMG_SIZE = 224
checkpoint_path_list = glob.glob('D:PythonData/Re-ID/Trained_Models/Octopus/VGG16/*/*.hdf5')

model = VGG19()


octodict = {}

# octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150127 Dive 2 Red Shell bed E facing W High'
# octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150128 Dive 1 Red Shell bed E facing W High'
# octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150128 Dive 2 BBEL2 Shell bed E facing W high'
# octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150128 Dive 3 Red Shell bed Shell bed E facing W high'
octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150128 Dive 4 BBEL2 Shell Bed E facing W high'
# octo_dir = 'D:PythonData\\Re-ID\\Octopus\\ExtractedOcto\\20150129 Dive 1 Red Shell bed E facing W high'

image_list = os.listdir(octo_dir)
for file in image_list:
    if file.endswith('.png'):
        img = cv2.imread(octo_dir + '\\{}'.format(file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        octodict[file] = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0




for file in os.listdir(octo_dir + '\\Questions'):

    maxsame = 0

    img2 = cv2.imread(octo_dir + '\\Questions\\{}'.format(file))
    height, width = img2.shape[0], img2.shape[1]
    print(height, width)
    img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
    img2 = np.array(img2).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0

    counter = 0
    octosum = 0
    maxcounter = 0

    for octo in octodict:
        img1 = octodict[octo]
        ensemble_out = np.array([0,0], dtype='float32')

        for checkpoint_path in checkpoint_path_list:
            model.load_weights(checkpoint_path)
            ensemble_out = ensemble_out + model.predict([img1, img2])

        ensemble_out /= 5

        print(file, octo, ensemble_out, height, width)

            # octosum += model_out[0][0]
            #
            # counter += 1
            # if model_out[0][0] > maxcounter: maxcounter = model_out[0][0]
            # if counter == 5:
            #     counter = 0
            #     print(file, octo, 'max: ', maxcounter)
            #     print(file, octo, 'avg: ', octosum/5)
            #     octosum = 0
            #     maxcounter = 0























