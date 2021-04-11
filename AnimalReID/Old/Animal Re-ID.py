from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
import numpy as np
import tqdm
import cv2

def pairings(source):
    result = []
    for p1 in range(len(source)):
        for p2 in range(p1 + 1, len(source)):
            result.append([source[p1], source[p2]])
    return result


def VGG16():

    img_input1 = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    img_input2 = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

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

    x2 = Flatten(name='flattenb')(x2)

    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)

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
videoname = 'GO011675'

capture = cv2.VideoCapture('D:PythonData/Octopus/RawVideos/{}.mp4'.format(videoname))
maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(capture.get(3)), int(capture.get(4))
OctoDatabase = {}
OctoCoordinate = {}
octonum = 0
IMG_SIZE = 50
checkpoint_path = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\Kaggle\\Octopus2\\' \
         'SiameseNetwork\\VGG16_saved_models\\50\\weights.68-0.922.hdf5'

model = VGG16()
model.load_weights(checkpoint_path)

squares = np.loadtxt('D:PythonData/Octopus/OutputCSV/{}frcnnboxes.csv'.format(videoname), delimiter=',').astype(int)

print(len(squares))
print(maxnumframes)


iterframe = 0
i = 0

for framenum in tqdm.trange(0, maxnumframes):
    ret, frame = capture.read()
    try:
        while True:
            if squares[i, 4] == framenum:
                y1, x1, y2, x2 = squares[i,0], squares[i,1], squares[i,2], squares[i,3]
                if framenum == 0:
                    OctoDatabase[octonum] = frame[y1:y2, x1:x2]
                    OctoCoordinate[octonum] = [y1, y2, x1, x2]
                    octonum += 1

                elif framenum % 100 == 0:

                    if 980 > y2 > 880 or 100 < y1 < 200 or 100 < x1 < 200 or 1820 > x2 > 1720:

                        Different = True

                        for key in OctoDatabase:

                            img1 = cv2.cvtColor(OctoDatabase[key], cv2.COLOR_BGR2GRAY)
                            img2 = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

                            img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
                            img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))

                            img1 = np.array(img1).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
                            img2 = np.array(img2).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0

                            model_out = model.predict([img1, img2])[0]
                            if np.argmax(model_out) == 0:
                                Different = False
                                print('Predicted Same')
                            else:
                                print('Predicted Different')

                            # cv2.imshow('imgdata', OctoDatabase[key])
                            # cv2.imshow('imgtest', frame[y1:y2, x1:x2])
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                        if Different == True:
                            OctoDatabase[octonum] = frame[y1:y2, x1:x2]
                            octonum += 1
                            print(len(OctoDatabase))
                i += 1
            else:
                iterframe += 1
                break
    except:
        pass

    # cv2.imshow('original', frame)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27: break

print(OctoDatabase)

for key in OctoDatabase:
    cv2.imshow('imgdata', OctoDatabase[key])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


capture.release()
cv2.destroyAllWindows()
