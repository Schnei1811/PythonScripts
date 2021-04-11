import numpy as np
import os
from tqdm import tqdm
import cv2
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

LR = 1e-4
IMG_SIZE = 50
MODEL_NAME = 'GrayscaleArylla-{}-{}.model'.format(LR, 'VGG16')

def create_data():
    data = []
    for img in tqdm(os.listdir('Data/joinedimages/blank/')):
        label = [1, 0]
        path = os.path.join('Data/joinedimages/blank/', img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        print(img[0])
        verticalmirrorimg = cv2.flip(img, 1)
        horizontalmirrorimg = cv2.flip(img, 0)
        data.append([np.array(img), np.array(label)])
        data.append([np.array(verticalmirrorimg), np.array(label)])
        data.append([np.array(horizontalmirrorimg), np.array(label)])
    for img in tqdm(os.listdir('Data/joinedimages/print/')):
        label = [0, 1]
        path = os.path.join('Data/joinedimages/print/', img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        verticalmirrorimg = cv2.flip(img, 1)
        horizontalmirrorimg = cv2.flip(img, 0)
        data.append([np.array(img), np.array(label)])
        data.append([np.array(verticalmirrorimg), np.array(label)])
        data.append([np.array(horizontalmirrorimg), np.array(label)])
    shuffle(data)
    np.save('Data.npy', data)
    return data

#Data = create_data()
data = np.load('Data.npy')


train, test = data[:-int(0.1 * len(data))], data[-int(0.1 * len(data)):]

print(len(train), len(test))


#Network architecture
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
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
network = fully_connected(network, 2, activation='softmax')     #2 = num classifications
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='targets')
model = tflearn.DNN(network, tensorboard_dir='log')


X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_y = [i[1] for i in test]

# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('Model Loaded!')
# else:
model.fit({'input': X}, {'targets': Y}, n_epoch=200, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)


fig = plt.figure()

for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1: str_label = 'Dog'
    else: str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()



