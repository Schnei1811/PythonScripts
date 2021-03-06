import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1000)

videoname = '1chargingbehaviour'

TRAIN_DIR = 'Files/Classifications/CreatureClassification/'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = '{}-{}.model'.format(LR, '8conv-basic')

def label_img(dir):
    if dir == 'Diver': return [1, 0, 0, 0]
    elif dir == 'Fish': return [0, 1, 0, 0]
    elif dir == 'Kelp': return [0, 0, 1, 0]
    elif dir == 'Octopus': return [0, 0, 0, 1]

def create_train_data():
    training_data = []
    for dir in tqdm(os.listdir(TRAIN_DIR)):
        CURRENT_DIR = os.path.join(TRAIN_DIR, dir)
        for img in os.listdir('Files/Classifications/CreatureClassification/{}'.format(dir)):
            label = label_img(dir)
            path = os.path.join(CURRENT_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('ConvNetData.npy', training_data)
    return training_data

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, name='targets', optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#Data = create_train_data()
data = np.load('ConvNetData.npy')

shuffle(data)

train = data[:-500]
test = data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')
else:
    model.fit({'input': X}, {'targets': y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

#tensorboard --logdir=foo:C:\Users\Stefan\Dropbox\PythonScripts\PythonScripts\opencv\Octopus\log

fig = plt.figure()

for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0: str_label = 'Diver'
    elif np.argmax(model_out) == 1: str_label = 'Fish'
    elif np.argmax(model_out) == 2: str_label = 'Kelp'
    else: str_label = 'Octopus'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

# with open('submission-file.csv', 'w') as f: f.write('id,label\n')
# with open('submission-file.csv', 'a') as f:
#     for Data in tqdm(test_data):
#         img_num = Data[1]
#         img_data = Data[0]
#         orig = img_data
#         Data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#         model_out = model.predict([Data])[0]
#         f.write('{},{}\n').format(img_num, model_out[1])







