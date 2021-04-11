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

TRAIN_DIR = 'Files/Classifications/OctopusSegmentClassification/'
ROI_SIZE = 40
LR = 1e-3

MODEL_DIR = 'Files/Classifications/Models/'
MODEL_NAME = 'FullModel-{}-{}-{}.model'.format(ROI_SIZE, LR, '8conv-tree')

def label_img(dir):
    if dir == 'MantleTip': return [1, 0, 0, 0, 0]
    elif dir == 'MantleCenter': return [0, 1, 0, 0, 0]
    elif dir == 'Tentacle': return [0, 0, 1, 0, 0]
    elif dir == 'Body': return [0, 0, 0, 1, 0]
    else: return [0, 0, 0, 0, 1]

def create_train_data():
    training_data = []
    for dir in tqdm(os.listdir(TRAIN_DIR)):
        CURRENT_DIR = os.path.join(TRAIN_DIR, dir)
        for img in os.listdir('Files/Classifications/OctopusSegmentClassification/{}'.format(dir)):
            label = label_img(dir)
            path = os.path.join(CURRENT_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save(MODEL_DIR + MODEL_NAME + '.npy', training_data)
    return training_data

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
convnet = fully_connected(convnet, 5, activation='softmax')         #2 = NUM EXAMPLES
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

data = create_train_data()
#Data = np.load(MODEL_DIR + MODEL_NAME + '.npy')

shuffle(data)

train = data[:-500]
test = data[-500:]


X = np.array([i[0] for i in data]).reshape(-1, ROI_SIZE, ROI_SIZE, 1)
y = np.array([i[1] for i in data])

test_x = np.array([i[0] for i in test]).reshape(-1, ROI_SIZE, ROI_SIZE, 1)
test_y = np.array([i[1] for i in test])

if os.path.exists(MODEL_DIR + '{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_DIR + MODEL_NAME)
    print('Model Loaded!')
else:
    model.fit({'input': X}, {'targets': y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_DIR + MODEL_NAME)

#tensorboard --logdir=foo:C:\Users\Stefan\Dropbox\PythonScripts\PythonScripts\DeepLearningModelTutorials\CatsDogsKaggle\log

fig = plt.figure()

for num, data in enumerate(data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(ROI_SIZE, ROI_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0: str_label = 'MantleTip'
    elif np.argmax(model_out) == 1: str_label = 'MantleCenter'
    elif np.argmax(model_out) == 2: str_label = 'Tentacle'
    elif np.argmax(model_out) == 3: str_label = 'Body'
    else: str_label = 'NoOctopus'

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







