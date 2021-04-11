import os
import numpy as np
import csv
import cv2
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from imgaug import augmenters as iaa
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam


init_aug_seq = iaa.SomeOf((1, 10), [                                         # Random number between 0, 3
    iaa.Fliplr(0.5),                                                         # Horizontal flips                     0.01
    iaa.WithChannels(0, iaa.Add((-30, 30)), 0, iaa.Affine(rotate=(-25, 25))),   # Random channel increase and rotation 0.03
    iaa.WithChannels(1, iaa.Add((-30, 30)), 1, iaa.Affine(rotate=(-25, 25))),
    iaa.WithChannels(2, iaa.Add((-30, 30)), 2, iaa.Affine(rotate=(-25, 25))),
    iaa.Add((-15, 15)),                                                      # Overall Brightness                   0.04
    iaa.Multiply((0.90, 1.10), per_channel=0.2),                             # Brightness multiplier per channel    0.05
    iaa.Sharpen(alpha=(0.1, 0.75), lightness=(0.85, 1.15)),                   # Sharpness                            0.05
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',           # Random HSV increase                  0.09
                       children=iaa.WithChannels(0, iaa.Add((-30, 30)))),
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                       children=iaa.WithChannels(1, iaa.Add((-30, 30)))),
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                       children=iaa.WithChannels(2, iaa.Add((-30, 30)))),
    iaa.AddElementwise((-10, 10)),                                           # Per pixel addition                   0.11
    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),               # Add large black squares              0.13
    iaa.GaussianBlur(sigma=(0.1, 1.0)),                                      # GaussianBlur                         0.14
    iaa.Grayscale(alpha=(0.1, 1.0)),                                         # Random Grayscale conversion          0.17
    iaa.Dropout(p=(0, 0.1), per_channel=0.2),                                # Add small black squares              0.17
    iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),     # Add Gaussian per pixel noise         0.26
    iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},                 # Affine: Scale/zoom,                  0.46
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},          #         Translate/move
        rotate=(-45, 45), shear=(-3, 3)),                                    #         Rotate and Shear
    iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.25),                   # Distort image by rearranging pixels  0.70
    iaa.ContrastNormalization((0.75, 1.5)),                                  # Contrast Normalization               0.95
    iaa.PiecewiseAffine(scale=(0, 0.05)),                                    # Distort Image similar water droplet  1.76
], random_order=True)


train_aug_seq = iaa.SomeOf((0, 3), [                                         # Random selection
    iaa.Fliplr(0.5),                                                         # Horizontal flips                     0.01
    iaa.WithChannels(0, iaa.Add((-30, 30)), 0, iaa.Affine(rotate=(-25, 25))),   # Random channel increase and rotation 0.03
    iaa.WithChannels(1, iaa.Add((-30, 30)), 1, iaa.Affine(rotate=(-25, 25))),
    iaa.WithChannels(2, iaa.Add((-30, 30)), 2, iaa.Affine(rotate=(-25, 25))),
    iaa.Add((-15, 15)),                                                      # Overall Brightness                   0.04
    iaa.Multiply((0.90, 1.10), per_channel=0.2),                             # Brightness multiplier per channel    0.05
    iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.85, 1.15)),                   # Sharpness                            0.05
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',           # Random HSV increase                  0.09
                       children=iaa.WithChannels(0, iaa.Add((-30, 30)))),
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                       children=iaa.WithChannels(1, iaa.Add((-30, 30)))),
    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                       children=iaa.WithChannels(2, iaa.Add((-30, 30)))),
    iaa.AddElementwise((-10, 10)),                                           # Per pixel addition                   0.11
    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),               # Add large black squares              0.13
    iaa.GaussianBlur(sigma=(0.1, 0.5)),                                      # GaussianBlur                         0.14
    iaa.Grayscale(alpha=(0.1, 1.0)),                                         # Random Grayscale conversion          0.17
    iaa.Dropout(p=(0, 0.1), per_channel=0.2),                                # Add small black squares              0.17
    # iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),   # Add Gaussian per pixel noise         0.26
    # iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},                   # Affine: Scale/zoom,                  0.46
    #     translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},        #         Translate/move
    #     rotate=(-45, 45), shear=(-8, 8)),                                  #         Rotate and Shear
    # iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25)                  # Distort image by rearranging pixels  0.70
    # iaa.ContrastNormalization((0.75, 1.5)),                                # Contrast Normalization               0.95
    # iaa.PiecewiseAffine(scale=(0, 0.05)),                                  # Distort Image similar water droplet  1.76
], random_order=True)



def VGG19():
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2a_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3a_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4a_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5a_pool')(x)

    x = Flatten(name='flattena')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.1))(x)

    output = Dense(Num_Classes, activation='softmax', name='similarity_out')(x)

    return Model(inputs=img_input, outputs=output, name='vgg19')



def create_data(data_list, DATA_TYPE):

    print('Number of Training Examples for each Classification: ')
    for object in Num_Images_Dict:
        print(object, ':', Num_Images_Dict[object])

    data, class_iter, k = [], 0, 0
    for object in tqdm(data_list):
        classification = class_list[class_iter]
        for i in range(Desired_Num_Images_Per_Classification):
            # if i > maxbreak: break
            if i < Num_Images_Dict[classification]:
                img = cv2.resize(cv2.imread(IMAGE_TASK_DIR + object), (IMG_SIZE, IMG_SIZE))
                zerolist = [0] * Num_Classes
                zerolist[class_list.index(classification)] += 1
                print('early', zerolist, i)

                data.append([np.array(img), zerolist])
            else:
                if DATA_TYPE == 'Train':
                    k = i
                    while k < Desired_Num_Images_Per_Classification:
                        imglist = os.listdir(IMAGE_TASK_DIR + classification)
                        rand_int = random.randint(0, Num_Images_Dict[classification])
                        print(IMAGE_TASK_DIR + classification + '/' + imglist[rand_int])
                        img = cv2.resize(cv2.imread(IMAGE_TASK_DIR + classification + '/' + imglist[rand_int]),
                                         (IMG_SIZE, IMG_SIZE))
                        img = init_aug_seq.augment_image(img)
                        zerolist = [0] * Num_Classes
                        zerolist[class_list.index(classification)] += 1
                        print('later', zerolist, k)
                        data.append([np.array(img), zerolist])
                        k += 1




                #zerolist[classnum] += 1

        # elif DATA_TYPE == 'Test':
        #     for image in os.listdir(DATA_DIR + animal):
        #         if i > maxbreak: break
        #         img = cv2.resize(cv2.imread(DATA_DIR + animal + '/' + image), (IMG_SIZE, IMG_SIZE))
        #         zerolist = [0] * Num_Classes
        #         zerolist[classnum] += 1
        #         data.append([np.array(img), zerolist])
        #         i += 1
    return data


def plot(data1, data2, IMG_SIZE, metric):
    plt.plot(data1)
    plt.plot(data2)
    plt.title('History of Multiclass Animal Model {} During Training'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}_{}_saved_models/{}/{}.png'.format(Network, srttm, IMG_SIZE, metric))
    plt.clf()


def generator(images, labels, batch_size):
    batch_features, batch_labels = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3)), np.zeros((batch_size, Num_Classes))
    while True:
        for i in range(batch_size):
            index = np.random.randint(0, len(images) - 1)
            random_augmented_image, random_augmented_label = images[index], labels[index]
            random_augmented_image = train_aug_seq.augment_image(random_augmented_image)
            batch_features[i], batch_labels[i] = random_augmented_image, random_augmented_label
        yield batch_features, batch_labels

def Train_Test_Split():
    print('Creating Train/Test Split...')
    #ans = input('Would you like to check if all images are valid? (Recommended but takes a long time): (y/n)')
    ans = 'n'
    train_list, test_list = [], []
    for classification in tqdm(os.listdir(IMAGE_TASK_DIR)):
        i = 0
        imagedir = os.listdir(IMAGE_TASK_DIR + classification)
        random.shuffle(imagedir)
        for image in imagedir:
            if ans == 'y':
                try: cv2.resize(cv2.imread(IMAGE_TASK_DIR + classification + '/' + image), (100, 100))
                except:
                    print('ERROR LOADING IMAGE!: ', IMAGE_TASK_DIR + classification + '/' + image)
                    if not os.path.exists(PROJECT_DIR + '{}_Error_Images/{}/'.format(imagetask, classification)):
                        os.makedirs(PROJECT_DIR + '{}_Error_Images/{}/'.format(imagetask, classification))
                    os.rename(IMAGE_TASK_DIR + classification + '/' + image,
                              PROJECT_DIR + '/{}_Error_Images/{}/{}'.format(imagetask, classification, image))
            if i < 5: test_list.append(classification + '/' + image)
            elif random.random() <= 0.05: test_list.append(classification + '/' + image)
            else: train_list.append(classification + '/' + image)
            i += 1

    with open(PROJECT_DIR + '{}_Train.csv'.format(imagetask), 'w', newline='') as f:
        writer = csv.writer(f)
        for item in train_list: writer.writerow([item])
    with open(PROJECT_DIR + '{}_Test.csv'.format(imagetask), 'w', newline='') as f:
        writer = csv.writer(f)
        for item in test_list: writer.writerow([item])



##TO DO
# Print Out Classification List
# Classification Image Threshold -> Add Other Category


#project = 'Peru/'
project, imagetask = 'ParksCanada', 'SpeciesID'


PROJECT_DIR = 'D:PythonData/{}/'.format(project)
IMAGE_TASK_DIR = 'D:PythonData/{}/{}/'.format(project, imagetask)

#if os.path.isfile(PROJECT_DIR + '{}_Test.csv'.format(imagetask)) == False: Train_Test_Split()

Train_Test_Split()
train_list, test_list = [], []

with open(PROJECT_DIR + '{}_Train.csv'.format(imagetask), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader: train_list.append(row[0])

with open(PROJECT_DIR + '{}_Test.csv'.format(imagetask), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader: test_list.append(row[0])

Num_Images_Dict = {}
for object in train_list:
    if object.split('/')[0] not in Num_Images_Dict: Num_Images_Dict[object.split('/')[0]] = 1
    else: Num_Images_Dict[object.split('/')[0]] += 1

max_class_num = Num_Images_Dict[max(Num_Images_Dict, key=Num_Images_Dict.get)]

IMG_SIZE = 100
batch_size = 128
epochs = 500
maxbreak = 10
class_list = os.listdir(IMAGE_TASK_DIR)
Num_Classes = len(class_list)
#Desired_Num_Images_Per_Classification = max_class_num
Desired_Num_Images_Per_Classification = 50

# Network = 'AlexNet'
# Network = 'SqueezeNet'
# Network = 'VGG16'
Network = 'VGG19'

train_data = create_data(train_list, DATA_TYPE='Train')
test_data = create_data(test_list, DATA_TYPE='Test')

srttm = time.strftime("%Y%m%d-%H%M%S")

print('Train Size: {}'.format(len(train_data)))
print('Test Size: {}'.format(len(test_data)))

x_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0
y_train = np.array([i[1] for i in train_data])
del train_data

x_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0
y_test = np.array([i[1] for i in test_data])
del test_data


# model = VGG19()
model = load_model('VGG19_20190212-175011_saved_models/100/weights.42-0.90.hdf5')


#check_dir_exists('{}_{}_saved_models/{}'.format(Network, srttm, IMG_SIZE))


model.summary()
model.compile(optimizer=Adam(lr=0.001, decay=1e-6),
              loss={'similarity_out': 'categorical_crossentropy'},
              loss_weights={'similarity_out': 1},
              metrics={'similarity_out': 'accuracy'})


# Save plot of train & test image distribution
# Write Text File of Class List

# TensorBoard(log_dir='{}_{}TB_Logger./log'.format(Network, starttime))
csv_log = CSVLogger('{}_{}_saved_models/{}/{}_{}.csv'.format(Network, srttm, IMG_SIZE, IMG_SIZE, srttm), separator=',')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit_generator(generator(x_train, y_train, batch_size),
                              validation_data=(x_test, y_test),
                              shuffle=True,
                              steps_per_epoch=x_train.shape[0] / batch_size,
                              epochs=epochs,
                              callbacks=[# EarlyStopping(min_delta=0.001, patience=3),
                                  reduce_lr, csv_log,
                                  ModelCheckpoint('%s_%s_saved_models/%s/weights.{epoch:02d}-'
                                                 '{val_acc:.2f}.hdf5' % (Network, srttm, IMG_SIZE),
                                                  monitor='val_acc', verbose=0, save_best_only=True,
                                                  save_weights_only=False, mode='auto', period=1)])

plot(history.history['loss'], history.history['val_loss'], IMG_SIZE, 'Loss')
plot(history.history['acc'], history.history['val_acc'], IMG_SIZE, 'Accuracy')

































