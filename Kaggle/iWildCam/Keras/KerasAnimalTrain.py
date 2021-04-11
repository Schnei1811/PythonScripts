from __future__ import print_function
import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import numpy as np


def VGG16():

    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=48,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)

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

    animal_output = Dense(2, activation='softmax', name='animal_out')(x)

    # Create model.
    model = Model(inputs=img_input, outputs=animal_output, name='vgg16')

    # if weights == 'imagenet':
    #     weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                             WEIGHTS_PATH_NO_TOP,
    #                             cache_subdir='models')
    #     model.load_weights(weights_path)
    return model



# os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680

# anno_dir = 'D:PythonData/iWildCam/Annotations/'
# TRAIN_DIR = 'D:PythonData/iWildCam/LowRes/Train/'
# VAL_DIR = 'D:PythonData/iWildCam/LowRes/Val/'
IMG_SIZE = 50
LR = 0.001

train_data = np.load('../data/{}_pixel_train_data.npy'.format(IMG_SIZE))
val_data = np.load('../data/{}_pixel_val_data.npy'.format(IMG_SIZE))

x_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train_animal = np.array([i[1] for i in train_data])

x_test = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test_animal = np.array([i[1] for i in val_data])

# x_train = x_train[:30]
# y_train_animal = y_train_animal[:30]
# y_train_location = y_train_location[:30]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = VGG16()

model.compile(loss=keras.losses.categorical_hinge,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001),
              metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False)  # randomly flip images

# input_generator = ImageDataGenerator(horizontal_flip=True,
#                                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
#                                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
#                                rotation_range=10)
#
# checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
#
# model.fit_generator(input_generator,
#                     steps_per_epoch=x_train.shape[0]//128,
#                     epochs=250,
#                     validation_data=(x_test, y_test_animal),
#                     callbacks=[#EarlyStopping(min_delta=0.001, patience=3),
#                      ModelCheckpoint('VGG16_saved_models/AnimalLocationMultitask/1xdif-50/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#                                      monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
#                                      mode='auto', period=5)])


model.fit(x_train, y_train_animal, epochs=10, batch_size=256, validation_data=(x_test, y_test_animal))

# Save model and weights

# Score trained model.
animal_scores = model.evaluate(x_test, y_test_animal, verbose=1)
print('Val loss:', animal_scores[0])
print('Val accuracy:', animal_scores[1])
























