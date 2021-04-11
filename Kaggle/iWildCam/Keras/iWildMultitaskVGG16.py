from __future__ import print_function

import numpy as np
import os
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


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


os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680

IMG_SIZE = 100

print('Loading Training Data...')
train_data = np.load('../data/{}_pixel_train_multitask_data.npy'.format(IMG_SIZE))
print('Loading Val Data...')
val_data = np.load('../data/{}_pixel_val_multitask_data.npy'.format(IMG_SIZE))

# train_data = train_data[:300]
# val_data = val_data[:300]
print('Processing Data...')
x_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train_animal = np.array([i[1] for i in train_data])
y_train_location = to_categorical(np.array([i[2] for i in train_data]))

x_test = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test_animal = np.array([i[1] for i in val_data])
y_test_location = to_categorical(np.array([np.random.randint(0, 65) for i in val_data]))
#y_test_location = to_categorical(np.array([i[2] for i in val_data]))

print(y_train_location.shape)
print(y_test_location.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model_name = 'VGG16_{}'.format(IMG_SIZE)

model = VGG16()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),
              loss={'first_animal_out':'categorical_crossentropy', 'first_location_out':'categorical_crossentropy',
                    'second_animal_out':'categorical_crossentropy', 'second_location_out':'categorical_crossentropy'},
              loss_weights={'first_animal_out':1, 'first_location_out':-0.1,
                            'second_animal_out':1, 'second_location_out':1},
              metrics={'first_animal_out':'accuracy', 'first_location_out':'accuracy',
                       'second_animal_out': 'accuracy', 'second_location_out': 'accuracy'})


generator = ImageDataGenerator(horizontal_flip=True,
                               width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                               rotation_range=10)

def format_gen_outputs(gen1, gen2):
    x = gen1[0]
    y1 = gen1[1]
    y2 = gen2[1]
    return x, [y2, y1, y2, y1]

input_generator = map(format_gen_outputs, generator.flow(x_train, y_train_animal),  generator.flow(x_train, y_train_location))

# input_imgen = ImageDataGenerator(
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


model.fit_generator(input_generator,
                    steps_per_epoch=x_train.shape[0]//128,
                    epochs=250,
                    validation_data=(x_test, [y_test_location, y_test_animal, y_test_location, y_test_animal]),
                    callbacks=[#EarlyStopping(min_delta=0.001, patience=3),
                     ModelCheckpoint('VGG16_saved_models/1xdif-100/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                     verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=0)])

#Train the model
# model.fit(x_train, [y_train_location, y_train_animal, y_train_location, y_train_animal],
#           batch_size=128,
#           shuffle=True,
#           epochs=150,
#           validation_data=(x_test, [y_test_location, y_test_animal, y_test_location, y_test_animal]),
#           callbacks=[#EarlyStopping(min_delta=0.001, patience=3),
#                      ModelCheckpoint('VGG16_saved_models/1xdif-100/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
#                                      verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)])

#Evaluate the model
scores = model.evaluate(x_test, [y_test_location, y_test_animal, y_test_location, y_test_animal])

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print('Loss: %.3f' % scores[2])
print('Accuracy: %.3f' % scores[3])
print('Loss: %.3f' % scores[4])
print('Accuracy: %.3f' % scores[5])
print('Loss: %.3f' % scores[6])
print('Accuracy: %.3f' % scores[7])

if not os.path.isdir('VGG16_saved_models'):
    os.makedirs('VGG16_saved_models')
model_path = os.path.join('VGG16_saved_models', model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)