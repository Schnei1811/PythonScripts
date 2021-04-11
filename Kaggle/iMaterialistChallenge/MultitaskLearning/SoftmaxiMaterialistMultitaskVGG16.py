from __future__ import print_function

import numpy as np
import os
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator



def VGG16():

    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
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
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x1 = Dense(64, activation='relu', name='x1fc1')(x)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(64, activation='relu', name='x1fc2')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(2, activation='softmax', name='x1_out')(x1)

    x2 = Dense(64, activation='relu', name='x2fc1')(x)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(64, activation='relu', name='x2fc2')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(2, activation='softmax', name='x2_out')(x2)

    x3 = Dense(64, activation='relu', name='x3fc1')(x)
    x3 = Dropout(0.2)(x3)
    x3 = Dense(64, activation='relu', name='x3fc2')(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Dense(2, activation='softmax', name='x3_out')(x3)

    x4 = Dense(64, activation='relu', name='x4fc1')(x)
    x4 = Dropout(0.2)(x4)
    x4 = Dense(64, activation='relu', name='x4fc2')(x4)
    x4 = Dropout(0.2)(x4)
    x4 = Dense(2, activation='softmax', name='x4_out')(x4)

    x5 = Dense(64, activation='relu', name='x5fc1')(x)
    x5 = Dropout(0.2)(x5)
    x5 = Dense(64, activation='relu', name='x5fc2')(x5)
    x5 = Dropout(0.2)(x5)
    x5 = Dense(2, activation='softmax', name='x5_out')(x5)

    x6 = Dense(64, activation='relu', name='x6fc1')(x)
    x6 = Dropout(0.2)(x6)
    x6 = Dense(64, activation='relu', name='x6fc2')(x6)
    x6 = Dropout(0.2)(x6)
    x6 = Dense(2, activation='softmax', name='x6_out')(x6)

    x7 = Dense(64, activation='relu', name='x7fc1')(x)
    x7 = Dropout(0.2)(x7)
    x7 = Dense(64, activation='relu', name='x7fc2')(x7)
    x7 = Dropout(0.2)(x7)
    x7 = Dense(2, activation='softmax', name='x7_out')(x7)

    x8 = Dense(64, activation='relu', name='x8fc1')(x)
    x8 = Dropout(0.2)(x8)
    x8 = Dense(64, activation='relu', name='x8fc2')(x8)
    x8 = Dropout(0.2)(x8)
    x8 = Dense(2, activation='softmax', name='x8_out')(x8)

    x9 = Dense(64, activation='relu', name='x9fc1')(x)
    x9 = Dropout(0.2)(x9)
    x9 = Dense(64, activation='relu', name='x9fc2')(x9)
    x9 = Dropout(0.2)(x9)
    x9 = Dense(2, activation='softmax', name='x9_out')(x9)

    x10 = Dense(64, activation='relu', name='x10fc1')(x)
    x10 = Dropout(0.2)(x10)
    x10 = Dense(64, activation='relu', name='x10fc2')(x10)
    x10 = Dropout(0.2)(x10)
    x10 = Dense(2, activation='softmax', name='x10_out')(x10)

    x11 = Dense(64, activation='relu', name='x11fc1')(x)
    x11 = Dropout(0.2)(x11)
    x11 = Dense(64, activation='relu', name='x11fc2')(x11)
    x11 = Dropout(0.2)(x11)
    x11 = Dense(2, activation='softmax', name='x11_out')(x11)

    x12 = Dense(64, activation='relu', name='x12fc1')(x)
    x12 = Dropout(0.2)(x12)
    x12 = Dense(64, activation='relu', name='x12fc2')(x12)
    x12 = Dropout(0.2)(x12)
    x12 = Dense(2, activation='softmax', name='x12_out')(x12)

    x13 = Dense(64, activation='relu', name='x13fc1')(x)
    x13 = Dropout(0.2)(x13)
    x13 = Dense(64, activation='relu', name='x13fc2')(x13)
    x13 = Dropout(0.2)(x13)
    x13 = Dense(2, activation='softmax', name='x13_out')(x13)

    x14 = Dense(64, activation='relu', name='x14fc1')(x)
    x14 = Dropout(0.2)(x14)
    x14 = Dense(64, activation='relu', name='x14fc2')(x14)
    x14 = Dropout(0.2)(x14)
    x14 = Dense(2, activation='softmax', name='x14_out')(x14)

    x15 = Dense(64, activation='relu', name='x15fc1')(x)
    x15 = Dropout(0.2)(x15)
    x15 = Dense(64, activation='relu', name='x15fc2')(x15)
    x15 = Dropout(0.2)(x15)
    x15 = Dense(2, activation='softmax', name='x15_out')(x15)

    x16 = Dense(64, activation='relu', name='x16fc1')(x)
    x16 = Dropout(0.2)(x16)
    x16 = Dense(64, activation='relu', name='x16fc2')(x16)
    x16 = Dropout(0.2)(x16)
    x16 = Dense(2, activation='softmax', name='x16_out')(x16)

    x17 = Dense(64, activation='relu', name='x17fc1')(x)
    x17 = Dropout(0.2)(x17)
    x17 = Dense(64, activation='relu', name='x17fc2')(x17)
    x17 = Dropout(0.2)(x17)
    x17 = Dense(2, activation='softmax', name='x17_out')(x17)

    x18 = Dense(64, activation='relu', name='x18fc1')(x)
    x18 = Dropout(0.2)(x18)
    x18 = Dense(64, activation='relu', name='x18fc2')(x18)
    x18 = Dropout(0.2)(x18)
    x18 = Dense(2, activation='softmax', name='x18_out')(x18)

    x19 = Dense(64, activation='relu', name='x19fc1')(x)
    x19 = Dropout(0.2)(x19)
    x19 = Dense(64, activation='relu', name='x19fc2')(x19)
    x19 = Dropout(0.2)(x19)
    x19 = Dense(2, activation='softmax', name='x19_out')(x19)

    x20 = Dense(64, activation='relu', name='x20fc1')(x)
    x20 = Dropout(0.2)(x20)
    x20 = Dense(64, activation='relu', name='x20fc2')(x20)
    x20 = Dropout(0.2)(x20)
    x20 = Dense(2, activation='softmax', name='x20_out')(x20)

    x21 = Dense(64, activation='relu', name='x21fc1')(x)
    x21 = Dropout(0.2)(x21)
    x21 = Dense(64, activation='relu', name='x21fc2')(x21)
    x21 = Dropout(0.2)(x21)
    x21 = Dense(2, activation='softmax', name='x21_out')(x21)

    x22 = Dense(64, activation='relu', name='x22fc1')(x)
    x22 = Dropout(0.2)(x22)
    x22 = Dense(64, activation='relu', name='x22fc2')(x22)
    x22 = Dropout(0.2)(x22)
    x22 = Dense(2, activation='softmax', name='x22_out')(x22)

    x23 = Dense(64, activation='relu', name='x23fc1')(x)
    x23 = Dropout(0.2)(x23)
    x23 = Dense(64, activation='relu', name='x23fc2')(x23)
    x23 = Dropout(0.2)(x23)
    x23 = Dense(2, activation='softmax', name='x23_out')(x23)

    x24 = Dense(64, activation='relu', name='x24fc1')(x)
    x24 = Dropout(0.2)(x24)
    x24 = Dense(64, activation='relu', name='x24fc2')(x24)
    x24 = Dropout(0.2)(x24)
    x24 = Dense(2, activation='softmax', name='x24_out')(x24)

    x25 = Dense(64, activation='relu', name='x25fc1')(x)
    x25 = Dropout(0.2)(x25)
    x25 = Dense(64, activation='relu', name='x25fc2')(x25)
    x25 = Dropout(0.2)(x25)
    x25 = Dense(2, activation='softmax', name='x25_out')(x25)

    x26 = Dense(64, activation='relu', name='x26fc1')(x)
    x26 = Dropout(0.2)(x26)
    x26 = Dense(64, activation='relu', name='x26fc2')(x26)
    x26 = Dropout(0.2)(x26)
    x26 = Dense(2, activation='softmax', name='x26_out')(x26)

    x27 = Dense(64, activation='relu', name='x27fc1')(x)
    x27 = Dropout(0.2)(x27)
    x27 = Dense(64, activation='relu', name='x27fc2')(x27)
    x27 = Dropout(0.2)(x27)
    x27 = Dense(2, activation='softmax', name='x27_out')(x27)

    x28 = Dense(64, activation='relu', name='x28fc1')(x)
    x28 = Dropout(0.2)(x28)
    x28 = Dense(64, activation='relu', name='x28fc2')(x28)
    x28 = Dropout(0.2)(x28)
    x28 = Dense(2, activation='softmax', name='x28_out')(x28)

    x29 = Dense(64, activation='relu', name='x29fc1')(x)
    x29 = Dropout(0.2)(x29)
    x29 = Dense(64, activation='relu', name='x29fc2')(x29)
    x29 = Dropout(0.2)(x29)
    x29 = Dense(2, activation='softmax', name='x29_out')(x29)

    x30 = Dense(64, activation='relu', name='x30fc1')(x)
    x30 = Dropout(0.2)(x30)
    x30 = Dense(64, activation='relu', name='x30fc2')(x30)
    x30 = Dropout(0.2)(x30)
    x30 = Dense(2, activation='softmax', name='x30_out')(x30)

    x31 = Dense(64, activation='relu', name='x31fc1')(x)
    x31 = Dropout(0.2)(x31)
    x31 = Dense(64, activation='relu', name='x31fc2')(x31)
    x31 = Dropout(0.2)(x31)
    x31 = Dense(2, activation='softmax', name='x31_out')(x31)

    x32 = Dense(64, activation='relu', name='x32fc1')(x)
    x32 = Dropout(0.2)(x32)
    x32 = Dense(64, activation='relu', name='x32fc2')(x32)
    x32 = Dropout(0.2)(x32)
    x32 = Dense(2, activation='softmax', name='x32_out')(x32)

    x33 = Dense(64, activation='relu', name='x33fc1')(x)
    x33 = Dropout(0.2)(x33)
    x33 = Dense(64, activation='relu', name='x33fc2')(x33)
    x33 = Dropout(0.2)(x33)
    x33 = Dense(2, activation='softmax', name='x33_out')(x33)

    x34 = Dense(64, activation='relu', name='x34fc1')(x)
    x34 = Dropout(0.2)(x34)
    x34 = Dense(64, activation='relu', name='x34fc2')(x34)
    x34 = Dropout(0.2)(x34)
    x34 = Dense(2, activation='softmax', name='x34_out')(x34)

    x35 = Dense(64, activation='relu', name='x35fc1')(x)
    x35 = Dropout(0.2)(x35)
    x35 = Dense(64, activation='relu', name='x35fc2')(x35)
    x35 = Dropout(0.2)(x35)
    x35 = Dense(2, activation='softmax', name='x35_out')(x35)

    x36 = Dense(64, activation='relu', name='x36fc1')(x)
    x36 = Dropout(0.2)(x36)
    x36 = Dense(64, activation='relu', name='x36fc2')(x36)
    x36 = Dropout(0.2)(x36)
    x36 = Dense(2, activation='softmax', name='x36_out')(x36)

    x37 = Dense(64, activation='relu', name='x37fc1')(x)
    x37 = Dropout(0.2)(x37)
    x37 = Dense(64, activation='relu', name='x37fc2')(x37)
    x37 = Dropout(0.2)(x37)
    x37 = Dense(2, activation='softmax', name='x37_out')(x37)

    x38 = Dense(64, activation='relu', name='x38fc1')(x)
    x38 = Dropout(0.2)(x38)
    x38 = Dense(64, activation='relu', name='x38fc2')(x38)
    x38 = Dropout(0.2)(x38)
    x38 = Dense(2, activation='softmax', name='x38_out')(x38)

    x39 = Dense(64, activation='relu', name='x39fc1')(x)
    x39 = Dropout(0.2)(x39)
    x39 = Dense(64, activation='relu', name='x39fc2')(x39)
    x39 = Dropout(0.2)(x39)
    x39 = Dense(2, activation='softmax', name='x39_out')(x39)

    x40 = Dense(64, activation='relu', name='x40fc1')(x)
    x40 = Dropout(0.2)(x40)
    x40 = Dense(64, activation='relu', name='x40fc2')(x40)
    x40 = Dropout(0.2)(x40)
    x40 = Dense(2, activation='softmax', name='x40_out')(x40)

    x41 = Dense(64, activation='relu', name='x41fc1')(x)
    x41 = Dropout(0.2)(x41)
    x41 = Dense(64, activation='relu', name='x41fc2')(x41)
    x41 = Dropout(0.2)(x41)
    x41 = Dense(2, activation='softmax', name='x41_out')(x41)

    x42 = Dense(64, activation='relu', name='x42fc1')(x)
    x42 = Dropout(0.2)(x42)
    x42 = Dense(64, activation='relu', name='x42fc2')(x42)
    x42 = Dropout(0.2)(x42)
    x42 = Dense(2, activation='softmax', name='x42_out')(x42)

    x43 = Dense(64, activation='relu', name='x43fc1')(x)
    x43 = Dropout(0.2)(x43)
    x43 = Dense(64, activation='relu', name='x43fc2')(x43)
    x43 = Dropout(0.2)(x43)
    x43 = Dense(2, activation='softmax', name='x43_out')(x43)

    x44 = Dense(64, activation='relu', name='x44fc1')(x)
    x44 = Dropout(0.2)(x44)
    x44 = Dense(64, activation='relu', name='x44fc2')(x44)
    x44 = Dropout(0.2)(x44)
    x44 = Dense(2, activation='softmax', name='x44_out')(x44)

    x45 = Dense(64, activation='relu', name='x45fc1')(x)
    x45 = Dropout(0.2)(x45)
    x45 = Dense(64, activation='relu', name='x45fc2')(x45)
    x45 = Dropout(0.2)(x45)
    x45 = Dense(2, activation='softmax', name='x45_out')(x45)

    x46 = Dense(64, activation='relu', name='x46fc1')(x)
    x46 = Dropout(0.2)(x46)
    x46 = Dense(64, activation='relu', name='x46fc2')(x46)
    x46 = Dropout(0.2)(x46)
    x46 = Dense(2, activation='softmax', name='x46_out')(x46)

    x47 = Dense(64, activation='relu', name='x47fc1')(x)
    x47 = Dropout(0.2)(x47)
    x47 = Dense(64, activation='relu', name='x47fc2')(x47)
    x47 = Dropout(0.2)(x47)
    x47 = Dense(2, activation='softmax', name='x47_out')(x47)

    x48 = Dense(64, activation='relu', name='x48fc1')(x)
    x48 = Dropout(0.2)(x48)
    x48 = Dense(64, activation='relu', name='x48fc2')(x48)
    x48 = Dropout(0.2)(x48)
    x48 = Dense(2, activation='softmax', name='x48_out')(x48)

    x49 = Dense(64, activation='relu', name='x49fc1')(x)
    x49 = Dropout(0.2)(x49)
    x49 = Dense(64, activation='relu', name='x49fc2')(x49)
    x49 = Dropout(0.2)(x49)
    x49 = Dense(2, activation='softmax', name='x49_out')(x49)

    x50 = Dense(64, activation='relu', name='x50fc1')(x)
    x50 = Dropout(0.2)(x50)
    x50 = Dense(64, activation='relu', name='x50fc2')(x50)
    x50 = Dropout(0.2)(x50)
    x50 = Dense(2, activation='softmax', name='x50_out')(x50)

    x51 = Dense(64, activation='relu', name='x51fc1')(x)
    x51 = Dropout(0.2)(x51)
    x51 = Dense(64, activation='relu', name='x51fc2')(x51)
    x51 = Dropout(0.2)(x51)
    x51 = Dense(2, activation='softmax', name='x51_out')(x51)

    x52 = Dense(64, activation='relu', name='x52fc1')(x)
    x52 = Dropout(0.2)(x52)
    x52 = Dense(64, activation='relu', name='x52fc2')(x52)
    x52 = Dropout(0.2)(x52)
    x52 = Dense(2, activation='softmax', name='x52_out')(x52)

    x53 = Dense(64, activation='relu', name='x53fc1')(x)
    x53 = Dropout(0.2)(x53)
    x53 = Dense(64, activation='relu', name='x53fc2')(x53)
    x53 = Dropout(0.2)(x53)
    x53 = Dense(2, activation='softmax', name='x53_out')(x53)

    x54 = Dense(64, activation='relu', name='x54fc1')(x)
    x54 = Dropout(0.2)(x54)
    x54 = Dense(64, activation='relu', name='x54fc2')(x54)
    x54 = Dropout(0.2)(x54)
    x54 = Dense(2, activation='softmax', name='x54_out')(x54)

    x55 = Dense(64, activation='relu', name='x55fc1')(x)
    x55 = Dropout(0.2)(x55)
    x55 = Dense(64, activation='relu', name='x55fc2')(x55)
    x55 = Dropout(0.2)(x55)
    x55 = Dense(2, activation='softmax', name='x55_out')(x55)

    x56 = Dense(64, activation='relu', name='x56fc1')(x)
    x56 = Dropout(0.2)(x56)
    x56 = Dense(64, activation='relu', name='x56fc2')(x56)
    x56 = Dropout(0.2)(x56)
    x56 = Dense(2, activation='softmax', name='x56_out')(x56)

    x57 = Dense(64, activation='relu', name='x57fc1')(x)
    x57 = Dropout(0.2)(x57)
    x57 = Dense(64, activation='relu', name='x57fc2')(x57)
    x57 = Dropout(0.2)(x57)
    x57 = Dense(2, activation='softmax', name='x57_out')(x57)

    x58 = Dense(64, activation='relu', name='x58fc1')(x)
    x58 = Dropout(0.2)(x58)
    x58 = Dense(64, activation='relu', name='x58fc2')(x58)
    x58 = Dropout(0.2)(x58)
    x58 = Dense(2, activation='softmax', name='x58_out')(x58)

    x59 = Dense(64, activation='relu', name='x59fc1')(x)
    x59 = Dropout(0.2)(x59)
    x59 = Dense(64, activation='relu', name='x59fc2')(x59)
    x59 = Dropout(0.2)(x59)
    x59 = Dense(2, activation='softmax', name='x59_out')(x59)

    x60 = Dense(64, activation='relu', name='x60fc1')(x)
    x60 = Dropout(0.2)(x60)
    x60 = Dense(64, activation='relu', name='x60fc2')(x60)
    x60 = Dropout(0.2)(x60)
    x60 = Dense(2, activation='softmax', name='x60_out')(x60)

    x61 = Dense(64, activation='relu', name='x61fc1')(x)
    x61 = Dropout(0.2)(x61)
    x61 = Dense(64, activation='relu', name='x61fc2')(x61)
    x61 = Dropout(0.2)(x61)
    x61 = Dense(2, activation='softmax', name='x61_out')(x61)

    x62 = Dense(64, activation='relu', name='x62fc1')(x)
    x62 = Dropout(0.2)(x62)
    x62 = Dense(64, activation='relu', name='x62fc2')(x62)
    x62 = Dropout(0.2)(x62)
    x62 = Dense(2, activation='softmax', name='x62_out')(x62)

    x63 = Dense(64, activation='relu', name='x63fc1')(x)
    x63 = Dropout(0.2)(x63)
    x63 = Dense(64, activation='relu', name='x63fc2')(x63)
    x63 = Dropout(0.2)(x63)
    x63 = Dense(2, activation='softmax', name='x63_out')(x63)

    x64 = Dense(64, activation='relu', name='x64fc1')(x)
    x64 = Dropout(0.2)(x64)
    x64 = Dense(64, activation='relu', name='x64fc2')(x64)
    x64 = Dropout(0.2)(x64)
    x64 = Dense(2, activation='softmax', name='x64_out')(x64)

    x65 = Dense(64, activation='relu', name='x65fc1')(x)
    x65 = Dropout(0.2)(x65)
    x65 = Dense(64, activation='relu', name='x65fc2')(x65)
    x65 = Dropout(0.2)(x65)
    x65 = Dense(2, activation='softmax', name='x65_out')(x65)

    x66 = Dense(64, activation='relu', name='x66fc1')(x)
    x66 = Dropout(0.2)(x66)
    x66 = Dense(64, activation='relu', name='x66fc2')(x66)
    x66 = Dropout(0.2)(x66)
    x66 = Dense(2, activation='softmax', name='x66_out')(x66)

    x67 = Dense(64, activation='relu', name='x67fc1')(x)
    x67 = Dropout(0.2)(x67)
    x67 = Dense(64, activation='relu', name='x67fc2')(x67)
    x67 = Dropout(0.2)(x67)
    x67 = Dense(2, activation='softmax', name='x67_out')(x67)

    x68 = Dense(64, activation='relu', name='x68fc1')(x)
    x68 = Dropout(0.2)(x68)
    x68 = Dense(64, activation='relu', name='x68fc2')(x68)
    x68 = Dropout(0.2)(x68)
    x68 = Dense(2, activation='softmax', name='x68_out')(x68)

    x69 = Dense(64, activation='relu', name='x69fc1')(x)
    x69 = Dropout(0.2)(x69)
    x69 = Dense(64, activation='relu', name='x69fc2')(x69)
    x69 = Dropout(0.2)(x69)
    x69 = Dense(2, activation='softmax', name='x69_out')(x69)

    x70 = Dense(64, activation='relu', name='x70fc1')(x)
    x70 = Dropout(0.2)(x70)
    x70 = Dense(64, activation='relu', name='x70fc2')(x70)
    x70 = Dropout(0.2)(x70)
    x70 = Dense(2, activation='softmax', name='x70_out')(x70)

    x71 = Dense(64, activation='relu', name='x71fc1')(x)
    x71 = Dropout(0.2)(x71)
    x71 = Dense(64, activation='relu', name='x71fc2')(x71)
    x71 = Dropout(0.2)(x71)
    x71 = Dense(2, activation='softmax', name='x71_out')(x71)

    x72 = Dense(64, activation='relu', name='x72fc1')(x)
    x72 = Dropout(0.2)(x72)
    x72 = Dense(64, activation='relu', name='x72fc2')(x72)
    x72 = Dropout(0.2)(x72)
    x72 = Dense(2, activation='softmax', name='x72_out')(x72)

    x73 = Dense(64, activation='relu', name='x73fc1')(x)
    x73 = Dropout(0.2)(x73)
    x73 = Dense(64, activation='relu', name='x73fc2')(x73)
    x73 = Dropout(0.2)(x73)
    x73 = Dense(2, activation='softmax', name='x73_out')(x73)

    x74 = Dense(64, activation='relu', name='x74fc1')(x)
    x74 = Dropout(0.2)(x74)
    x74 = Dense(64, activation='relu', name='x74fc2')(x74)
    x74 = Dropout(0.2)(x74)
    x74 = Dense(2, activation='softmax', name='x74_out')(x74)

    x75 = Dense(64, activation='relu', name='x75fc1')(x)
    x75 = Dropout(0.2)(x75)
    x75 = Dense(64, activation='relu', name='x75fc2')(x75)
    x75 = Dropout(0.2)(x75)
    x75 = Dense(2, activation='softmax', name='x75_out')(x75)

    x76 = Dense(64, activation='relu', name='x76fc1')(x)
    x76 = Dropout(0.2)(x76)
    x76 = Dense(64, activation='relu', name='x76fc2')(x76)
    x76 = Dropout(0.2)(x76)
    x76 = Dense(2, activation='softmax', name='x76_out')(x76)

    x77 = Dense(64, activation='relu', name='x77fc1')(x)
    x77 = Dropout(0.2)(x77)
    x77 = Dense(64, activation='relu', name='x77fc2')(x77)
    x77 = Dropout(0.2)(x77)
    x77 = Dense(2, activation='softmax', name='x77_out')(x77)

    x78 = Dense(64, activation='relu', name='x78fc1')(x)
    x78 = Dropout(0.2)(x78)
    x78 = Dense(64, activation='relu', name='x78fc2')(x78)
    x78 = Dropout(0.2)(x78)
    x78 = Dense(2, activation='softmax', name='x78_out')(x78)

    x79 = Dense(64, activation='relu', name='x79fc1')(x)
    x79 = Dropout(0.2)(x79)
    x79 = Dense(64, activation='relu', name='x79fc2')(x79)
    x79 = Dropout(0.2)(x79)
    x79 = Dense(2, activation='softmax', name='x79_out')(x79)

    x80 = Dense(64, activation='relu', name='x80fc1')(x)
    x80 = Dropout(0.2)(x80)
    x80 = Dense(64, activation='relu', name='x80fc2')(x80)
    x80 = Dropout(0.2)(x80)
    x80 = Dense(2, activation='softmax', name='x80_out')(x80)

    x81 = Dense(64, activation='relu', name='x81fc1')(x)
    x81 = Dropout(0.2)(x81)
    x81 = Dense(64, activation='relu', name='x81fc2')(x81)
    x81 = Dropout(0.2)(x81)
    x81 = Dense(2, activation='softmax', name='x81_out')(x81)

    x82 = Dense(64, activation='relu', name='x82fc1')(x)
    x82 = Dropout(0.2)(x82)
    x82 = Dense(64, activation='relu', name='x82fc2')(x82)
    x82 = Dropout(0.2)(x82)
    x82 = Dense(2, activation='softmax', name='x82_out')(x82)

    x83 = Dense(64, activation='relu', name='x83fc1')(x)
    x83 = Dropout(0.2)(x83)
    x83 = Dense(64, activation='relu', name='x83fc2')(x83)
    x83 = Dropout(0.2)(x83)
    x83 = Dense(2, activation='softmax', name='x83_out')(x83)

    x84 = Dense(64, activation='relu', name='x84fc1')(x)
    x84 = Dropout(0.2)(x84)
    x84 = Dense(64, activation='relu', name='x84fc2')(x84)
    x84 = Dropout(0.2)(x84)
    x84 = Dense(2, activation='softmax', name='x84_out')(x84)

    x85 = Dense(64, activation='relu', name='x85fc1')(x)
    x85 = Dropout(0.2)(x85)
    x85 = Dense(64, activation='relu', name='x85fc2')(x85)
    x85 = Dropout(0.2)(x85)
    x85 = Dense(2, activation='softmax', name='x85_out')(x85)

    x86 = Dense(64, activation='relu', name='x86fc1')(x)
    x86 = Dropout(0.2)(x86)
    x86 = Dense(64, activation='relu', name='x86fc2')(x86)
    x86 = Dropout(0.2)(x86)
    x86 = Dense(2, activation='softmax', name='x86_out')(x86)

    x87 = Dense(64, activation='relu', name='x87fc1')(x)
    x87 = Dropout(0.2)(x87)
    x87 = Dense(64, activation='relu', name='x87fc2')(x87)
    x87 = Dropout(0.2)(x87)
    x87 = Dense(2, activation='softmax', name='x87_out')(x87)

    x88 = Dense(64, activation='relu', name='x88fc1')(x)
    x88 = Dropout(0.2)(x88)
    x88 = Dense(64, activation='relu', name='x88fc2')(x88)
    x88 = Dropout(0.2)(x88)
    x88 = Dense(2, activation='softmax', name='x88_out')(x88)

    x89 = Dense(64, activation='relu', name='x89fc1')(x)
    x89 = Dropout(0.2)(x89)
    x89 = Dense(64, activation='relu', name='x89fc2')(x89)
    x89 = Dropout(0.2)(x89)
    x89 = Dense(2, activation='softmax', name='x89_out')(x89)

    x90 = Dense(64, activation='relu', name='x90fc1')(x)
    x90 = Dropout(0.2)(x90)
    x90 = Dense(64, activation='relu', name='x90fc2')(x90)
    x90 = Dropout(0.2)(x90)
    x90 = Dense(2, activation='softmax', name='x90_out')(x90)

    x91 = Dense(64, activation='relu', name='x91fc1')(x)
    x91 = Dropout(0.2)(x91)
    x91 = Dense(64, activation='relu', name='x91fc2')(x91)
    x91 = Dropout(0.2)(x91)
    x91 = Dense(2, activation='softmax', name='x91_out')(x91)

    x92 = Dense(64, activation='relu', name='x92fc1')(x)
    x92 = Dropout(0.2)(x92)
    x92 = Dense(64, activation='relu', name='x92fc2')(x92)
    x92 = Dropout(0.2)(x92)
    x92 = Dense(2, activation='softmax', name='x92_out')(x92)

    x93 = Dense(64, activation='relu', name='x93fc1')(x)
    x93 = Dropout(0.2)(x93)
    x93 = Dense(64, activation='relu', name='x93fc2')(x93)
    x93 = Dropout(0.2)(x93)
    x93 = Dense(2, activation='softmax', name='x93_out')(x93)

    x94 = Dense(64, activation='relu', name='x94fc1')(x)
    x94 = Dropout(0.2)(x94)
    x94 = Dense(64, activation='relu', name='x94fc2')(x94)
    x94 = Dropout(0.2)(x94)
    x94 = Dense(2, activation='softmax', name='x94_out')(x94)

    x95 = Dense(64, activation='relu', name='x95fc1')(x)
    x95 = Dropout(0.2)(x95)
    x95 = Dense(64, activation='relu', name='x95fc2')(x95)
    x95 = Dropout(0.2)(x95)
    x95 = Dense(2, activation='softmax', name='x95_out')(x95)

    x96 = Dense(64, activation='relu', name='x96fc1')(x)
    x96 = Dropout(0.2)(x96)
    x96 = Dense(64, activation='relu', name='x96fc2')(x96)
    x96 = Dropout(0.2)(x96)
    x96 = Dense(2, activation='softmax', name='x96_out')(x96)

    x97 = Dense(64, activation='relu', name='x97fc1')(x)
    x97 = Dropout(0.2)(x97)
    x97 = Dense(64, activation='relu', name='x97fc2')(x97)
    x97 = Dropout(0.2)(x97)
    x97 = Dense(2, activation='softmax', name='x97_out')(x97)

    x98 = Dense(64, activation='relu', name='x98fc1')(x)
    x98 = Dropout(0.2)(x98)
    x98 = Dense(64, activation='relu', name='x98fc2')(x98)
    x98 = Dropout(0.2)(x98)
    x98 = Dense(2, activation='softmax', name='x98_out')(x98)

    x99 = Dense(64, activation='relu', name='x99fc1')(x)
    x99 = Dropout(0.2)(x99)
    x99 = Dense(64, activation='relu', name='x99fc2')(x99)
    x99 = Dropout(0.2)(x99)
    x99 = Dense(2, activation='softmax', name='x99_out')(x99)

    x100 = Dense(64, activation='relu', name='x100fc1')(x)
    x100 = Dropout(0.2)(x100)
    x100 = Dense(64, activation='relu', name='x100fc2')(x100)
    x100 = Dropout(0.2)(x100)
    x100 = Dense(2, activation='softmax', name='x100_out')(x100)

    x101 = Dense(64, activation='relu', name='x101fc1')(x)
    x101 = Dropout(0.2)(x101)
    x101 = Dense(64, activation='relu', name='x101fc2')(x101)
    x101 = Dropout(0.2)(x101)
    x101 = Dense(2, activation='softmax', name='x101_out')(x101)

    x102 = Dense(64, activation='relu', name='x102fc1')(x)
    x102 = Dropout(0.2)(x102)
    x102 = Dense(64, activation='relu', name='x102fc2')(x102)
    x102 = Dropout(0.2)(x102)
    x102 = Dense(2, activation='softmax', name='x102_out')(x102)

    x103 = Dense(64, activation='relu', name='x103fc1')(x)
    x103 = Dropout(0.2)(x103)
    x103 = Dense(64, activation='relu', name='x103fc2')(x103)
    x103 = Dropout(0.2)(x103)
    x103 = Dense(2, activation='softmax', name='x103_out')(x103)

    x104 = Dense(64, activation='relu', name='x104fc1')(x)
    x104 = Dropout(0.2)(x104)
    x104 = Dense(64, activation='relu', name='x104fc2')(x104)
    x104 = Dropout(0.2)(x104)
    x104 = Dense(2, activation='softmax', name='x104_out')(x104)

    x105 = Dense(64, activation='relu', name='x105fc1')(x)
    x105 = Dropout(0.2)(x105)
    x105 = Dense(64, activation='relu', name='x105fc2')(x105)
    x105 = Dropout(0.2)(x105)
    x105 = Dense(2, activation='softmax', name='x105_out')(x105)

    x106 = Dense(64, activation='relu', name='x106fc1')(x)
    x106 = Dropout(0.2)(x106)
    x106 = Dense(64, activation='relu', name='x106fc2')(x106)
    x106 = Dropout(0.2)(x106)
    x106 = Dense(2, activation='softmax', name='x106_out')(x106)

    x107 = Dense(64, activation='relu', name='x107fc1')(x)
    x107 = Dropout(0.2)(x107)
    x107 = Dense(64, activation='relu', name='x107fc2')(x107)
    x107 = Dropout(0.2)(x107)
    x107 = Dense(2, activation='softmax', name='x107_out')(x107)

    x108 = Dense(64, activation='relu', name='x108fc1')(x)
    x108 = Dropout(0.2)(x108)
    x108 = Dense(64, activation='relu', name='x108fc2')(x108)
    x108 = Dropout(0.2)(x108)
    x108 = Dense(2, activation='softmax', name='x108_out')(x108)

    x109 = Dense(64, activation='relu', name='x109fc1')(x)
    x109 = Dropout(0.2)(x109)
    x109 = Dense(64, activation='relu', name='x109fc2')(x109)
    x109 = Dropout(0.2)(x109)
    x109 = Dense(2, activation='softmax', name='x109_out')(x109)

    x110 = Dense(64, activation='relu', name='x110fc1')(x)
    x110 = Dropout(0.2)(x110)
    x110 = Dense(64, activation='relu', name='x110fc2')(x110)
    x110 = Dropout(0.2)(x110)
    x110 = Dense(2, activation='softmax', name='x110_out')(x110)

    x111 = Dense(64, activation='relu', name='x111fc1')(x)
    x111 = Dropout(0.2)(x111)
    x111 = Dense(64, activation='relu', name='x111fc2')(x111)
    x111 = Dropout(0.2)(x111)
    x111 = Dense(2, activation='softmax', name='x111_out')(x111)

    x112 = Dense(64, activation='relu', name='x112fc1')(x)
    x112 = Dropout(0.2)(x112)
    x112 = Dense(64, activation='relu', name='x112fc2')(x112)
    x112 = Dropout(0.2)(x112)
    x112 = Dense(2, activation='softmax', name='x112_out')(x112)

    x113 = Dense(64, activation='relu', name='x113fc1')(x)
    x113 = Dropout(0.2)(x113)
    x113 = Dense(64, activation='relu', name='x113fc2')(x113)
    x113 = Dropout(0.2)(x113)
    x113 = Dense(2, activation='softmax', name='x113_out')(x113)

    x114 = Dense(64, activation='relu', name='x114fc1')(x)
    x114 = Dropout(0.2)(x114)
    x114 = Dense(64, activation='relu', name='x114fc2')(x114)
    x114 = Dropout(0.2)(x114)
    x114 = Dense(2, activation='softmax', name='x114_out')(x114)

    x115 = Dense(64, activation='relu', name='x115fc1')(x)
    x115 = Dropout(0.2)(x115)
    x115 = Dense(64, activation='relu', name='x115fc2')(x115)
    x115 = Dropout(0.2)(x115)
    x115 = Dense(2, activation='softmax', name='x115_out')(x115)

    x116 = Dense(64, activation='relu', name='x116fc1')(x)
    x116 = Dropout(0.2)(x116)
    x116 = Dense(64, activation='relu', name='x116fc2')(x116)
    x116 = Dropout(0.2)(x116)
    x116 = Dense(2, activation='softmax', name='x116_out')(x116)

    x117 = Dense(64, activation='relu', name='x117fc1')(x)
    x117 = Dropout(0.2)(x117)
    x117 = Dense(64, activation='relu', name='x117fc2')(x117)
    x117 = Dropout(0.2)(x117)
    x117 = Dense(2, activation='softmax', name='x117_out')(x117)

    x118 = Dense(64, activation='relu', name='x118fc1')(x)
    x118 = Dropout(0.2)(x118)
    x118 = Dense(64, activation='relu', name='x118fc2')(x118)
    x118 = Dropout(0.2)(x118)
    x118 = Dense(2, activation='softmax', name='x118_out')(x118)

    x119 = Dense(64, activation='relu', name='x119fc1')(x)
    x119 = Dropout(0.2)(x119)
    x119 = Dense(64, activation='relu', name='x119fc2')(x119)
    x119 = Dropout(0.2)(x119)
    x119 = Dense(2, activation='softmax', name='x119_out')(x119)

    x120 = Dense(64, activation='relu', name='x120fc1')(x)
    x120 = Dropout(0.2)(x120)
    x120 = Dense(64, activation='relu', name='x120fc2')(x120)
    x120 = Dropout(0.2)(x120)
    x120 = Dense(2, activation='softmax', name='x120_out')(x120)

    x121 = Dense(64, activation='relu', name='x121fc1')(x)
    x121 = Dropout(0.2)(x121)
    x121 = Dense(64, activation='relu', name='x121fc2')(x121)
    x121 = Dropout(0.2)(x121)
    x121 = Dense(2, activation='softmax', name='x121_out')(x121)

    x122 = Dense(64, activation='relu', name='x122fc1')(x)
    x122 = Dropout(0.2)(x122)
    x122 = Dense(64, activation='relu', name='x122fc2')(x122)
    x122 = Dropout(0.2)(x122)
    x122 = Dense(2, activation='softmax', name='x122_out')(x122)

    x123 = Dense(64, activation='relu', name='x123fc1')(x)
    x123 = Dropout(0.2)(x123)
    x123 = Dense(64, activation='relu', name='x123fc2')(x123)
    x123 = Dropout(0.2)(x123)
    x123 = Dense(2, activation='softmax', name='x123_out')(x123)

    x124 = Dense(64, activation='relu', name='x124fc1')(x)
    x124 = Dropout(0.2)(x124)
    x124 = Dense(64, activation='relu', name='x124fc2')(x124)
    x124 = Dropout(0.2)(x124)
    x124 = Dense(2, activation='softmax', name='x124_out')(x124)

    x125 = Dense(64, activation='relu', name='x125fc1')(x)
    x125 = Dropout(0.2)(x125)
    x125 = Dense(64, activation='relu', name='x125fc2')(x125)
    x125 = Dropout(0.2)(x125)
    x125 = Dense(2, activation='softmax', name='x125_out')(x125)

    x126 = Dense(64, activation='relu', name='x126fc1')(x)
    x126 = Dropout(0.2)(x126)
    x126 = Dense(64, activation='relu', name='x126fc2')(x126)
    x126 = Dropout(0.2)(x126)
    x126 = Dense(2, activation='softmax', name='x126_out')(x126)

    x127 = Dense(64, activation='relu', name='x127fc1')(x)
    x127 = Dropout(0.2)(x127)
    x127 = Dense(64, activation='relu', name='x127fc2')(x127)
    x127 = Dropout(0.2)(x127)
    x127 = Dense(2, activation='softmax', name='x127_out')(x127)

    x128 = Dense(64, activation='relu', name='x128fc1')(x)
    x128 = Dropout(0.2)(x128)
    x128 = Dense(64, activation='relu', name='x128fc2')(x128)
    x128 = Dropout(0.2)(x128)
    x128 = Dense(2, activation='softmax', name='x128_out')(x128)

    x129 = Dense(64, activation='relu', name='x129fc1')(x)
    x129 = Dropout(0.2)(x129)
    x129 = Dense(64, activation='relu', name='x129fc2')(x129)
    x129 = Dropout(0.2)(x129)
    x129 = Dense(2, activation='softmax', name='x129_out')(x129)

    x130 = Dense(64, activation='relu', name='x130fc1')(x)
    x130 = Dropout(0.2)(x130)
    x130 = Dense(64, activation='relu', name='x130fc2')(x130)
    x130 = Dropout(0.2)(x130)
    x130 = Dense(2, activation='softmax', name='x130_out')(x130)

    x131 = Dense(64, activation='relu', name='x131fc1')(x)
    x131 = Dropout(0.2)(x131)
    x131 = Dense(64, activation='relu', name='x131fc2')(x131)
    x131 = Dropout(0.2)(x131)
    x131 = Dense(2, activation='softmax', name='x131_out')(x131)

    x132 = Dense(64, activation='relu', name='x132fc1')(x)
    x132 = Dropout(0.2)(x132)
    x132 = Dense(64, activation='relu', name='x132fc2')(x132)
    x132 = Dropout(0.2)(x132)
    x132 = Dense(2, activation='softmax', name='x132_out')(x132)

    x133 = Dense(64, activation='relu', name='x133fc1')(x)
    x133 = Dropout(0.2)(x133)
    x133 = Dense(64, activation='relu', name='x133fc2')(x133)
    x133 = Dropout(0.2)(x133)
    x133 = Dense(2, activation='softmax', name='x133_out')(x133)

    x134 = Dense(64, activation='relu', name='x134fc1')(x)
    x134 = Dropout(0.2)(x134)
    x134 = Dense(64, activation='relu', name='x134fc2')(x134)
    x134 = Dropout(0.2)(x134)
    x134 = Dense(2, activation='softmax', name='x134_out')(x134)

    x135 = Dense(64, activation='relu', name='x135fc1')(x)
    x135 = Dropout(0.2)(x135)
    x135 = Dense(64, activation='relu', name='x135fc2')(x135)
    x135 = Dropout(0.2)(x135)
    x135 = Dense(2, activation='softmax', name='x135_out')(x135)

    x136 = Dense(64, activation='relu', name='x136fc1')(x)
    x136 = Dropout(0.2)(x136)
    x136 = Dense(64, activation='relu', name='x136fc2')(x136)
    x136 = Dropout(0.2)(x136)
    x136 = Dense(2, activation='softmax', name='x136_out')(x136)

    x137 = Dense(64, activation='relu', name='x137fc1')(x)
    x137 = Dropout(0.2)(x137)
    x137 = Dense(64, activation='relu', name='x137fc2')(x137)
    x137 = Dropout(0.2)(x137)
    x137 = Dense(2, activation='softmax', name='x137_out')(x137)

    x138 = Dense(64, activation='relu', name='x138fc1')(x)
    x138 = Dropout(0.2)(x138)
    x138 = Dense(64, activation='relu', name='x138fc2')(x138)
    x138 = Dropout(0.2)(x138)
    x138 = Dense(2, activation='softmax', name='x138_out')(x138)

    x139 = Dense(64, activation='relu', name='x139fc1')(x)
    x139 = Dropout(0.2)(x139)
    x139 = Dense(64, activation='relu', name='x139fc2')(x139)
    x139 = Dropout(0.2)(x139)
    x139 = Dense(2, activation='softmax', name='x139_out')(x139)

    x140 = Dense(64, activation='relu', name='x140fc1')(x)
    x140 = Dropout(0.2)(x140)
    x140 = Dense(64, activation='relu', name='x140fc2')(x140)
    x140 = Dropout(0.2)(x140)
    x140 = Dense(2, activation='softmax', name='x140_out')(x140)

    x141 = Dense(64, activation='relu', name='x141fc1')(x)
    x141 = Dropout(0.2)(x141)
    x141 = Dense(64, activation='relu', name='x141fc2')(x141)
    x141 = Dropout(0.2)(x141)
    x141 = Dense(2, activation='softmax', name='x141_out')(x141)

    x142 = Dense(64, activation='relu', name='x142fc1')(x)
    x142 = Dropout(0.2)(x142)
    x142 = Dense(64, activation='relu', name='x142fc2')(x142)
    x142 = Dropout(0.2)(x142)
    x142 = Dense(2, activation='softmax', name='x142_out')(x142)

    x143 = Dense(64, activation='relu', name='x143fc1')(x)
    x143 = Dropout(0.2)(x143)
    x143 = Dense(64, activation='relu', name='x143fc2')(x143)
    x143 = Dropout(0.2)(x143)
    x143 = Dense(2, activation='softmax', name='x143_out')(x143)

    x144 = Dense(64, activation='relu', name='x144fc1')(x)
    x144 = Dropout(0.2)(x144)
    x144 = Dense(64, activation='relu', name='x144fc2')(x144)
    x144 = Dropout(0.2)(x144)
    x144 = Dense(2, activation='softmax', name='x144_out')(x144)

    x145 = Dense(64, activation='relu', name='x145fc1')(x)
    x145 = Dropout(0.2)(x145)
    x145 = Dense(64, activation='relu', name='x145fc2')(x145)
    x145 = Dropout(0.2)(x145)
    x145 = Dense(2, activation='softmax', name='x145_out')(x145)

    x146 = Dense(64, activation='relu', name='x146fc1')(x)
    x146 = Dropout(0.2)(x146)
    x146 = Dense(64, activation='relu', name='x146fc2')(x146)
    x146 = Dropout(0.2)(x146)
    x146 = Dense(2, activation='softmax', name='x146_out')(x146)

    x147 = Dense(64, activation='relu', name='x147fc1')(x)
    x147 = Dropout(0.2)(x147)
    x147 = Dense(64, activation='relu', name='x147fc2')(x147)
    x147 = Dropout(0.2)(x147)
    x147 = Dense(2, activation='softmax', name='x147_out')(x147)

    x148 = Dense(64, activation='relu', name='x148fc1')(x)
    x148 = Dropout(0.2)(x148)
    x148 = Dense(64, activation='relu', name='x148fc2')(x148)
    x148 = Dropout(0.2)(x148)
    x148 = Dense(2, activation='softmax', name='x148_out')(x148)

    x149 = Dense(64, activation='relu', name='x149fc1')(x)
    x149 = Dropout(0.2)(x149)
    x149 = Dense(64, activation='relu', name='x149fc2')(x149)
    x149 = Dropout(0.2)(x149)
    x149 = Dense(2, activation='softmax', name='x149_out')(x149)

    x150 = Dense(64, activation='relu', name='x150fc1')(x)
    x150 = Dropout(0.2)(x150)
    x150 = Dense(64, activation='relu', name='x150fc2')(x150)
    x150 = Dropout(0.2)(x150)
    x150 = Dense(2, activation='softmax', name='x150_out')(x150)

    x151 = Dense(64, activation='relu', name='x151fc1')(x)
    x151 = Dropout(0.2)(x151)
    x151 = Dense(64, activation='relu', name='x151fc2')(x151)
    x151 = Dropout(0.2)(x151)
    x151 = Dense(2, activation='softmax', name='x151_out')(x151)

    x152 = Dense(64, activation='relu', name='x152fc1')(x)
    x152 = Dropout(0.2)(x152)
    x152 = Dense(64, activation='relu', name='x152fc2')(x152)
    x152 = Dropout(0.2)(x152)
    x152 = Dense(2, activation='softmax', name='x152_out')(x152)

    x153 = Dense(64, activation='relu', name='x153fc1')(x)
    x153 = Dropout(0.2)(x153)
    x153 = Dense(64, activation='relu', name='x153fc2')(x153)
    x153 = Dropout(0.2)(x153)
    x153 = Dense(2, activation='softmax', name='x153_out')(x153)

    x154 = Dense(64, activation='relu', name='x154fc1')(x)
    x154 = Dropout(0.2)(x154)
    x154 = Dense(64, activation='relu', name='x154fc2')(x154)
    x154 = Dropout(0.2)(x154)
    x154 = Dense(2, activation='softmax', name='x154_out')(x154)

    x155 = Dense(64, activation='relu', name='x155fc1')(x)
    x155 = Dropout(0.2)(x155)
    x155 = Dense(64, activation='relu', name='x155fc2')(x155)
    x155 = Dropout(0.2)(x155)
    x155 = Dense(2, activation='softmax', name='x155_out')(x155)

    x156 = Dense(64, activation='relu', name='x156fc1')(x)
    x156 = Dropout(0.2)(x156)
    x156 = Dense(64, activation='relu', name='x156fc2')(x156)
    x156 = Dropout(0.2)(x156)
    x156 = Dense(2, activation='softmax', name='x156_out')(x156)

    x157 = Dense(64, activation='relu', name='x157fc1')(x)
    x157 = Dropout(0.2)(x157)
    x157 = Dense(64, activation='relu', name='x157fc2')(x157)
    x157 = Dropout(0.2)(x157)
    x157 = Dense(2, activation='softmax', name='x157_out')(x157)

    x158 = Dense(64, activation='relu', name='x158fc1')(x)
    x158 = Dropout(0.2)(x158)
    x158 = Dense(64, activation='relu', name='x158fc2')(x158)
    x158 = Dropout(0.2)(x158)
    x158 = Dense(2, activation='softmax', name='x158_out')(x158)

    x159 = Dense(64, activation='relu', name='x159fc1')(x)
    x159 = Dropout(0.2)(x159)
    x159 = Dense(64, activation='relu', name='x159fc2')(x159)
    x159 = Dropout(0.2)(x159)
    x159 = Dense(2, activation='softmax', name='x159_out')(x159)

    x160 = Dense(64, activation='relu', name='x160fc1')(x)
    x160 = Dropout(0.2)(x160)
    x160 = Dense(64, activation='relu', name='x160fc2')(x160)
    x160 = Dropout(0.2)(x160)
    x160 = Dense(2, activation='softmax', name='x160_out')(x160)

    x161 = Dense(64, activation='relu', name='x161fc1')(x)
    x161 = Dropout(0.2)(x161)
    x161 = Dense(64, activation='relu', name='x161fc2')(x161)
    x161 = Dropout(0.2)(x161)
    x161 = Dense(2, activation='softmax', name='x161_out')(x161)

    x162 = Dense(64, activation='relu', name='x162fc1')(x)
    x162 = Dropout(0.2)(x162)
    x162 = Dense(64, activation='relu', name='x162fc2')(x162)
    x162 = Dropout(0.2)(x162)
    x162 = Dense(2, activation='softmax', name='x162_out')(x162)

    x163 = Dense(64, activation='relu', name='x163fc1')(x)
    x163 = Dropout(0.2)(x163)
    x163 = Dense(64, activation='relu', name='x163fc2')(x163)
    x163 = Dropout(0.2)(x163)
    x163 = Dense(2, activation='softmax', name='x163_out')(x163)

    x164 = Dense(64, activation='relu', name='x164fc1')(x)
    x164 = Dropout(0.2)(x164)
    x164 = Dense(64, activation='relu', name='x164fc2')(x164)
    x164 = Dropout(0.2)(x164)
    x164 = Dense(2, activation='softmax', name='x164_out')(x164)

    x165 = Dense(64, activation='relu', name='x165fc1')(x)
    x165 = Dropout(0.2)(x165)
    x165 = Dense(64, activation='relu', name='x165fc2')(x165)
    x165 = Dropout(0.2)(x165)
    x165 = Dense(2, activation='softmax', name='x165_out')(x165)

    x166 = Dense(64, activation='relu', name='x166fc1')(x)
    x166 = Dropout(0.2)(x166)
    x166 = Dense(64, activation='relu', name='x166fc2')(x166)
    x166 = Dropout(0.2)(x166)
    x166 = Dense(2, activation='softmax', name='x166_out')(x166)

    x167 = Dense(64, activation='relu', name='x167fc1')(x)
    x167 = Dropout(0.2)(x167)
    x167 = Dense(64, activation='relu', name='x167fc2')(x167)
    x167 = Dropout(0.2)(x167)
    x167 = Dense(2, activation='softmax', name='x167_out')(x167)

    x168 = Dense(64, activation='relu', name='x168fc1')(x)
    x168 = Dropout(0.2)(x168)
    x168 = Dense(64, activation='relu', name='x168fc2')(x168)
    x168 = Dropout(0.2)(x168)
    x168 = Dense(2, activation='softmax', name='x168_out')(x168)

    x169 = Dense(64, activation='relu', name='x169fc1')(x)
    x169 = Dropout(0.2)(x169)
    x169 = Dense(64, activation='relu', name='x169fc2')(x169)
    x169 = Dropout(0.2)(x169)
    x169 = Dense(2, activation='softmax', name='x169_out')(x169)

    x170 = Dense(64, activation='relu', name='x170fc1')(x)
    x170 = Dropout(0.2)(x170)
    x170 = Dense(64, activation='relu', name='x170fc2')(x170)
    x170 = Dropout(0.2)(x170)
    x170 = Dense(2, activation='softmax', name='x170_out')(x170)

    x171 = Dense(64, activation='relu', name='x171fc1')(x)
    x171 = Dropout(0.2)(x171)
    x171 = Dense(64, activation='relu', name='x171fc2')(x171)
    x171 = Dropout(0.2)(x171)
    x171 = Dense(2, activation='softmax', name='x171_out')(x171)

    x172 = Dense(64, activation='relu', name='x172fc1')(x)
    x172 = Dropout(0.2)(x172)
    x172 = Dense(64, activation='relu', name='x172fc2')(x172)
    x172 = Dropout(0.2)(x172)
    x172 = Dense(2, activation='softmax', name='x172_out')(x172)

    x173 = Dense(64, activation='relu', name='x173fc1')(x)
    x173 = Dropout(0.2)(x173)
    x173 = Dense(64, activation='relu', name='x173fc2')(x173)
    x173 = Dropout(0.2)(x173)
    x173 = Dense(2, activation='softmax', name='x173_out')(x173)

    x174 = Dense(64, activation='relu', name='x174fc1')(x)
    x174 = Dropout(0.2)(x174)
    x174 = Dense(64, activation='relu', name='x174fc2')(x174)
    x174 = Dropout(0.2)(x174)
    x174 = Dense(2, activation='softmax', name='x174_out')(x174)

    x175 = Dense(64, activation='relu', name='x175fc1')(x)
    x175 = Dropout(0.2)(x175)
    x175 = Dense(64, activation='relu', name='x175fc2')(x175)
    x175 = Dropout(0.2)(x175)
    x175 = Dense(2, activation='softmax', name='x175_out')(x175)

    x176 = Dense(64, activation='relu', name='x176fc1')(x)
    x176 = Dropout(0.2)(x176)
    x176 = Dense(64, activation='relu', name='x176fc2')(x176)
    x176 = Dropout(0.2)(x176)
    x176 = Dense(2, activation='softmax', name='x176_out')(x176)

    x177 = Dense(64, activation='relu', name='x177fc1')(x)
    x177 = Dropout(0.2)(x177)
    x177 = Dense(64, activation='relu', name='x177fc2')(x177)
    x177 = Dropout(0.2)(x177)
    x177 = Dense(2, activation='softmax', name='x177_out')(x177)

    x178 = Dense(64, activation='relu', name='x178fc1')(x)
    x178 = Dropout(0.2)(x178)
    x178 = Dense(64, activation='relu', name='x178fc2')(x178)
    x178 = Dropout(0.2)(x178)
    x178 = Dense(2, activation='softmax', name='x178_out')(x178)

    x179 = Dense(64, activation='relu', name='x179fc1')(x)
    x179 = Dropout(0.2)(x179)
    x179 = Dense(64, activation='relu', name='x179fc2')(x179)
    x179 = Dropout(0.2)(x179)
    x179 = Dense(2, activation='softmax', name='x179_out')(x179)

    x180 = Dense(64, activation='relu', name='x180fc1')(x)
    x180 = Dropout(0.2)(x180)
    x180 = Dense(64, activation='relu', name='x180fc2')(x180)
    x180 = Dropout(0.2)(x180)
    x180 = Dense(2, activation='softmax', name='x180_out')(x180)

    x181 = Dense(64, activation='relu', name='x181fc1')(x)
    x181 = Dropout(0.2)(x181)
    x181 = Dense(64, activation='relu', name='x181fc2')(x181)
    x181 = Dropout(0.2)(x181)
    x181 = Dense(2, activation='softmax', name='x181_out')(x181)

    x182 = Dense(64, activation='relu', name='x182fc1')(x)
    x182 = Dropout(0.2)(x182)
    x182 = Dense(64, activation='relu', name='x182fc2')(x182)
    x182 = Dropout(0.2)(x182)
    x182 = Dense(2, activation='softmax', name='x182_out')(x182)

    x183 = Dense(64, activation='relu', name='x183fc1')(x)
    x183 = Dropout(0.2)(x183)
    x183 = Dense(64, activation='relu', name='x183fc2')(x183)
    x183 = Dropout(0.2)(x183)
    x183 = Dense(2, activation='softmax', name='x183_out')(x183)

    x184 = Dense(64, activation='relu', name='x184fc1')(x)
    x184 = Dropout(0.2)(x184)
    x184 = Dense(64, activation='relu', name='x184fc2')(x184)
    x184 = Dropout(0.2)(x184)
    x184 = Dense(2, activation='softmax', name='x184_out')(x184)

    x185 = Dense(64, activation='relu', name='x185fc1')(x)
    x185 = Dropout(0.2)(x185)
    x185 = Dense(64, activation='relu', name='x185fc2')(x185)
    x185 = Dropout(0.2)(x185)
    x185 = Dense(2, activation='softmax', name='x185_out')(x185)

    x186 = Dense(64, activation='relu', name='x186fc1')(x)
    x186 = Dropout(0.2)(x186)
    x186 = Dense(64, activation='relu', name='x186fc2')(x186)
    x186 = Dropout(0.2)(x186)
    x186 = Dense(2, activation='softmax', name='x186_out')(x186)

    x187 = Dense(64, activation='relu', name='x187fc1')(x)
    x187 = Dropout(0.2)(x187)
    x187 = Dense(64, activation='relu', name='x187fc2')(x187)
    x187 = Dropout(0.2)(x187)
    x187 = Dense(2, activation='softmax', name='x187_out')(x187)

    x188 = Dense(64, activation='relu', name='x188fc1')(x)
    x188 = Dropout(0.2)(x188)
    x188 = Dense(64, activation='relu', name='x188fc2')(x188)
    x188 = Dropout(0.2)(x188)
    x188 = Dense(2, activation='softmax', name='x188_out')(x188)

    x189 = Dense(64, activation='relu', name='x189fc1')(x)
    x189 = Dropout(0.2)(x189)
    x189 = Dense(64, activation='relu', name='x189fc2')(x189)
    x189 = Dropout(0.2)(x189)
    x189 = Dense(2, activation='softmax', name='x189_out')(x189)

    x190 = Dense(64, activation='relu', name='x190fc1')(x)
    x190 = Dropout(0.2)(x190)
    x190 = Dense(64, activation='relu', name='x190fc2')(x190)
    x190 = Dropout(0.2)(x190)
    x190 = Dense(2, activation='softmax', name='x190_out')(x190)

    x191 = Dense(64, activation='relu', name='x191fc1')(x)
    x191 = Dropout(0.2)(x191)
    x191 = Dense(64, activation='relu', name='x191fc2')(x191)
    x191 = Dropout(0.2)(x191)
    x191 = Dense(2, activation='softmax', name='x191_out')(x191)

    x192 = Dense(64, activation='relu', name='x192fc1')(x)
    x192 = Dropout(0.2)(x192)
    x192 = Dense(64, activation='relu', name='x192fc2')(x192)
    x192 = Dropout(0.2)(x192)
    x192 = Dense(2, activation='softmax', name='x192_out')(x192)

    x193 = Dense(64, activation='relu', name='x193fc1')(x)
    x193 = Dropout(0.2)(x193)
    x193 = Dense(64, activation='relu', name='x193fc2')(x193)
    x193 = Dropout(0.2)(x193)
    x193 = Dense(2, activation='softmax', name='x193_out')(x193)

    x194 = Dense(64, activation='relu', name='x194fc1')(x)
    x194 = Dropout(0.2)(x194)
    x194 = Dense(64, activation='relu', name='x194fc2')(x194)
    x194 = Dropout(0.2)(x194)
    x194 = Dense(2, activation='softmax', name='x194_out')(x194)

    x195 = Dense(64, activation='relu', name='x195fc1')(x)
    x195 = Dropout(0.2)(x195)
    x195 = Dense(64, activation='relu', name='x195fc2')(x195)
    x195 = Dropout(0.2)(x195)
    x195 = Dense(2, activation='softmax', name='x195_out')(x195)

    x196 = Dense(64, activation='relu', name='x196fc1')(x)
    x196 = Dropout(0.2)(x196)
    x196 = Dense(64, activation='relu', name='x196fc2')(x196)
    x196 = Dropout(0.2)(x196)
    x196 = Dense(2, activation='softmax', name='x196_out')(x196)

    x197 = Dense(64, activation='relu', name='x197fc1')(x)
    x197 = Dropout(0.2)(x197)
    x197 = Dense(64, activation='relu', name='x197fc2')(x197)
    x197 = Dropout(0.2)(x197)
    x197 = Dense(2, activation='softmax', name='x197_out')(x197)

    x198 = Dense(64, activation='relu', name='x198fc1')(x)
    x198 = Dropout(0.2)(x198)
    x198 = Dense(64, activation='relu', name='x198fc2')(x198)
    x198 = Dropout(0.2)(x198)
    x198 = Dense(2, activation='softmax', name='x198_out')(x198)

    x199 = Dense(64, activation='relu', name='x199fc1')(x)
    x199 = Dropout(0.2)(x199)
    x199 = Dense(64, activation='relu', name='x199fc2')(x199)
    x199 = Dropout(0.2)(x199)
    x199 = Dense(2, activation='softmax', name='x199_out')(x199)

    x200 = Dense(64, activation='relu', name='x200fc1')(x)
    x200 = Dropout(0.2)(x200)
    x200 = Dense(64, activation='relu', name='x200fc2')(x200)
    x200 = Dropout(0.2)(x200)
    x200 = Dense(2, activation='softmax', name='x200_out')(x200)

    x201 = Dense(64, activation='relu', name='x201fc1')(x)
    x201 = Dropout(0.2)(x201)
    x201 = Dense(64, activation='relu', name='x201fc2')(x201)
    x201 = Dropout(0.2)(x201)
    x201 = Dense(2, activation='softmax', name='x201_out')(x201)

    x202 = Dense(64, activation='relu', name='x202fc1')(x)
    x202 = Dropout(0.2)(x202)
    x202 = Dense(64, activation='relu', name='x202fc2')(x202)
    x202 = Dropout(0.2)(x202)
    x202 = Dense(2, activation='softmax', name='x202_out')(x202)

    x203 = Dense(64, activation='relu', name='x203fc1')(x)
    x203 = Dropout(0.2)(x203)
    x203 = Dense(64, activation='relu', name='x203fc2')(x203)
    x203 = Dropout(0.2)(x203)
    x203 = Dense(2, activation='softmax', name='x203_out')(x203)

    x204 = Dense(64, activation='relu', name='x204fc1')(x)
    x204 = Dropout(0.2)(x204)
    x204 = Dense(64, activation='relu', name='x204fc2')(x204)
    x204 = Dropout(0.2)(x204)
    x204 = Dense(2, activation='softmax', name='x204_out')(x204)

    x205 = Dense(64, activation='relu', name='x205fc1')(x)
    x205 = Dropout(0.2)(x205)
    x205 = Dense(64, activation='relu', name='x205fc2')(x205)
    x205 = Dropout(0.2)(x205)
    x205 = Dense(2, activation='softmax', name='x205_out')(x205)

    x206 = Dense(64, activation='relu', name='x206fc1')(x)
    x206 = Dropout(0.2)(x206)
    x206 = Dense(64, activation='relu', name='x206fc2')(x206)
    x206 = Dropout(0.2)(x206)
    x206 = Dense(2, activation='softmax', name='x206_out')(x206)

    x207 = Dense(64, activation='relu', name='x207fc1')(x)
    x207 = Dropout(0.2)(x207)
    x207 = Dense(64, activation='relu', name='x207fc2')(x207)
    x207 = Dropout(0.2)(x207)
    x207 = Dense(2, activation='softmax', name='x207_out')(x207)

    x208 = Dense(64, activation='relu', name='x208fc1')(x)
    x208 = Dropout(0.2)(x208)
    x208 = Dense(64, activation='relu', name='x208fc2')(x208)
    x208 = Dropout(0.2)(x208)
    x208 = Dense(2, activation='softmax', name='x208_out')(x208)

    x209 = Dense(64, activation='relu', name='x209fc1')(x)
    x209 = Dropout(0.2)(x209)
    x209 = Dense(64, activation='relu', name='x209fc2')(x209)
    x209 = Dropout(0.2)(x209)
    x209 = Dense(2, activation='softmax', name='x209_out')(x209)

    x210 = Dense(64, activation='relu', name='x210fc1')(x)
    x210 = Dropout(0.2)(x210)
    x210 = Dense(64, activation='relu', name='x210fc2')(x210)
    x210 = Dropout(0.2)(x210)
    x210 = Dense(2, activation='softmax', name='x210_out')(x210)

    x211 = Dense(64, activation='relu', name='x211fc1')(x)
    x211 = Dropout(0.2)(x211)
    x211 = Dense(64, activation='relu', name='x211fc2')(x211)
    x211 = Dropout(0.2)(x211)
    x211 = Dense(2, activation='softmax', name='x211_out')(x211)

    x212 = Dense(64, activation='relu', name='x212fc1')(x)
    x212 = Dropout(0.2)(x212)
    x212 = Dense(64, activation='relu', name='x212fc2')(x212)
    x212 = Dropout(0.2)(x212)
    x212 = Dense(2, activation='softmax', name='x212_out')(x212)

    x213 = Dense(64, activation='relu', name='x213fc1')(x)
    x213 = Dropout(0.2)(x213)
    x213 = Dense(64, activation='relu', name='x213fc2')(x213)
    x213 = Dropout(0.2)(x213)
    x213 = Dense(2, activation='softmax', name='x213_out')(x213)

    x214 = Dense(64, activation='relu', name='x214fc1')(x)
    x214 = Dropout(0.2)(x214)
    x214 = Dense(64, activation='relu', name='x214fc2')(x214)
    x214 = Dropout(0.2)(x214)
    x214 = Dense(2, activation='softmax', name='x214_out')(x214)

    x215 = Dense(64, activation='relu', name='x215fc1')(x)
    x215 = Dropout(0.2)(x215)
    x215 = Dense(64, activation='relu', name='x215fc2')(x215)
    x215 = Dropout(0.2)(x215)
    x215 = Dense(2, activation='softmax', name='x215_out')(x215)

    x216 = Dense(64, activation='relu', name='x216fc1')(x)
    x216 = Dropout(0.2)(x216)
    x216 = Dense(64, activation='relu', name='x216fc2')(x216)
    x216 = Dropout(0.2)(x216)
    x216 = Dense(2, activation='softmax', name='x216_out')(x216)

    x217 = Dense(64, activation='relu', name='x217fc1')(x)
    x217 = Dropout(0.2)(x217)
    x217 = Dense(64, activation='relu', name='x217fc2')(x217)
    x217 = Dropout(0.2)(x217)
    x217 = Dense(2, activation='softmax', name='x217_out')(x217)

    x218 = Dense(64, activation='relu', name='x218fc1')(x)
    x218 = Dropout(0.2)(x218)
    x218 = Dense(64, activation='relu', name='x218fc2')(x218)
    x218 = Dropout(0.2)(x218)
    x218 = Dense(2, activation='softmax', name='x218_out')(x218)

    x219 = Dense(64, activation='relu', name='x219fc1')(x)
    x219 = Dropout(0.2)(x219)
    x219 = Dense(64, activation='relu', name='x219fc2')(x219)
    x219 = Dropout(0.2)(x219)
    x219 = Dense(2, activation='softmax', name='x219_out')(x219)

    x220 = Dense(64, activation='relu', name='x220fc1')(x)
    x220 = Dropout(0.2)(x220)
    x220 = Dense(64, activation='relu', name='x220fc2')(x220)
    x220 = Dropout(0.2)(x220)
    x220 = Dense(2, activation='softmax', name='x220_out')(x220)

    x221 = Dense(64, activation='relu', name='x221fc1')(x)
    x221 = Dropout(0.2)(x221)
    x221 = Dense(64, activation='relu', name='x221fc2')(x221)
    x221 = Dropout(0.2)(x221)
    x221 = Dense(2, activation='softmax', name='x221_out')(x221)

    x222 = Dense(64, activation='relu', name='x222fc1')(x)
    x222 = Dropout(0.2)(x222)
    x222 = Dense(64, activation='relu', name='x222fc2')(x222)
    x222 = Dropout(0.2)(x222)
    x222 = Dense(2, activation='softmax', name='x222_out')(x222)

    x223 = Dense(64, activation='relu', name='x223fc1')(x)
    x223 = Dropout(0.2)(x223)
    x223 = Dense(64, activation='relu', name='x223fc2')(x223)
    x223 = Dropout(0.2)(x223)
    x223 = Dense(2, activation='softmax', name='x223_out')(x223)

    x224 = Dense(64, activation='relu', name='x224fc1')(x)
    x224 = Dropout(0.2)(x224)
    x224 = Dense(64, activation='relu', name='x224fc2')(x224)
    x224 = Dropout(0.2)(x224)
    x224 = Dense(2, activation='softmax', name='x224_out')(x224)

    x225 = Dense(64, activation='relu', name='x225fc1')(x)
    x225 = Dropout(0.2)(x225)
    x225 = Dense(64, activation='relu', name='x225fc2')(x225)
    x225 = Dropout(0.2)(x225)
    x225 = Dense(2, activation='softmax', name='x225_out')(x225)

    x226 = Dense(64, activation='relu', name='x226fc1')(x)
    x226 = Dropout(0.2)(x226)
    x226 = Dense(64, activation='relu', name='x226fc2')(x226)
    x226 = Dropout(0.2)(x226)
    x226 = Dense(2, activation='softmax', name='x226_out')(x226)

    x227 = Dense(64, activation='relu', name='x227fc1')(x)
    x227 = Dropout(0.2)(x227)
    x227 = Dense(64, activation='relu', name='x227fc2')(x227)
    x227 = Dropout(0.2)(x227)
    x227 = Dense(2, activation='softmax', name='x227_out')(x227)

    x228 = Dense(64, activation='relu', name='x228fc1')(x)
    x228 = Dropout(0.2)(x228)
    x228 = Dense(64, activation='relu', name='x228fc2')(x228)
    x228 = Dropout(0.2)(x228)
    x228 = Dense(2, activation='softmax', name='x228_out')(x228)

    # Create model.
    model = Model(inputs=img_input, outputs=[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                                             x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                                             x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
                                             x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
                                             x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
                                             x51, x52, x53, x54, x55, x56, x57, x58, x59, x60,
                                             x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,
                                             x71, x72, x73, x74, x75, x76, x77, x78, x79, x80,
                                             x81, x82, x83, x84, x85, x86, x87, x88, x89, x90,
                                             x91, x92, x93, x94, x95, x96, x97, x98, x99, x100,
                                             x101, x102, x103, x104, x105, x106, x107, x108, x109, x110,
                                             x111, x112, x113, x114, x115, x116, x117, x118, x119, x120,
                                             x121, x122, x123, x124, x125, x126, x127, x128, x129, x130,
                                             x131, x132, x133, x134, x135, x136, x137, x138, x139, x140,
                                             x141, x142, x143, x144, x145, x146, x147, x148, x149, x150,
                                             x151, x152, x153, x154, x155, x156, x157, x158, x159, x160,
                                             x161, x162, x163, x164, x165, x166, x167, x168, x169, x170,
                                             x171, x172, x173, x174, x175, x176, x177, x178, x179, x180,
                                             x181, x182, x183, x184, x185, x186, x187, x188, x189, x190,
                                             x191, x192, x193, x194, x195, x196, x197, x198, x199, x200,
                                             x201, x202, x203, x204, x205, x206, x207, x208, x209, x210,
                                             x211, x212, x213, x214, x215, x216, x217, x218, x219, x220,
                                             x221, x222, x223, x224, x225, x226, x227, x228], name='vgg16')
    return model

# os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680

IMG_SIZE = 50

print('Loading Training Data...')
# train_data = np.load('../data/{}_pixel_train_multitask_data100000.npy'.format(IMG_SIZE))
train_data = np.load('../data/{}_pixel_train_multitask_data800000.npy'.format(IMG_SIZE))
print('Loading Val Data...')
val_data = np.load('../data/{}_pixel_val_multitask_data.npy'.format(IMG_SIZE))
#val_data = np.load('../data/{}_pixel_val_multitask_data200.npy'.format(IMG_SIZE))

train_data = train_data[:300]
val_data = val_data[:300]
print('Processing Data...')
x_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y1_train = np.array([i[1] for i in train_data])
y2_train = np.array([i[2] for i in train_data])
y3_train = np.array([i[3] for i in train_data])
y4_train = np.array([i[4] for i in train_data])
y5_train = np.array([i[5] for i in train_data])
y6_train = np.array([i[6] for i in train_data])
y7_train = np.array([i[7] for i in train_data])
y8_train = np.array([i[8] for i in train_data])
y9_train = np.array([i[9] for i in train_data])
y10_train = np.array([i[10] for i in train_data])
y11_train = np.array([i[11] for i in train_data])
y12_train = np.array([i[12] for i in train_data])
y13_train = np.array([i[13] for i in train_data])
y14_train = np.array([i[14] for i in train_data])
y15_train = np.array([i[15] for i in train_data])
y16_train = np.array([i[16] for i in train_data])
y17_train = np.array([i[17] for i in train_data])
y18_train = np.array([i[18] for i in train_data])
y19_train = np.array([i[19] for i in train_data])
y20_train = np.array([i[20] for i in train_data])
y21_train = np.array([i[21] for i in train_data])
y22_train = np.array([i[22] for i in train_data])
y23_train = np.array([i[23] for i in train_data])
y24_train = np.array([i[24] for i in train_data])
y25_train = np.array([i[25] for i in train_data])
y26_train = np.array([i[26] for i in train_data])
y27_train = np.array([i[27] for i in train_data])
y28_train = np.array([i[28] for i in train_data])
y29_train = np.array([i[29] for i in train_data])
y30_train = np.array([i[30] for i in train_data])
y31_train = np.array([i[31] for i in train_data])
y32_train = np.array([i[32] for i in train_data])
y33_train = np.array([i[33] for i in train_data])
y34_train = np.array([i[34] for i in train_data])
y35_train = np.array([i[35] for i in train_data])
y36_train = np.array([i[36] for i in train_data])
y37_train = np.array([i[37] for i in train_data])
y38_train = np.array([i[38] for i in train_data])
y39_train = np.array([i[39] for i in train_data])
y40_train = np.array([i[40] for i in train_data])
y41_train = np.array([i[41] for i in train_data])
y42_train = np.array([i[42] for i in train_data])
y43_train = np.array([i[43] for i in train_data])
y44_train = np.array([i[44] for i in train_data])
y45_train = np.array([i[45] for i in train_data])
y46_train = np.array([i[46] for i in train_data])
y47_train = np.array([i[47] for i in train_data])
y48_train = np.array([i[48] for i in train_data])
y49_train = np.array([i[49] for i in train_data])
y50_train = np.array([i[50] for i in train_data])
y51_train = np.array([i[51] for i in train_data])
y52_train = np.array([i[52] for i in train_data])
y53_train = np.array([i[53] for i in train_data])
y54_train = np.array([i[54] for i in train_data])
y55_train = np.array([i[55] for i in train_data])
y56_train = np.array([i[56] for i in train_data])
y57_train = np.array([i[57] for i in train_data])
y58_train = np.array([i[58] for i in train_data])
y59_train = np.array([i[59] for i in train_data])
y60_train = np.array([i[60] for i in train_data])
y61_train = np.array([i[61] for i in train_data])
y62_train = np.array([i[62] for i in train_data])
y63_train = np.array([i[63] for i in train_data])
y64_train = np.array([i[64] for i in train_data])
y65_train = np.array([i[65] for i in train_data])
y66_train = np.array([i[66] for i in train_data])
y67_train = np.array([i[67] for i in train_data])
y68_train = np.array([i[68] for i in train_data])
y69_train = np.array([i[69] for i in train_data])
y70_train = np.array([i[70] for i in train_data])
y71_train = np.array([i[71] for i in train_data])
y72_train = np.array([i[72] for i in train_data])
y73_train = np.array([i[73] for i in train_data])
y74_train = np.array([i[74] for i in train_data])
y75_train = np.array([i[75] for i in train_data])
y76_train = np.array([i[76] for i in train_data])
y77_train = np.array([i[77] for i in train_data])
y78_train = np.array([i[78] for i in train_data])
y79_train = np.array([i[79] for i in train_data])
y80_train = np.array([i[80] for i in train_data])
y81_train = np.array([i[81] for i in train_data])
y82_train = np.array([i[82] for i in train_data])
y83_train = np.array([i[83] for i in train_data])
y84_train = np.array([i[84] for i in train_data])
y85_train = np.array([i[85] for i in train_data])
y86_train = np.array([i[86] for i in train_data])
y87_train = np.array([i[87] for i in train_data])
y88_train = np.array([i[88] for i in train_data])
y89_train = np.array([i[89] for i in train_data])
y90_train = np.array([i[90] for i in train_data])
y91_train = np.array([i[91] for i in train_data])
y92_train = np.array([i[92] for i in train_data])
y93_train = np.array([i[93] for i in train_data])
y94_train = np.array([i[94] for i in train_data])
y95_train = np.array([i[95] for i in train_data])
y96_train = np.array([i[96] for i in train_data])
y97_train = np.array([i[97] for i in train_data])
y98_train = np.array([i[98] for i in train_data])
y99_train = np.array([i[99] for i in train_data])
y100_train = np.array([i[100] for i in train_data])
y101_train = np.array([i[101] for i in train_data])
y102_train = np.array([i[102] for i in train_data])
y103_train = np.array([i[103] for i in train_data])
y104_train = np.array([i[104] for i in train_data])
y105_train = np.array([i[105] for i in train_data])
y106_train = np.array([i[106] for i in train_data])
y107_train = np.array([i[107] for i in train_data])
y108_train = np.array([i[108] for i in train_data])
y109_train = np.array([i[109] for i in train_data])
y110_train = np.array([i[110] for i in train_data])
y111_train = np.array([i[111] for i in train_data])
y112_train = np.array([i[112] for i in train_data])
y113_train = np.array([i[113] for i in train_data])
y114_train = np.array([i[114] for i in train_data])
y115_train = np.array([i[115] for i in train_data])
y116_train = np.array([i[116] for i in train_data])
y117_train = np.array([i[117] for i in train_data])
y118_train = np.array([i[118] for i in train_data])
y119_train = np.array([i[119] for i in train_data])
y120_train = np.array([i[120] for i in train_data])
y121_train = np.array([i[121] for i in train_data])
y122_train = np.array([i[122] for i in train_data])
y123_train = np.array([i[123] for i in train_data])
y124_train = np.array([i[124] for i in train_data])
y125_train = np.array([i[125] for i in train_data])
y126_train = np.array([i[126] for i in train_data])
y127_train = np.array([i[127] for i in train_data])
y128_train = np.array([i[128] for i in train_data])
y129_train = np.array([i[129] for i in train_data])
y130_train = np.array([i[130] for i in train_data])
y131_train = np.array([i[131] for i in train_data])
y132_train = np.array([i[132] for i in train_data])
y133_train = np.array([i[133] for i in train_data])
y134_train = np.array([i[134] for i in train_data])
y135_train = np.array([i[135] for i in train_data])
y136_train = np.array([i[136] for i in train_data])
y137_train = np.array([i[137] for i in train_data])
y138_train = np.array([i[138] for i in train_data])
y139_train = np.array([i[139] for i in train_data])
y140_train = np.array([i[140] for i in train_data])
y141_train = np.array([i[141] for i in train_data])
y142_train = np.array([i[142] for i in train_data])
y143_train = np.array([i[143] for i in train_data])
y144_train = np.array([i[144] for i in train_data])
y145_train = np.array([i[145] for i in train_data])
y146_train = np.array([i[146] for i in train_data])
y147_train = np.array([i[147] for i in train_data])
y148_train = np.array([i[148] for i in train_data])
y149_train = np.array([i[149] for i in train_data])
y150_train = np.array([i[150] for i in train_data])
y151_train = np.array([i[151] for i in train_data])
y152_train = np.array([i[152] for i in train_data])
y153_train = np.array([i[153] for i in train_data])
y154_train = np.array([i[154] for i in train_data])
y155_train = np.array([i[155] for i in train_data])
y156_train = np.array([i[156] for i in train_data])
y157_train = np.array([i[157] for i in train_data])
y158_train = np.array([i[158] for i in train_data])
y159_train = np.array([i[159] for i in train_data])
y160_train = np.array([i[160] for i in train_data])
y161_train = np.array([i[161] for i in train_data])
y162_train = np.array([i[162] for i in train_data])
y163_train = np.array([i[163] for i in train_data])
y164_train = np.array([i[164] for i in train_data])
y165_train = np.array([i[165] for i in train_data])
y166_train = np.array([i[166] for i in train_data])
y167_train = np.array([i[167] for i in train_data])
y168_train = np.array([i[168] for i in train_data])
y169_train = np.array([i[169] for i in train_data])
y170_train = np.array([i[170] for i in train_data])
y171_train = np.array([i[171] for i in train_data])
y172_train = np.array([i[172] for i in train_data])
y173_train = np.array([i[173] for i in train_data])
y174_train = np.array([i[174] for i in train_data])
y175_train = np.array([i[175] for i in train_data])
y176_train = np.array([i[176] for i in train_data])
y177_train = np.array([i[177] for i in train_data])
y178_train = np.array([i[178] for i in train_data])
y179_train = np.array([i[179] for i in train_data])
y180_train = np.array([i[180] for i in train_data])
y181_train = np.array([i[181] for i in train_data])
y182_train = np.array([i[182] for i in train_data])
y183_train = np.array([i[183] for i in train_data])
y184_train = np.array([i[184] for i in train_data])
y185_train = np.array([i[185] for i in train_data])
y186_train = np.array([i[186] for i in train_data])
y187_train = np.array([i[187] for i in train_data])
y188_train = np.array([i[188] for i in train_data])
y189_train = np.array([i[189] for i in train_data])
y190_train = np.array([i[190] for i in train_data])
y191_train = np.array([i[191] for i in train_data])
y192_train = np.array([i[192] for i in train_data])
y193_train = np.array([i[193] for i in train_data])
y194_train = np.array([i[194] for i in train_data])
y195_train = np.array([i[195] for i in train_data])
y196_train = np.array([i[196] for i in train_data])
y197_train = np.array([i[197] for i in train_data])
y198_train = np.array([i[198] for i in train_data])
y199_train = np.array([i[199] for i in train_data])
y200_train = np.array([i[200] for i in train_data])
y201_train = np.array([i[201] for i in train_data])
y202_train = np.array([i[202] for i in train_data])
y203_train = np.array([i[203] for i in train_data])
y204_train = np.array([i[204] for i in train_data])
y205_train = np.array([i[205] for i in train_data])
y206_train = np.array([i[206] for i in train_data])
y207_train = np.array([i[207] for i in train_data])
y208_train = np.array([i[208] for i in train_data])
y209_train = np.array([i[209] for i in train_data])
y210_train = np.array([i[200] for i in train_data])
y211_train = np.array([i[211] for i in train_data])
y212_train = np.array([i[212] for i in train_data])
y213_train = np.array([i[213] for i in train_data])
y214_train = np.array([i[214] for i in train_data])
y215_train = np.array([i[215] for i in train_data])
y216_train = np.array([i[216] for i in train_data])
y217_train = np.array([i[217] for i in train_data])
y218_train = np.array([i[218] for i in train_data])
y219_train = np.array([i[219] for i in train_data])
y220_train = np.array([i[210] for i in train_data])
y221_train = np.array([i[221] for i in train_data])
y222_train = np.array([i[222] for i in train_data])
y223_train = np.array([i[223] for i in train_data])
y224_train = np.array([i[224] for i in train_data])
y225_train = np.array([i[225] for i in train_data])
y226_train = np.array([i[226] for i in train_data])
y227_train = np.array([i[227] for i in train_data])
y228_train = np.array([i[228] for i in train_data])


x_test = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y1_test = np.array([i[1] for i in val_data])
y2_test = np.array([i[2] for i in val_data])
y3_test = np.array([i[3] for i in val_data])
y4_test = np.array([i[4] for i in val_data])
y5_test = np.array([i[5] for i in val_data])
y6_test = np.array([i[6] for i in val_data])
y7_test = np.array([i[7] for i in val_data])
y8_test = np.array([i[8] for i in val_data])
y9_test = np.array([i[9] for i in val_data])
y10_test = np.array([i[10] for i in val_data])
y11_test = np.array([i[11] for i in val_data])
y12_test = np.array([i[12] for i in val_data])
y13_test = np.array([i[13] for i in val_data])
y14_test = np.array([i[14] for i in val_data])
y15_test = np.array([i[15] for i in val_data])
y16_test = np.array([i[16] for i in val_data])
y17_test = np.array([i[17] for i in val_data])
y18_test = np.array([i[18] for i in val_data])
y19_test = np.array([i[19] for i in val_data])
y20_test = np.array([i[20] for i in val_data])
y21_test = np.array([i[21] for i in val_data])
y22_test = np.array([i[22] for i in val_data])
y23_test = np.array([i[23] for i in val_data])
y24_test = np.array([i[24] for i in val_data])
y25_test = np.array([i[25] for i in val_data])
y26_test = np.array([i[26] for i in val_data])
y27_test = np.array([i[27] for i in val_data])
y28_test = np.array([i[28] for i in val_data])
y29_test = np.array([i[29] for i in val_data])
y30_test = np.array([i[30] for i in val_data])
y31_test = np.array([i[31] for i in val_data])
y32_test = np.array([i[32] for i in val_data])
y33_test = np.array([i[33] for i in val_data])
y34_test = np.array([i[34] for i in val_data])
y35_test = np.array([i[35] for i in val_data])
y36_test = np.array([i[36] for i in val_data])
y37_test = np.array([i[37] for i in val_data])
y38_test = np.array([i[38] for i in val_data])
y39_test = np.array([i[39] for i in val_data])
y40_test = np.array([i[40] for i in val_data])
y41_test = np.array([i[41] for i in val_data])
y42_test = np.array([i[42] for i in val_data])
y43_test = np.array([i[43] for i in val_data])
y44_test = np.array([i[44] for i in val_data])
y45_test = np.array([i[45] for i in val_data])
y46_test = np.array([i[46] for i in val_data])
y47_test = np.array([i[47] for i in val_data])
y48_test = np.array([i[48] for i in val_data])
y49_test = np.array([i[49] for i in val_data])
y50_test = np.array([i[50] for i in val_data])
y51_test = np.array([i[51] for i in val_data])
y52_test = np.array([i[52] for i in val_data])
y53_test = np.array([i[53] for i in val_data])
y54_test = np.array([i[54] for i in val_data])
y55_test = np.array([i[55] for i in val_data])
y56_test = np.array([i[56] for i in val_data])
y57_test = np.array([i[57] for i in val_data])
y58_test = np.array([i[58] for i in val_data])
y59_test = np.array([i[59] for i in val_data])
y60_test = np.array([i[60] for i in val_data])
y61_test = np.array([i[61] for i in val_data])
y62_test = np.array([i[62] for i in val_data])
y63_test = np.array([i[63] for i in val_data])
y64_test = np.array([i[64] for i in val_data])
y65_test = np.array([i[65] for i in val_data])
y66_test = np.array([i[66] for i in val_data])
y67_test = np.array([i[67] for i in val_data])
y68_test = np.array([i[68] for i in val_data])
y69_test = np.array([i[69] for i in val_data])
y70_test = np.array([i[70] for i in val_data])
y71_test = np.array([i[71] for i in val_data])
y72_test = np.array([i[72] for i in val_data])
y73_test = np.array([i[73] for i in val_data])
y74_test = np.array([i[74] for i in val_data])
y75_test = np.array([i[75] for i in val_data])
y76_test = np.array([i[76] for i in val_data])
y77_test = np.array([i[77] for i in val_data])
y78_test = np.array([i[78] for i in val_data])
y79_test = np.array([i[79] for i in val_data])
y80_test = np.array([i[80] for i in val_data])
y81_test = np.array([i[81] for i in val_data])
y82_test = np.array([i[82] for i in val_data])
y83_test = np.array([i[83] for i in val_data])
y84_test = np.array([i[84] for i in val_data])
y85_test = np.array([i[85] for i in val_data])
y86_test = np.array([i[86] for i in val_data])
y87_test = np.array([i[87] for i in val_data])
y88_test = np.array([i[88] for i in val_data])
y89_test = np.array([i[89] for i in val_data])
y90_test = np.array([i[90] for i in val_data])
y91_test = np.array([i[91] for i in val_data])
y92_test = np.array([i[92] for i in val_data])
y93_test = np.array([i[93] for i in val_data])
y94_test = np.array([i[94] for i in val_data])
y95_test = np.array([i[95] for i in val_data])
y96_test = np.array([i[96] for i in val_data])
y97_test = np.array([i[97] for i in val_data])
y98_test = np.array([i[98] for i in val_data])
y99_test = np.array([i[99] for i in val_data])
y100_test = np.array([i[100] for i in val_data])
y101_test = np.array([i[101] for i in val_data])
y102_test = np.array([i[102] for i in val_data])
y103_test = np.array([i[103] for i in val_data])
y104_test = np.array([i[104] for i in val_data])
y105_test = np.array([i[105] for i in val_data])
y106_test = np.array([i[106] for i in val_data])
y107_test = np.array([i[107] for i in val_data])
y108_test = np.array([i[108] for i in val_data])
y109_test = np.array([i[109] for i in val_data])
y110_test = np.array([i[110] for i in val_data])
y111_test = np.array([i[111] for i in val_data])
y112_test = np.array([i[112] for i in val_data])
y113_test = np.array([i[113] for i in val_data])
y114_test = np.array([i[114] for i in val_data])
y115_test = np.array([i[115] for i in val_data])
y116_test = np.array([i[116] for i in val_data])
y117_test = np.array([i[117] for i in val_data])
y118_test = np.array([i[118] for i in val_data])
y119_test = np.array([i[119] for i in val_data])
y120_test = np.array([i[120] for i in val_data])
y121_test = np.array([i[121] for i in val_data])
y122_test = np.array([i[122] for i in val_data])
y123_test = np.array([i[123] for i in val_data])
y124_test = np.array([i[124] for i in val_data])
y125_test = np.array([i[125] for i in val_data])
y126_test = np.array([i[126] for i in val_data])
y127_test = np.array([i[127] for i in val_data])
y128_test = np.array([i[128] for i in val_data])
y129_test = np.array([i[129] for i in val_data])
y130_test = np.array([i[130] for i in val_data])
y131_test = np.array([i[131] for i in val_data])
y132_test = np.array([i[132] for i in val_data])
y133_test = np.array([i[133] for i in val_data])
y134_test = np.array([i[134] for i in val_data])
y135_test = np.array([i[135] for i in val_data])
y136_test = np.array([i[136] for i in val_data])
y137_test = np.array([i[137] for i in val_data])
y138_test = np.array([i[138] for i in val_data])
y139_test = np.array([i[139] for i in val_data])
y140_test = np.array([i[140] for i in val_data])
y141_test = np.array([i[141] for i in val_data])
y142_test = np.array([i[142] for i in val_data])
y143_test = np.array([i[143] for i in val_data])
y144_test = np.array([i[144] for i in val_data])
y145_test = np.array([i[145] for i in val_data])
y146_test = np.array([i[146] for i in val_data])
y147_test = np.array([i[147] for i in val_data])
y148_test = np.array([i[148] for i in val_data])
y149_test = np.array([i[149] for i in val_data])
y150_test = np.array([i[150] for i in val_data])
y151_test = np.array([i[151] for i in val_data])
y152_test = np.array([i[152] for i in val_data])
y153_test = np.array([i[153] for i in val_data])
y154_test = np.array([i[154] for i in val_data])
y155_test = np.array([i[155] for i in val_data])
y156_test = np.array([i[156] for i in val_data])
y157_test = np.array([i[157] for i in val_data])
y158_test = np.array([i[158] for i in val_data])
y159_test = np.array([i[159] for i in val_data])
y160_test = np.array([i[160] for i in val_data])
y161_test = np.array([i[161] for i in val_data])
y162_test = np.array([i[162] for i in val_data])
y163_test = np.array([i[163] for i in val_data])
y164_test = np.array([i[164] for i in val_data])
y165_test = np.array([i[165] for i in val_data])
y166_test = np.array([i[166] for i in val_data])
y167_test = np.array([i[167] for i in val_data])
y168_test = np.array([i[168] for i in val_data])
y169_test = np.array([i[169] for i in val_data])
y170_test = np.array([i[170] for i in val_data])
y171_test = np.array([i[171] for i in val_data])
y172_test = np.array([i[172] for i in val_data])
y173_test = np.array([i[173] for i in val_data])
y174_test = np.array([i[174] for i in val_data])
y175_test = np.array([i[175] for i in val_data])
y176_test = np.array([i[176] for i in val_data])
y177_test = np.array([i[177] for i in val_data])
y178_test = np.array([i[178] for i in val_data])
y179_test = np.array([i[179] for i in val_data])
y180_test = np.array([i[180] for i in val_data])
y181_test = np.array([i[181] for i in val_data])
y182_test = np.array([i[182] for i in val_data])
y183_test = np.array([i[183] for i in val_data])
y184_test = np.array([i[184] for i in val_data])
y185_test = np.array([i[185] for i in val_data])
y186_test = np.array([i[186] for i in val_data])
y187_test = np.array([i[187] for i in val_data])
y188_test = np.array([i[188] for i in val_data])
y189_test = np.array([i[189] for i in val_data])
y190_test = np.array([i[190] for i in val_data])
y191_test = np.array([i[191] for i in val_data])
y192_test = np.array([i[192] for i in val_data])
y193_test = np.array([i[193] for i in val_data])
y194_test = np.array([i[194] for i in val_data])
y195_test = np.array([i[195] for i in val_data])
y196_test = np.array([i[196] for i in val_data])
y197_test = np.array([i[197] for i in val_data])
y198_test = np.array([i[198] for i in val_data])
y199_test = np.array([i[199] for i in val_data])
y200_test = np.array([i[200] for i in val_data])
y201_test = np.array([i[201] for i in val_data])
y202_test = np.array([i[202] for i in val_data])
y203_test = np.array([i[203] for i in val_data])
y204_test = np.array([i[204] for i in val_data])
y205_test = np.array([i[205] for i in val_data])
y206_test = np.array([i[206] for i in val_data])
y207_test = np.array([i[207] for i in val_data])
y208_test = np.array([i[208] for i in val_data])
y209_test = np.array([i[209] for i in val_data])
y210_test = np.array([i[200] for i in val_data])
y211_test = np.array([i[211] for i in val_data])
y212_test = np.array([i[212] for i in val_data])
y213_test = np.array([i[213] for i in val_data])
y214_test = np.array([i[214] for i in val_data])
y215_test = np.array([i[215] for i in val_data])
y216_test = np.array([i[216] for i in val_data])
y217_test = np.array([i[217] for i in val_data])
y218_test = np.array([i[218] for i in val_data])
y219_test = np.array([i[219] for i in val_data])
y220_test = np.array([i[210] for i in val_data])
y221_test = np.array([i[221] for i in val_data])
y222_test = np.array([i[222] for i in val_data])
y223_test = np.array([i[223] for i in val_data])
y224_test = np.array([i[224] for i in val_data])
y225_test = np.array([i[225] for i in val_data])
y226_test = np.array([i[226] for i in val_data])
y227_test = np.array([i[227] for i in val_data])
y228_test = np.array([i[228] for i in val_data])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model_name = 'VGG16_{}'.format(IMG_SIZE)

model = VGG16()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),
              loss={'x1_out':'binary_crossentropy',
                    'x2_out':'binary_crossentropy',
                    'x3_out':'binary_crossentropy',
                    'x4_out':'binary_crossentropy',
                    'x5_out':'binary_crossentropy',
                    'x6_out':'binary_crossentropy',
                    'x7_out':'binary_crossentropy',
                    'x8_out':'binary_crossentropy',
                    'x9_out':'binary_crossentropy',
                    'x10_out':'binary_crossentropy',
                    'x11_out':'binary_crossentropy',
                    'x12_out':'binary_crossentropy',
                    'x13_out':'binary_crossentropy',
                    'x14_out':'binary_crossentropy',
                    'x15_out':'binary_crossentropy',
                    'x16_out':'binary_crossentropy',
                    'x17_out':'binary_crossentropy',
                    'x18_out':'binary_crossentropy',
                    'x19_out':'binary_crossentropy',
                    'x20_out':'binary_crossentropy',
                    'x21_out':'binary_crossentropy',
                    'x22_out':'binary_crossentropy',
                    'x23_out':'binary_crossentropy',
                    'x24_out':'binary_crossentropy',
                    'x25_out':'binary_crossentropy',
                    'x26_out':'binary_crossentropy',
                    'x27_out':'binary_crossentropy',
                    'x28_out':'binary_crossentropy',
                    'x29_out':'binary_crossentropy',
                    'x30_out':'binary_crossentropy',
                    'x31_out':'binary_crossentropy',
                    'x32_out':'binary_crossentropy',
                    'x33_out':'binary_crossentropy',
                    'x34_out':'binary_crossentropy',
                    'x35_out':'binary_crossentropy',
                    'x36_out':'binary_crossentropy',
                    'x37_out':'binary_crossentropy',
                    'x38_out':'binary_crossentropy',
                    'x39_out':'binary_crossentropy',
                    'x40_out':'binary_crossentropy',
                    'x41_out':'binary_crossentropy',
                    'x42_out':'binary_crossentropy',
                    'x43_out':'binary_crossentropy',
                    'x44_out':'binary_crossentropy',
                    'x45_out':'binary_crossentropy',
                    'x46_out':'binary_crossentropy',
                    'x47_out':'binary_crossentropy',
                    'x48_out':'binary_crossentropy',
                    'x49_out':'binary_crossentropy',
                    'x50_out':'binary_crossentropy',
                    'x51_out':'binary_crossentropy',
                    'x52_out':'binary_crossentropy',
                    'x53_out':'binary_crossentropy',
                    'x54_out':'binary_crossentropy',
                    'x55_out':'binary_crossentropy',
                    'x56_out':'binary_crossentropy',
                    'x57_out':'binary_crossentropy',
                    'x58_out':'binary_crossentropy',
                    'x59_out':'binary_crossentropy',
                    'x60_out':'binary_crossentropy',
                    'x61_out':'binary_crossentropy',
                    'x62_out':'binary_crossentropy',
                    'x63_out':'binary_crossentropy',
                    'x64_out':'binary_crossentropy',
                    'x65_out':'binary_crossentropy',
                    'x66_out':'binary_crossentropy',
                    'x67_out':'binary_crossentropy',
                    'x68_out':'binary_crossentropy',
                    'x69_out':'binary_crossentropy',
                    'x70_out':'binary_crossentropy',
                    'x71_out':'binary_crossentropy',
                    'x72_out':'binary_crossentropy',
                    'x73_out':'binary_crossentropy',
                    'x74_out':'binary_crossentropy',
                    'x75_out':'binary_crossentropy',
                    'x76_out':'binary_crossentropy',
                    'x77_out':'binary_crossentropy',
                    'x78_out':'binary_crossentropy',
                    'x79_out':'binary_crossentropy',
                    'x80_out':'binary_crossentropy',
                    'x81_out':'binary_crossentropy',
                    'x82_out':'binary_crossentropy',
                    'x83_out':'binary_crossentropy',
                    'x84_out':'binary_crossentropy',
                    'x85_out':'binary_crossentropy',
                    'x86_out':'binary_crossentropy',
                    'x87_out':'binary_crossentropy',
                    'x88_out':'binary_crossentropy',
                    'x89_out':'binary_crossentropy',
                    'x90_out':'binary_crossentropy',
                    'x91_out':'binary_crossentropy',
                    'x92_out':'binary_crossentropy',
                    'x93_out':'binary_crossentropy',
                    'x94_out':'binary_crossentropy',
                    'x95_out':'binary_crossentropy',
                    'x96_out':'binary_crossentropy',
                    'x97_out':'binary_crossentropy',
                    'x98_out':'binary_crossentropy',
                    'x99_out':'binary_crossentropy',
                    'x100_out':'binary_crossentropy',
                    'x101_out':'binary_crossentropy',
                    'x102_out':'binary_crossentropy',
                    'x103_out':'binary_crossentropy',
                    'x104_out':'binary_crossentropy',
                    'x105_out':'binary_crossentropy',
                    'x106_out':'binary_crossentropy',
                    'x107_out':'binary_crossentropy',
                    'x108_out':'binary_crossentropy',
                    'x109_out':'binary_crossentropy',
                    'x110_out':'binary_crossentropy',
                    'x111_out':'binary_crossentropy',
                    'x112_out':'binary_crossentropy',
                    'x113_out':'binary_crossentropy',
                    'x114_out':'binary_crossentropy',
                    'x115_out':'binary_crossentropy',
                    'x116_out':'binary_crossentropy',
                    'x117_out':'binary_crossentropy',
                    'x118_out':'binary_crossentropy',
                    'x119_out':'binary_crossentropy',
                    'x120_out':'binary_crossentropy',
                    'x121_out':'binary_crossentropy',
                    'x122_out':'binary_crossentropy',
                    'x123_out':'binary_crossentropy',
                    'x124_out':'binary_crossentropy',
                    'x125_out':'binary_crossentropy',
                    'x126_out':'binary_crossentropy',
                    'x127_out':'binary_crossentropy',
                    'x128_out':'binary_crossentropy',
                    'x129_out':'binary_crossentropy',
                    'x130_out':'binary_crossentropy',
                    'x131_out':'binary_crossentropy',
                    'x132_out':'binary_crossentropy',
                    'x133_out':'binary_crossentropy',
                    'x134_out':'binary_crossentropy',
                    'x135_out':'binary_crossentropy',
                    'x136_out':'binary_crossentropy',
                    'x137_out':'binary_crossentropy',
                    'x138_out':'binary_crossentropy',
                    'x139_out':'binary_crossentropy',
                    'x140_out':'binary_crossentropy',
                    'x141_out':'binary_crossentropy',
                    'x142_out':'binary_crossentropy',
                    'x143_out':'binary_crossentropy',
                    'x144_out':'binary_crossentropy',
                    'x145_out':'binary_crossentropy',
                    'x146_out':'binary_crossentropy',
                    'x147_out':'binary_crossentropy',
                    'x148_out':'binary_crossentropy',
                    'x149_out':'binary_crossentropy',
                    'x150_out':'binary_crossentropy',
                    'x151_out':'binary_crossentropy',
                    'x152_out':'binary_crossentropy',
                    'x153_out':'binary_crossentropy',
                    'x154_out':'binary_crossentropy',
                    'x155_out':'binary_crossentropy',
                    'x156_out':'binary_crossentropy',
                    'x157_out':'binary_crossentropy',
                    'x158_out':'binary_crossentropy',
                    'x159_out':'binary_crossentropy',
                    'x160_out':'binary_crossentropy',
                    'x161_out':'binary_crossentropy',
                    'x162_out':'binary_crossentropy',
                    'x163_out':'binary_crossentropy',
                    'x164_out':'binary_crossentropy',
                    'x165_out':'binary_crossentropy',
                    'x166_out':'binary_crossentropy',
                    'x167_out':'binary_crossentropy',
                    'x168_out':'binary_crossentropy',
                    'x169_out':'binary_crossentropy',
                    'x170_out':'binary_crossentropy',
                    'x171_out':'binary_crossentropy',
                    'x172_out':'binary_crossentropy',
                    'x173_out':'binary_crossentropy',
                    'x174_out':'binary_crossentropy',
                    'x175_out':'binary_crossentropy',
                    'x176_out':'binary_crossentropy',
                    'x177_out':'binary_crossentropy',
                    'x178_out':'binary_crossentropy',
                    'x179_out':'binary_crossentropy',
                    'x180_out':'binary_crossentropy',
                    'x181_out':'binary_crossentropy',
                    'x182_out':'binary_crossentropy',
                    'x183_out':'binary_crossentropy',
                    'x184_out':'binary_crossentropy',
                    'x185_out':'binary_crossentropy',
                    'x186_out':'binary_crossentropy',
                    'x187_out':'binary_crossentropy',
                    'x188_out':'binary_crossentropy',
                    'x189_out':'binary_crossentropy',
                    'x190_out':'binary_crossentropy',
                    'x191_out':'binary_crossentropy',
                    'x192_out':'binary_crossentropy',
                    'x193_out':'binary_crossentropy',
                    'x194_out':'binary_crossentropy',
                    'x195_out':'binary_crossentropy',
                    'x196_out':'binary_crossentropy',
                    'x197_out':'binary_crossentropy',
                    'x198_out':'binary_crossentropy',
                    'x199_out':'binary_crossentropy',
                    'x200_out':'binary_crossentropy',
                    'x201_out':'binary_crossentropy',
                    'x202_out':'binary_crossentropy',
                    'x203_out':'binary_crossentropy',
                    'x204_out':'binary_crossentropy',
                    'x205_out':'binary_crossentropy',
                    'x206_out':'binary_crossentropy',
                    'x207_out':'binary_crossentropy',
                    'x208_out':'binary_crossentropy',
                    'x209_out':'binary_crossentropy',
                    'x210_out':'binary_crossentropy',
                    'x211_out':'binary_crossentropy',
                    'x212_out':'binary_crossentropy',
                    'x213_out':'binary_crossentropy',
                    'x214_out':'binary_crossentropy',
                    'x215_out':'binary_crossentropy',
                    'x216_out':'binary_crossentropy',
                    'x217_out':'binary_crossentropy',
                    'x218_out':'binary_crossentropy',
                    'x219_out':'binary_crossentropy',
                    'x220_out':'binary_crossentropy',
                    'x221_out':'binary_crossentropy',
                    'x222_out':'binary_crossentropy',
                    'x223_out':'binary_crossentropy',
                    'x224_out':'binary_crossentropy',
                    'x225_out':'binary_crossentropy',
                    'x226_out':'binary_crossentropy',
                    'x227_out':'binary_crossentropy',
                    'x228_out':'binary_crossentropy'},
              loss_weights={'x1_out':1,
                    'x2_out':1,
                    'x3_out':1,
                    'x4_out':1,
                    'x5_out':1,
                    'x6_out':1,
                    'x7_out':1,
                    'x8_out':1,
                    'x9_out':1,
                    'x10_out':1,
                    'x11_out':1,
                    'x12_out':1,
                    'x13_out':1,
                    'x14_out':1,
                    'x15_out':1,
                    'x16_out':1,
                    'x17_out':1,
                    'x18_out':1,
                    'x19_out':1,
                    'x20_out':1,
                    'x21_out':1,
                    'x22_out':1,
                    'x23_out':1,
                    'x24_out':1,
                    'x25_out':1,
                    'x26_out':1,
                    'x27_out':1,
                    'x28_out':1,
                    'x29_out':1,
                    'x30_out':1,
                    'x31_out':1,
                    'x32_out':1,
                    'x33_out':1,
                    'x34_out':1,
                    'x35_out':1,
                    'x36_out':1,
                    'x37_out':1,
                    'x38_out':1,
                    'x39_out':1,
                    'x40_out':1,
                    'x41_out':1,
                    'x42_out':1,
                    'x43_out':1,
                    'x44_out':1,
                    'x45_out':1,
                    'x46_out':1,
                    'x47_out':1,
                    'x48_out':1,
                    'x49_out':1,
                    'x50_out':1,
                    'x51_out':1,
                    'x52_out':1,
                    'x53_out':1,
                    'x54_out':1,
                    'x55_out':1,
                    'x56_out':1,
                    'x57_out':1,
                    'x58_out':1,
                    'x59_out':1,
                    'x60_out':1,
                    'x61_out':1,
                    'x62_out':1,
                    'x63_out':1,
                    'x64_out':1,
                    'x65_out':1,
                    'x66_out':1,
                    'x67_out':1,
                    'x68_out':1,
                    'x69_out':1,
                    'x70_out':1,
                    'x71_out':1,
                    'x72_out':1,
                    'x73_out':1,
                    'x74_out':1,
                    'x75_out':1,
                    'x76_out':1,
                    'x77_out':1,
                    'x78_out':1,
                    'x79_out':1,
                    'x80_out':1,
                    'x81_out':1,
                    'x82_out':1,
                    'x83_out':1,
                    'x84_out':1,
                    'x85_out':1,
                    'x86_out':1,
                    'x87_out':1,
                    'x88_out':1,
                    'x89_out':1,
                    'x90_out':1,
                    'x91_out':1,
                    'x92_out':1,
                    'x93_out':1,
                    'x94_out':1,
                    'x95_out':1,
                    'x96_out':1,
                    'x97_out':1,
                    'x98_out':1,
                    'x99_out':1,
                    'x100_out':1,
                    'x101_out':1,
                    'x102_out':1,
                    'x103_out':1,
                    'x104_out':1,
                    'x105_out':1,
                    'x106_out':1,
                    'x107_out':1,
                    'x108_out':1,
                    'x109_out':1,
                    'x110_out':1,
                    'x111_out':1,
                    'x112_out':1,
                    'x113_out':1,
                    'x114_out':1,
                    'x115_out':1,
                    'x116_out':1,
                    'x117_out':1,
                    'x118_out':1,
                    'x119_out':1,
                    'x120_out':1,
                    'x121_out':1,
                    'x122_out':1,
                    'x123_out':1,
                    'x124_out':1,
                    'x125_out':1,
                    'x126_out':1,
                    'x127_out':1,
                    'x128_out':1,
                    'x129_out':1,
                    'x130_out':1,
                    'x131_out':1,
                    'x132_out':1,
                    'x133_out':1,
                    'x134_out':1,
                    'x135_out':1,
                    'x136_out':1,
                    'x137_out':1,
                    'x138_out':1,
                    'x139_out':1,
                    'x140_out':1,
                    'x141_out':1,
                    'x142_out':1,
                    'x143_out':1,
                    'x144_out':1,
                    'x145_out':1,
                    'x146_out':1,
                    'x147_out':1,
                    'x148_out':1,
                    'x149_out':1,
                    'x150_out':1,
                    'x151_out':1,
                    'x152_out':1,
                    'x153_out':1,
                    'x154_out':1,
                    'x155_out':1,
                    'x156_out':1,
                    'x157_out':1,
                    'x158_out':1,
                    'x159_out':1,
                    'x160_out':1,
                    'x161_out':1,
                    'x162_out':1,
                    'x163_out':1,
                    'x164_out':1,
                    'x165_out':1,
                    'x166_out':1,
                    'x167_out':1,
                    'x168_out':1,
                    'x169_out':1,
                    'x170_out':1,
                    'x171_out':1,
                    'x172_out':1,
                    'x173_out':1,
                    'x174_out':1,
                    'x175_out':1,
                    'x176_out':1,
                    'x177_out':1,
                    'x178_out':1,
                    'x179_out':1,
                    'x180_out':1,
                    'x181_out':1,
                    'x182_out':1,
                    'x183_out':1,
                    'x184_out':1,
                    'x185_out':1,
                    'x186_out':1,
                    'x187_out':1,
                    'x188_out':1,
                    'x189_out':1,
                    'x190_out':1,
                    'x191_out':1,
                    'x192_out':1,
                    'x193_out':1,
                    'x194_out':1,
                    'x195_out':1,
                    'x196_out':1,
                    'x197_out':1,
                    'x198_out':1,
                    'x199_out':1,
                    'x200_out':1,
                    'x201_out':1,
                    'x202_out':1,
                    'x203_out':1,
                    'x204_out':1,
                    'x205_out':1,
                    'x206_out':1,
                    'x207_out':1,
                    'x208_out':1,
                    'x209_out':1,
                    'x210_out':1,
                    'x211_out':1,
                    'x212_out':1,
                    'x213_out':1,
                    'x214_out':1,
                    'x215_out':1,
                    'x216_out':1,
                    'x217_out':1,
                    'x218_out':1,
                    'x219_out':1,
                    'x220_out':1,
                    'x221_out':1,
                    'x222_out':1,
                    'x223_out':1,
                    'x224_out':1,
                    'x225_out':1,
                    'x226_out':1,
                    'x227_out':1,
                    'x228_out':1},
              metrics={'x1_out':'binary_accuracy',
                    'x2_out':'binary_accuracy',
                    'x3_out':'binary_accuracy',
                    'x4_out':'binary_accuracy',
                    'x5_out':'binary_accuracy',
                    'x6_out':'binary_accuracy',
                    'x7_out':'binary_accuracy',
                    'x8_out':'binary_accuracy',
                    'x9_out':'binary_accuracy',
                    'x10_out':'binary_accuracy',
                    'x11_out':'binary_accuracy',
                    'x12_out':'binary_accuracy',
                    'x13_out':'binary_accuracy',
                    'x14_out':'binary_accuracy',
                    'x15_out':'binary_accuracy',
                    'x16_out':'binary_accuracy',
                    'x17_out':'binary_accuracy',
                    'x18_out':'binary_accuracy',
                    'x19_out':'binary_accuracy',
                    'x20_out':'binary_accuracy',
                    'x21_out':'binary_accuracy',
                    'x22_out':'binary_accuracy',
                    'x23_out':'binary_accuracy',
                    'x24_out':'binary_accuracy',
                    'x25_out':'binary_accuracy',
                    'x26_out':'binary_accuracy',
                    'x27_out':'binary_accuracy',
                    'x28_out':'binary_accuracy',
                    'x29_out':'binary_accuracy',
                    'x30_out':'binary_accuracy',
                    'x31_out':'binary_accuracy',
                    'x32_out':'binary_accuracy',
                    'x33_out':'binary_accuracy',
                    'x34_out':'binary_accuracy',
                    'x35_out':'binary_accuracy',
                    'x36_out':'binary_accuracy',
                    'x37_out':'binary_accuracy',
                    'x38_out':'binary_accuracy',
                    'x39_out':'binary_accuracy',
                    'x40_out':'binary_accuracy',
                    'x41_out':'binary_accuracy',
                    'x42_out':'binary_accuracy',
                    'x43_out':'binary_accuracy',
                    'x44_out':'binary_accuracy',
                    'x45_out':'binary_accuracy',
                    'x46_out':'binary_accuracy',
                    'x47_out':'binary_accuracy',
                    'x48_out':'binary_accuracy',
                    'x49_out':'binary_accuracy',
                    'x50_out':'binary_accuracy',
                    'x51_out':'binary_accuracy',
                    'x52_out':'binary_accuracy',
                    'x53_out':'binary_accuracy',
                    'x54_out':'binary_accuracy',
                    'x55_out':'binary_accuracy',
                    'x56_out':'binary_accuracy',
                    'x57_out':'binary_accuracy',
                    'x58_out':'binary_accuracy',
                    'x59_out':'binary_accuracy',
                    'x60_out':'binary_accuracy',
                    'x61_out':'binary_accuracy',
                    'x62_out':'binary_accuracy',
                    'x63_out':'binary_accuracy',
                    'x64_out':'binary_accuracy',
                    'x65_out':'binary_accuracy',
                    'x66_out':'binary_accuracy',
                    'x67_out':'binary_accuracy',
                    'x68_out':'binary_accuracy',
                    'x69_out':'binary_accuracy',
                    'x70_out':'binary_accuracy',
                    'x71_out':'binary_accuracy',
                    'x72_out':'binary_accuracy',
                    'x73_out':'binary_accuracy',
                    'x74_out':'binary_accuracy',
                    'x75_out':'binary_accuracy',
                    'x76_out':'binary_accuracy',
                    'x77_out':'binary_accuracy',
                    'x78_out':'binary_accuracy',
                    'x79_out':'binary_accuracy',
                    'x80_out':'binary_accuracy',
                    'x81_out':'binary_accuracy',
                    'x82_out':'binary_accuracy',
                    'x83_out':'binary_accuracy',
                    'x84_out':'binary_accuracy',
                    'x85_out':'binary_accuracy',
                    'x86_out':'binary_accuracy',
                    'x87_out':'binary_accuracy',
                    'x88_out':'binary_accuracy',
                    'x89_out':'binary_accuracy',
                    'x90_out':'binary_accuracy',
                    'x91_out':'binary_accuracy',
                    'x92_out':'binary_accuracy',
                    'x93_out':'binary_accuracy',
                    'x94_out':'binary_accuracy',
                    'x95_out':'binary_accuracy',
                    'x96_out':'binary_accuracy',
                    'x97_out':'binary_accuracy',
                    'x98_out':'binary_accuracy',
                    'x99_out':'binary_accuracy',
                    'x100_out':'binary_accuracy',
                    'x101_out':'binary_accuracy',
                    'x102_out':'binary_accuracy',
                    'x103_out':'binary_accuracy',
                    'x104_out':'binary_accuracy',
                    'x105_out':'binary_accuracy',
                    'x106_out':'binary_accuracy',
                    'x107_out':'binary_accuracy',
                    'x108_out':'binary_accuracy',
                    'x109_out':'binary_accuracy',
                    'x110_out':'binary_accuracy',
                    'x111_out':'binary_accuracy',
                    'x112_out':'binary_accuracy',
                    'x113_out':'binary_accuracy',
                    'x114_out':'binary_accuracy',
                    'x115_out':'binary_accuracy',
                    'x116_out':'binary_accuracy',
                    'x117_out':'binary_accuracy',
                    'x118_out':'binary_accuracy',
                    'x119_out':'binary_accuracy',
                    'x120_out':'binary_accuracy',
                    'x121_out':'binary_accuracy',
                    'x122_out':'binary_accuracy',
                    'x123_out':'binary_accuracy',
                    'x124_out':'binary_accuracy',
                    'x125_out':'binary_accuracy',
                    'x126_out':'binary_accuracy',
                    'x127_out':'binary_accuracy',
                    'x128_out':'binary_accuracy',
                    'x129_out':'binary_accuracy',
                    'x130_out':'binary_accuracy',
                    'x131_out':'binary_accuracy',
                    'x132_out':'binary_accuracy',
                    'x133_out':'binary_accuracy',
                    'x134_out':'binary_accuracy',
                    'x135_out':'binary_accuracy',
                    'x136_out':'binary_accuracy',
                    'x137_out':'binary_accuracy',
                    'x138_out':'binary_accuracy',
                    'x139_out':'binary_accuracy',
                    'x140_out':'binary_accuracy',
                    'x141_out':'binary_accuracy',
                    'x142_out':'binary_accuracy',
                    'x143_out':'binary_accuracy',
                    'x144_out':'binary_accuracy',
                    'x145_out':'binary_accuracy',
                    'x146_out':'binary_accuracy',
                    'x147_out':'binary_accuracy',
                    'x148_out':'binary_accuracy',
                    'x149_out':'binary_accuracy',
                    'x150_out':'binary_accuracy',
                    'x151_out':'binary_accuracy',
                    'x152_out':'binary_accuracy',
                    'x153_out':'binary_accuracy',
                    'x154_out':'binary_accuracy',
                    'x155_out':'binary_accuracy',
                    'x156_out':'binary_accuracy',
                    'x157_out':'binary_accuracy',
                    'x158_out':'binary_accuracy',
                    'x159_out':'binary_accuracy',
                    'x160_out':'binary_accuracy',
                    'x161_out':'binary_accuracy',
                    'x162_out':'binary_accuracy',
                    'x163_out':'binary_accuracy',
                    'x164_out':'binary_accuracy',
                    'x165_out':'binary_accuracy',
                    'x166_out':'binary_accuracy',
                    'x167_out':'binary_accuracy',
                    'x168_out':'binary_accuracy',
                    'x169_out':'binary_accuracy',
                    'x170_out':'binary_accuracy',
                    'x171_out':'binary_accuracy',
                    'x172_out':'binary_accuracy',
                    'x173_out':'binary_accuracy',
                    'x174_out':'binary_accuracy',
                    'x175_out':'binary_accuracy',
                    'x176_out':'binary_accuracy',
                    'x177_out':'binary_accuracy',
                    'x178_out':'binary_accuracy',
                    'x179_out':'binary_accuracy',
                    'x180_out':'binary_accuracy',
                    'x181_out':'binary_accuracy',
                    'x182_out':'binary_accuracy',
                    'x183_out':'binary_accuracy',
                    'x184_out':'binary_accuracy',
                    'x185_out':'binary_accuracy',
                    'x186_out':'binary_accuracy',
                    'x187_out':'binary_accuracy',
                    'x188_out':'binary_accuracy',
                    'x189_out':'binary_accuracy',
                    'x190_out':'binary_accuracy',
                    'x191_out':'binary_accuracy',
                    'x192_out':'binary_accuracy',
                    'x193_out':'binary_accuracy',
                    'x194_out':'binary_accuracy',
                    'x195_out':'binary_accuracy',
                    'x196_out':'binary_accuracy',
                    'x197_out':'binary_accuracy',
                    'x198_out':'binary_accuracy',
                    'x199_out':'binary_accuracy',
                    'x200_out':'binary_accuracy',
                    'x201_out':'binary_accuracy',
                    'x202_out':'binary_accuracy',
                    'x203_out':'binary_accuracy',
                    'x204_out':'binary_accuracy',
                    'x205_out':'binary_accuracy',
                    'x206_out':'binary_accuracy',
                    'x207_out':'binary_accuracy',
                    'x208_out':'binary_accuracy',
                    'x209_out':'binary_accuracy',
                    'x210_out':'binary_accuracy',
                    'x211_out':'binary_accuracy',
                    'x212_out':'binary_accuracy',
                    'x213_out':'binary_accuracy',
                    'x214_out':'binary_accuracy',
                    'x215_out':'binary_accuracy',
                    'x216_out':'binary_accuracy',
                    'x217_out':'binary_accuracy',
                    'x218_out':'binary_accuracy',
                    'x219_out':'binary_accuracy',
                    'x220_out':'binary_accuracy',
                    'x221_out':'binary_accuracy',
                    'x222_out':'binary_accuracy',
                    'x223_out':'binary_accuracy',
                    'x224_out':'binary_accuracy',
                    'x225_out':'binary_accuracy',
                    'x226_out':'binary_accuracy',
                    'x227_out':'binary_accuracy',
                    'x228_out':'binary_accuracy'})


# gen = ImageDataGenerator(horizontal_flip = True,
#                          width_shift_range = 0.1,
#                          height_shift_range = 0.1,
#                          zoom_range = 0.1)
#
#
# def gen_flow_for_two_inputs(X1, X2, y):
#     genX1 = gen.flow(X1,y,  batch_size=128)
#     genX2 = gen.flow(X1,X2, batch_size=128)
#     while True:
#             X1i = genX1.next()
#             X2i = genX2.next()
#             #Assert arrays are equal - this was for peace of mind, but slows down training
#             #np.testing.assert_array_equal(X1i[0],X2i[0])
#             yield [X1i[0], X2i[1]], X1i[1]
#
# gen_flow = gen_flow_for_two_inputs(x1_train, x2_train, y_train)
#
# model.fit_generator(gen_flow,
#                     validation_data=([x1_val, x2_val], y_val),
#                     steps_per_epoch=x1_train.shape[0] / batch_size,
#                     epochs=250,
#                     callbacks=[  # EarlyStopping(min_delta=0.001, patience=3),
#                         ModelCheckpoint('VGG16_saved_models/1xdif-100/weights.{epoch:02d}-{val_acc:.3f}.hdf5', monitor='val_loss',
#                                         verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
#                                         period=1)])

#Train the model
model.fit(x_train, [y1_train,
                    y2_train,
                    y3_train,
                    y4_train,
                    y5_train,
                    y6_train,
                    y7_train,
                    y8_train,
                    y9_train,
                    y10_train,
                    y11_train,
                    y12_train,
                    y13_train,
                    y14_train,
                    y15_train,
                    y16_train,
                    y17_train,
                    y18_train,
                    y19_train,
                    y20_train,
                    y21_train,
                    y22_train,
                    y23_train,
                    y24_train,
                    y25_train,
                    y26_train,
                    y27_train,
                    y28_train,
                    y29_train,
                    y30_train,
                    y31_train,
                    y32_train,
                    y33_train,
                    y34_train,
                    y35_train,
                    y36_train,
                    y37_train,
                    y38_train,
                    y39_train,
                    y40_train,
                    y41_train,
                    y42_train,
                    y43_train,
                    y44_train,
                    y45_train,
                    y46_train,
                    y47_train,
                    y48_train,
                    y49_train,
                    y50_train,
                    y51_train,
                    y52_train,
                    y53_train,
                    y54_train,
                    y55_train,
                    y56_train,
                    y57_train,
                    y58_train,
                    y59_train,
                    y60_train,
                    y61_train,
                    y62_train,
                    y63_train,
                    y64_train,
                    y65_train,
                    y66_train,
                    y67_train,
                    y68_train,
                    y69_train,
                    y70_train,
                    y71_train,
                    y72_train,
                    y73_train,
                    y74_train,
                    y75_train,
                    y76_train,
                    y77_train,
                    y78_train,
                    y79_train,
                    y80_train,
                    y81_train,
                    y82_train,
                    y83_train,
                    y84_train,
                    y85_train,
                    y86_train,
                    y87_train,
                    y88_train,
                    y89_train,
                    y90_train,
                    y91_train,
                    y92_train,
                    y93_train,
                    y94_train,
                    y95_train,
                    y96_train,
                    y97_train,
                    y98_train,
                    y99_train,
                    y100_train,
                    y101_train,
                    y102_train,
                    y103_train,
                    y104_train,
                    y105_train,
                    y106_train,
                    y107_train,
                    y108_train,
                    y109_train,
                    y110_train,
                    y111_train,
                    y112_train,
                    y113_train,
                    y114_train,
                    y115_train,
                    y116_train,
                    y117_train,
                    y118_train,
                    y119_train,
                    y120_train,
                    y121_train,
                    y122_train,
                    y123_train,
                    y124_train,
                    y125_train,
                    y126_train,
                    y127_train,
                    y128_train,
                    y129_train,
                    y130_train,
                    y131_train,
                    y132_train,
                    y133_train,
                    y134_train,
                    y135_train,
                    y136_train,
                    y137_train,
                    y138_train,
                    y139_train,
                    y140_train,
                    y141_train,
                    y142_train,
                    y143_train,
                    y144_train,
                    y145_train,
                    y146_train,
                    y147_train,
                    y148_train,
                    y149_train,
                    y150_train,
                    y151_train,
                    y152_train,
                    y153_train,
                    y154_train,
                    y155_train,
                    y156_train,
                    y157_train,
                    y158_train,
                    y159_train,
                    y160_train,
                    y161_train,
                    y162_train,
                    y163_train,
                    y164_train,
                    y165_train,
                    y166_train,
                    y167_train,
                    y168_train,
                    y169_train,
                    y170_train,
                    y171_train,
                    y172_train,
                    y173_train,
                    y174_train,
                    y175_train,
                    y176_train,
                    y177_train,
                    y178_train,
                    y179_train,
                    y180_train,
                    y181_train,
                    y182_train,
                    y183_train,
                    y184_train,
                    y185_train,
                    y186_train,
                    y187_train,
                    y188_train,
                    y189_train,
                    y190_train,
                    y191_train,
                    y192_train,
                    y193_train,
                    y194_train,
                    y195_train,
                    y196_train,
                    y197_train,
                    y198_train,
                    y199_train,
                    y200_train,
                    y201_train,
                    y202_train,
                    y203_train,
                    y204_train,
                    y205_train,
                    y206_train,
                    y207_train,
                    y208_train,
                    y209_train,
                    y210_train,
                    y211_train,
                    y212_train,
                    y213_train,
                    y214_train,
                    y215_train,
                    y216_train,
                    y217_train,
                    y218_train,
                    y219_train,
                    y220_train,
                    y221_train,
                    y222_train,
                    y223_train,
                    y224_train,
                    y225_train,
                    y226_train,
                    y227_train,
                    y228_train],
          batch_size=128,
          shuffle=True,
          epochs=150,
          validation_data=(x_test, [y1_test,
                    y2_test,
                    y3_test,
                    y4_test,
                    y5_test,
                    y6_test,
                    y7_test,
                    y8_test,
                    y9_test,
                    y10_test,
                    y11_test,
                    y12_test,
                    y13_test,
                    y14_test,
                    y15_test,
                    y16_test,
                    y17_test,
                    y18_test,
                    y19_test,
                    y20_test,
                    y21_test,
                    y22_test,
                    y23_test,
                    y24_test,
                    y25_test,
                    y26_test,
                    y27_test,
                    y28_test,
                    y29_test,
                    y30_test,
                    y31_test,
                    y32_test,
                    y33_test,
                    y34_test,
                    y35_test,
                    y36_test,
                    y37_test,
                    y38_test,
                    y39_test,
                    y40_test,
                    y41_test,
                    y42_test,
                    y43_test,
                    y44_test,
                    y45_test,
                    y46_test,
                    y47_test,
                    y48_test,
                    y49_test,
                    y50_test,
                    y51_test,
                    y52_test,
                    y53_test,
                    y54_test,
                    y55_test,
                    y56_test,
                    y57_test,
                    y58_test,
                    y59_test,
                    y60_test,
                    y61_test,
                    y62_test,
                    y63_test,
                    y64_test,
                    y65_test,
                    y66_test,
                    y67_test,
                    y68_test,
                    y69_test,
                    y70_test,
                    y71_test,
                    y72_test,
                    y73_test,
                    y74_test,
                    y75_test,
                    y76_test,
                    y77_test,
                    y78_test,
                    y79_test,
                    y80_test,
                    y81_test,
                    y82_test,
                    y83_test,
                    y84_test,
                    y85_test,
                    y86_test,
                    y87_test,
                    y88_test,
                    y89_test,
                    y90_test,
                    y91_test,
                    y92_test,
                    y93_test,
                    y94_test,
                    y95_test,
                    y96_test,
                    y97_test,
                    y98_test,
                    y99_test,
                    y100_test,
                    y101_test,
                    y102_test,
                    y103_test,
                    y104_test,
                    y105_test,
                    y106_test,
                    y107_test,
                    y108_test,
                    y109_test,
                    y110_test,
                    y111_test,
                    y112_test,
                    y113_test,
                    y114_test,
                    y115_test,
                    y116_test,
                    y117_test,
                    y118_test,
                    y119_test,
                    y120_test,
                    y121_test,
                    y122_test,
                    y123_test,
                    y124_test,
                    y125_test,
                    y126_test,
                    y127_test,
                    y128_test,
                    y129_test,
                    y130_test,
                    y131_test,
                    y132_test,
                    y133_test,
                    y134_test,
                    y135_test,
                    y136_test,
                    y137_test,
                    y138_test,
                    y139_test,
                    y140_test,
                    y141_test,
                    y142_test,
                    y143_test,
                    y144_test,
                    y145_test,
                    y146_test,
                    y147_test,
                    y148_test,
                    y149_test,
                    y150_test,
                    y151_test,
                    y152_test,
                    y153_test,
                    y154_test,
                    y155_test,
                    y156_test,
                    y157_test,
                    y158_test,
                    y159_test,
                    y160_test,
                    y161_test,
                    y162_test,
                    y163_test,
                    y164_test,
                    y165_test,
                    y166_test,
                    y167_test,
                    y168_test,
                    y169_test,
                    y170_test,
                    y171_test,
                    y172_test,
                    y173_test,
                    y174_test,
                    y175_test,
                    y176_test,
                    y177_test,
                    y178_test,
                    y179_test,
                    y180_test,
                    y181_test,
                    y182_test,
                    y183_test,
                    y184_test,
                    y185_test,
                    y186_test,
                    y187_test,
                    y188_test,
                    y189_test,
                    y190_test,
                    y191_test,
                    y192_test,
                    y193_test,
                    y194_test,
                    y195_test,
                    y196_test,
                    y197_test,
                    y198_test,
                    y199_test,
                    y200_test,
                    y201_test,
                    y202_test,
                    y203_test,
                    y204_test,
                    y205_test,
                    y206_test,
                    y207_test,
                    y208_test,
                    y209_test,
                    y210_test,
                    y211_test,
                    y212_test,
                    y213_test,
                    y214_test,
                    y215_test,
                    y216_test,
                    y217_test,
                    y218_test,
                    y219_test,
                    y220_test,
                    y221_test,
                    y222_test,
                    y223_test,
                    y224_test,
                    y225_test,
                    y226_test,
                    y227_test,
                    y228_test]),
          callbacks=[#EarlyStopping(min_delta=0.001, patience=3),
                     ModelCheckpoint('VGG16_saved_models/50/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                     verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)])

#Evaluate the model
scores = model.evaluate(x_test, [y1_test,
                    y2_test,
                    y3_test,
                    y4_test,
                    y5_test,
                    y6_test,
                    y7_test,
                    y8_test,
                    y9_test,
                    y10_test,
                    y11_test,
                    y12_test,
                    y13_test,
                    y14_test,
                    y15_test,
                    y16_test,
                    y17_test,
                    y18_test,
                    y19_test,
                    y20_test,
                    y21_test,
                    y22_test,
                    y23_test,
                    y24_test,
                    y25_test,
                    y26_test,
                    y27_test,
                    y28_test,
                    y29_test,
                    y30_test,
                    y31_test,
                    y32_test,
                    y33_test,
                    y34_test,
                    y35_test,
                    y36_test,
                    y37_test,
                    y38_test,
                    y39_test,
                    y40_test,
                    y41_test,
                    y42_test,
                    y43_test,
                    y44_test,
                    y45_test,
                    y46_test,
                    y47_test,
                    y48_test,
                    y49_test,
                    y50_test,
                    y51_test,
                    y52_test,
                    y53_test,
                    y54_test,
                    y55_test,
                    y56_test,
                    y57_test,
                    y58_test,
                    y59_test,
                    y60_test,
                    y61_test,
                    y62_test,
                    y63_test,
                    y64_test,
                    y65_test,
                    y66_test,
                    y67_test,
                    y68_test,
                    y69_test,
                    y70_test,
                    y71_test,
                    y72_test,
                    y73_test,
                    y74_test,
                    y75_test,
                    y76_test,
                    y77_test,
                    y78_test,
                    y79_test,
                    y80_test,
                    y81_test,
                    y82_test,
                    y83_test,
                    y84_test,
                    y85_test,
                    y86_test,
                    y87_test,
                    y88_test,
                    y89_test,
                    y90_test,
                    y91_test,
                    y92_test,
                    y93_test,
                    y94_test,
                    y95_test,
                    y96_test,
                    y97_test,
                    y98_test,
                    y99_test,
                    y100_test,
                    y101_test,
                    y102_test,
                    y103_test,
                    y104_test,
                    y105_test,
                    y106_test,
                    y107_test,
                    y108_test,
                    y109_test,
                    y110_test,
                    y111_test,
                    y112_test,
                    y113_test,
                    y114_test,
                    y115_test,
                    y116_test,
                    y117_test,
                    y118_test,
                    y119_test,
                    y120_test,
                    y121_test,
                    y122_test,
                    y123_test,
                    y124_test,
                    y125_test,
                    y126_test,
                    y127_test,
                    y128_test,
                    y129_test,
                    y130_test,
                    y131_test,
                    y132_test,
                    y133_test,
                    y134_test,
                    y135_test,
                    y136_test,
                    y137_test,
                    y138_test,
                    y139_test,
                    y140_test,
                    y141_test,
                    y142_test,
                    y143_test,
                    y144_test,
                    y145_test,
                    y146_test,
                    y147_test,
                    y148_test,
                    y149_test,
                    y150_test,
                    y151_test,
                    y152_test,
                    y153_test,
                    y154_test,
                    y155_test,
                    y156_test,
                    y157_test,
                    y158_test,
                    y159_test,
                    y160_test,
                    y161_test,
                    y162_test,
                    y163_test,
                    y164_test,
                    y165_test,
                    y166_test,
                    y167_test,
                    y168_test,
                    y169_test,
                    y170_test,
                    y171_test,
                    y172_test,
                    y173_test,
                    y174_test,
                    y175_test,
                    y176_test,
                    y177_test,
                    y178_test,
                    y179_test,
                    y180_test,
                    y181_test,
                    y182_test,
                    y183_test,
                    y184_test,
                    y185_test,
                    y186_test,
                    y187_test,
                    y188_test,
                    y189_test,
                    y190_test,
                    y191_test,
                    y192_test,
                    y193_test,
                    y194_test,
                    y195_test,
                    y196_test,
                    y197_test,
                    y198_test,
                    y199_test,
                    y200_test,
                    y201_test,
                    y202_test,
                    y203_test,
                    y204_test,
                    y205_test,
                    y206_test,
                    y207_test,
                    y208_test,
                    y209_test,
                    y210_test,
                    y211_test,
                    y212_test,
                    y213_test,
                    y214_test,
                    y215_test,
                    y216_test,
                    y217_test,
                    y218_test,
                    y219_test,
                    y220_test,
                    y221_test,
                    y222_test,
                    y223_test,
                    y224_test,
                    y225_test,
                    y226_test,
                    y227_test,
                    y228_test])

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