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
from keras.layers import Input, BatchNormalization
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
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x1 = Dense(64, activation='relu', name='x1fc1')(x)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(64, activation='relu', name='x1fc2')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(2, activation='softmax', name='x1_out')(x1)

    x2 = Dense(64, activation='relu', name='x2fc1')(x)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(64, activation='relu', name='x2fc2')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(2, activation='softmax', name='x2_out')(x2)

    x3 = Dense(64, activation='relu', name='x3fc1')(x)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(64, activation='relu', name='x3fc2')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(2, activation='softmax', name='x3_out')(x3)

    x4 = Dense(64, activation='relu', name='x4fc1')(x)
    x4 = Dropout(0.5)(x4)
    x4 = Dense(64, activation='relu', name='x4fc2')(x4)
    x4 = Dropout(0.5)(x4)
    x4 = Dense(2, activation='softmax', name='x4_out')(x4)

    x5 = Dense(64, activation='relu', name='x5fc1')(x)
    x5 = Dropout(0.5)(x5)
    x5 = Dense(64, activation='relu', name='x5fc2')(x5)
    x5 = Dropout(0.5)(x5)
    x5 = Dense(2, activation='softmax', name='x5_out')(x5)

    x6 = Dense(64, activation='relu', name='x6fc1')(x)
    x6 = Dropout(0.5)(x6)
    x6 = Dense(64, activation='relu', name='x6fc2')(x6)
    x6 = Dropout(0.5)(x6)
    x6 = Dense(2, activation='softmax', name='x6_out')(x6)

    x7 = Dense(64, activation='relu', name='x7fc1')(x)
    x7 = Dropout(0.5)(x7)
    x7 = Dense(64, activation='relu', name='x7fc2')(x7)
    x7 = Dropout(0.5)(x7)
    x7 = Dense(2, activation='softmax', name='x7_out')(x7)

    x8 = Dense(64, activation='relu', name='x8fc1')(x)
    x8 = Dropout(0.5)(x8)
    x8 = Dense(64, activation='relu', name='x8fc2')(x8)
    x8 = Dropout(0.5)(x8)
    x8 = Dense(2, activation='softmax', name='x8_out')(x8)

    x9 = Dense(64, activation='relu', name='x9fc1')(x)
    x9 = Dropout(0.5)(x9)
    x9 = Dense(64, activation='relu', name='x9fc2')(x9)
    x9 = Dropout(0.5)(x9)
    x9 = Dense(2, activation='softmax', name='x9_out')(x9)

    x10 = Dense(64, activation='relu', name='x10fc1')(x)
    x10 = Dropout(0.5)(x10)
    x10 = Dense(64, activation='relu', name='x10fc2')(x10)
    x10 = Dropout(0.5)(x10)
    x10 = Dense(2, activation='softmax', name='x10_out')(x10)

    x11 = Dense(64, activation='relu', name='x11fc1')(x)
    x11 = Dropout(0.5)(x11)
    x11 = Dense(64, activation='relu', name='x11fc2')(x11)
    x11 = Dropout(0.5)(x11)
    x11 = Dense(2, activation='softmax', name='x11_out')(x11)

    x12 = Dense(64, activation='relu', name='x12fc1')(x)
    x12 = Dropout(0.5)(x12)
    x12 = Dense(64, activation='relu', name='x12fc2')(x12)
    x12 = Dropout(0.5)(x12)
    x12 = Dense(2, activation='softmax', name='x12_out')(x12)

    x13 = Dense(64, activation='relu', name='x13fc1')(x)
    x13 = Dropout(0.5)(x13)
    x13 = Dense(64, activation='relu', name='x13fc2')(x13)
    x13 = Dropout(0.5)(x13)
    x13 = Dense(2, activation='softmax', name='x13_out')(x13)

    x14 = Dense(64, activation='relu', name='x14fc1')(x)
    x14 = Dropout(0.5)(x14)
    x14 = Dense(64, activation='relu', name='x14fc2')(x14)
    x14 = Dropout(0.5)(x14)
    x14 = Dense(2, activation='softmax', name='x14_out')(x14)

    x15 = Dense(64, activation='relu', name='x15fc1')(x)
    x15 = Dropout(0.5)(x15)
    x15 = Dense(64, activation='relu', name='x15fc2')(x15)
    x15 = Dropout(0.5)(x15)
    x15 = Dense(2, activation='softmax', name='x15_out')(x15)

    x16 = Dense(64, activation='relu', name='x16fc1')(x)
    x16 = Dropout(0.5)(x16)
    x16 = Dense(64, activation='relu', name='x16fc2')(x16)
    x16 = Dropout(0.5)(x16)
    x16 = Dense(2, activation='softmax', name='x16_out')(x16)

    x17 = Dense(64, activation='relu', name='x17fc1')(x)
    x17 = Dropout(0.5)(x17)
    x17 = Dense(64, activation='relu', name='x17fc2')(x17)
    x17 = Dropout(0.5)(x17)
    x17 = Dense(2, activation='softmax', name='x17_out')(x17)

    x18 = Dense(64, activation='relu', name='x18fc1')(x)
    x18 = Dropout(0.5)(x18)
    x18 = Dense(64, activation='relu', name='x18fc2')(x18)
    x18 = Dropout(0.5)(x18)
    x18 = Dense(2, activation='softmax', name='x18_out')(x18)

    x19 = Dense(64, activation='relu', name='x19fc1')(x)
    x19 = Dropout(0.5)(x19)
    x19 = Dense(64, activation='relu', name='x19fc2')(x19)
    x19 = Dropout(0.5)(x19)
    x19 = Dense(2, activation='softmax', name='x19_out')(x19)

    x20 = Dense(64, activation='relu', name='x20fc1')(x)
    x20 = Dropout(0.5)(x20)
    x20 = Dense(64, activation='relu', name='x20fc2')(x20)
    x20 = Dropout(0.5)(x20)
    x20 = Dense(2, activation='softmax', name='x20_out')(x20)

    x21 = Dense(64, activation='relu', name='x21fc1')(x)
    x21 = Dropout(0.5)(x21)
    x21 = Dense(64, activation='relu', name='x21fc2')(x21)
    x21 = Dropout(0.5)(x21)
    x21 = Dense(2, activation='softmax', name='x21_out')(x21)

    x22 = Dense(64, activation='relu', name='x22fc1')(x)
    x22 = Dropout(0.5)(x22)
    x22 = Dense(64, activation='relu', name='x22fc2')(x22)
    x22 = Dropout(0.5)(x22)
    x22 = Dense(2, activation='softmax', name='x22_out')(x22)

    x23 = Dense(64, activation='relu', name='x23fc1')(x)
    x23 = Dropout(0.5)(x23)
    x23 = Dense(64, activation='relu', name='x23fc2')(x23)
    x23 = Dropout(0.5)(x23)
    x23 = Dense(2, activation='softmax', name='x23_out')(x23)

    x24 = Dense(64, activation='relu', name='x24fc1')(x)
    x24 = Dropout(0.5)(x24)
    x24 = Dense(64, activation='relu', name='x24fc2')(x24)
    x24 = Dropout(0.5)(x24)
    x24 = Dense(2, activation='softmax', name='x24_out')(x24)

    x25 = Dense(64, activation='relu', name='x25fc1')(x)
    x25 = Dropout(0.5)(x25)
    x25 = Dense(64, activation='relu', name='x25fc2')(x25)
    x25 = Dropout(0.5)(x25)
    x25 = Dense(2, activation='softmax', name='x25_out')(x25)

    x26 = Dense(64, activation='relu', name='x26fc1')(x)
    x26 = Dropout(0.5)(x26)
    x26 = Dense(64, activation='relu', name='x26fc2')(x26)
    x26 = Dropout(0.5)(x26)
    x26 = Dense(2, activation='softmax', name='x26_out')(x26)

    x27 = Dense(64, activation='relu', name='x27fc1')(x)
    x27 = Dropout(0.5)(x27)
    x27 = Dense(64, activation='relu', name='x27fc2')(x27)
    x27 = Dropout(0.5)(x27)
    x27 = Dense(2, activation='softmax', name='x27_out')(x27)

    x28 = Dense(64, activation='relu', name='x28fc1')(x)
    x28 = Dropout(0.5)(x28)
    x28 = Dense(64, activation='relu', name='x28fc2')(x28)
    x28 = Dropout(0.5)(x28)
    x28 = Dense(2, activation='softmax', name='x28_out')(x28)

    x29 = Dense(64, activation='relu', name='x29fc1')(x)
    x29 = Dropout(0.5)(x29)
    x29 = Dense(64, activation='relu', name='x29fc2')(x29)
    x29 = Dropout(0.5)(x29)
    x29 = Dense(2, activation='softmax', name='x29_out')(x29)

    x30 = Dense(64, activation='relu', name='x30fc1')(x)
    x30 = Dropout(0.5)(x30)
    x30 = Dense(64, activation='relu', name='x30fc2')(x30)
    x30 = Dropout(0.5)(x30)
    x30 = Dense(2, activation='softmax', name='x30_out')(x30)

    x31 = Dense(64, activation='relu', name='x31fc1')(x)
    x31 = Dropout(0.5)(x31)
    x31 = Dense(64, activation='relu', name='x31fc2')(x31)
    x31 = Dropout(0.5)(x31)
    x31 = Dense(2, activation='softmax', name='x31_out')(x31)

    x32 = Dense(64, activation='relu', name='x32fc1')(x)
    x32 = Dropout(0.5)(x32)
    x32 = Dense(64, activation='relu', name='x32fc2')(x32)
    x32 = Dropout(0.5)(x32)
    x32 = Dense(2, activation='softmax', name='x32_out')(x32)

    x33 = Dense(64, activation='relu', name='x33fc1')(x)
    x33 = Dropout(0.5)(x33)
    x33 = Dense(64, activation='relu', name='x33fc2')(x33)
    x33 = Dropout(0.5)(x33)
    x33 = Dense(2, activation='softmax', name='x33_out')(x33)

    x34 = Dense(64, activation='relu', name='x34fc1')(x)
    x34 = Dropout(0.5)(x34)
    x34 = Dense(64, activation='relu', name='x34fc2')(x34)
    x34 = Dropout(0.5)(x34)
    x34 = Dense(2, activation='softmax', name='x34_out')(x34)

    x35 = Dense(64, activation='relu', name='x35fc1')(x)
    x35 = Dropout(0.5)(x35)
    x35 = Dense(64, activation='relu', name='x35fc2')(x35)
    x35 = Dropout(0.5)(x35)
    x35 = Dense(2, activation='softmax', name='x35_out')(x35)

    x36 = Dense(64, activation='relu', name='x36fc1')(x)
    x36 = Dropout(0.5)(x36)
    x36 = Dense(64, activation='relu', name='x36fc2')(x36)
    x36 = Dropout(0.5)(x36)
    x36 = Dense(2, activation='softmax', name='x36_out')(x36)

    x37 = Dense(64, activation='relu', name='x37fc1')(x)
    x37 = Dropout(0.5)(x37)
    x37 = Dense(64, activation='relu', name='x37fc2')(x37)
    x37 = Dropout(0.5)(x37)
    x37 = Dense(2, activation='softmax', name='x37_out')(x37)

    x38 = Dense(64, activation='relu', name='x38fc1')(x)
    x38 = Dropout(0.5)(x38)
    x38 = Dense(64, activation='relu', name='x38fc2')(x38)
    x38 = Dropout(0.5)(x38)
    x38 = Dense(2, activation='softmax', name='x38_out')(x38)

    x39 = Dense(64, activation='relu', name='x39fc1')(x)
    x39 = Dropout(0.5)(x39)
    x39 = Dense(64, activation='relu', name='x39fc2')(x39)
    x39 = Dropout(0.5)(x39)
    x39 = Dense(2, activation='softmax', name='x39_out')(x39)

    x40 = Dense(64, activation='relu', name='x40fc1')(x)
    x40 = Dropout(0.5)(x40)
    x40 = Dense(64, activation='relu', name='x40fc2')(x40)
    x40 = Dropout(0.5)(x40)
    x40 = Dense(2, activation='softmax', name='x40_out')(x40)

    x41 = Dense(64, activation='relu', name='x41fc1')(x)
    x41 = Dropout(0.5)(x41)
    x41 = Dense(64, activation='relu', name='x41fc2')(x41)
    x41 = Dropout(0.5)(x41)
    x41 = Dense(2, activation='softmax', name='x41_out')(x41)

    x42 = Dense(64, activation='relu', name='x42fc1')(x)
    x42 = Dropout(0.5)(x42)
    x42 = Dense(64, activation='relu', name='x42fc2')(x42)
    x42 = Dropout(0.5)(x42)
    x42 = Dense(2, activation='softmax', name='x42_out')(x42)

    x43 = Dense(64, activation='relu', name='x43fc1')(x)
    x43 = Dropout(0.5)(x43)
    x43 = Dense(64, activation='relu', name='x43fc2')(x43)
    x43 = Dropout(0.5)(x43)
    x43 = Dense(2, activation='softmax', name='x43_out')(x43)

    x44 = Dense(64, activation='relu', name='x44fc1')(x)
    x44 = Dropout(0.5)(x44)
    x44 = Dense(64, activation='relu', name='x44fc2')(x44)
    x44 = Dropout(0.5)(x44)
    x44 = Dense(2, activation='softmax', name='x44_out')(x44)

    x45 = Dense(64, activation='relu', name='x45fc1')(x)
    x45 = Dropout(0.5)(x45)
    x45 = Dense(64, activation='relu', name='x45fc2')(x45)
    x45 = Dropout(0.5)(x45)
    x45 = Dense(2, activation='softmax', name='x45_out')(x45)

    x46 = Dense(64, activation='relu', name='x46fc1')(x)
    x46 = Dropout(0.5)(x46)
    x46 = Dense(64, activation='relu', name='x46fc2')(x46)
    x46 = Dropout(0.5)(x46)
    x46 = Dense(2, activation='softmax', name='x46_out')(x46)

    x47 = Dense(64, activation='relu', name='x47fc1')(x)
    x47 = Dropout(0.5)(x47)
    x47 = Dense(64, activation='relu', name='x47fc2')(x47)
    x47 = Dropout(0.5)(x47)
    x47 = Dense(2, activation='softmax', name='x47_out')(x47)

    x48 = Dense(64, activation='relu', name='x48fc1')(x)
    x48 = Dropout(0.5)(x48)
    x48 = Dense(64, activation='relu', name='x48fc2')(x48)
    x48 = Dropout(0.5)(x48)
    x48 = Dense(2, activation='softmax', name='x48_out')(x48)

    x49 = Dense(64, activation='relu', name='x49fc1')(x)
    x49 = Dropout(0.5)(x49)
    x49 = Dense(64, activation='relu', name='x49fc2')(x49)
    x49 = Dropout(0.5)(x49)
    x49 = Dense(2, activation='softmax', name='x49_out')(x49)

    x50 = Dense(64, activation='relu', name='x50fc1')(x)
    x50 = Dropout(0.5)(x50)
    x50 = Dense(64, activation='relu', name='x50fc2')(x50)
    x50 = Dropout(0.5)(x50)
    x50 = Dense(2, activation='softmax', name='x50_out')(x50)

    x51 = Dense(64, activation='relu', name='x51fc1')(x)
    x51 = Dropout(0.5)(x51)
    x51 = Dense(64, activation='relu', name='x51fc2')(x51)
    x51 = Dropout(0.5)(x51)
    x51 = Dense(2, activation='softmax', name='x51_out')(x51)

    x52 = Dense(64, activation='relu', name='x52fc1')(x)
    x52 = Dropout(0.5)(x52)
    x52 = Dense(64, activation='relu', name='x52fc2')(x52)
    x52 = Dropout(0.5)(x52)
    x52 = Dense(2, activation='softmax', name='x52_out')(x52)

    x53 = Dense(64, activation='relu', name='x53fc1')(x)
    x53 = Dropout(0.5)(x53)
    x53 = Dense(64, activation='relu', name='x53fc2')(x53)
    x53 = Dropout(0.5)(x53)
    x53 = Dense(2, activation='softmax', name='x53_out')(x53)

    x54 = Dense(64, activation='relu', name='x54fc1')(x)
    x54 = Dropout(0.5)(x54)
    x54 = Dense(64, activation='relu', name='x54fc2')(x54)
    x54 = Dropout(0.5)(x54)
    x54 = Dense(2, activation='softmax', name='x54_out')(x54)

    x55 = Dense(64, activation='relu', name='x55fc1')(x)
    x55 = Dropout(0.5)(x55)
    x55 = Dense(64, activation='relu', name='x55fc2')(x55)
    x55 = Dropout(0.5)(x55)
    x55 = Dense(2, activation='softmax', name='x55_out')(x55)

    x56 = Dense(64, activation='relu', name='x56fc1')(x)
    x56 = Dropout(0.5)(x56)
    x56 = Dense(64, activation='relu', name='x56fc2')(x56)
    x56 = Dropout(0.5)(x56)
    x56 = Dense(2, activation='softmax', name='x56_out')(x56)

    x57 = Dense(64, activation='relu', name='x57fc1')(x)
    x57 = Dropout(0.5)(x57)
    x57 = Dense(64, activation='relu', name='x57fc2')(x57)
    x57 = Dropout(0.5)(x57)
    x57 = Dense(2, activation='softmax', name='x57_out')(x57)

    x58 = Dense(64, activation='relu', name='x58fc1')(x)
    x58 = Dropout(0.5)(x58)
    x58 = Dense(64, activation='relu', name='x58fc2')(x58)
    x58 = Dropout(0.5)(x58)
    x58 = Dense(2, activation='softmax', name='x58_out')(x58)

    x59 = Dense(64, activation='relu', name='x59fc1')(x)
    x59 = Dropout(0.5)(x59)
    x59 = Dense(64, activation='relu', name='x59fc2')(x59)
    x59 = Dropout(0.5)(x59)
    x59 = Dense(2, activation='softmax', name='x59_out')(x59)

    x60 = Dense(64, activation='relu', name='x60fc1')(x)
    x60 = Dropout(0.5)(x60)
    x60 = Dense(64, activation='relu', name='x60fc2')(x60)
    x60 = Dropout(0.5)(x60)
    x60 = Dense(2, activation='softmax', name='x60_out')(x60)

    x61 = Dense(64, activation='relu', name='x61fc1')(x)
    x61 = Dropout(0.5)(x61)
    x61 = Dense(64, activation='relu', name='x61fc2')(x61)
    x61 = Dropout(0.5)(x61)
    x61 = Dense(2, activation='softmax', name='x61_out')(x61)

    x62 = Dense(64, activation='relu', name='x62fc1')(x)
    x62 = Dropout(0.5)(x62)
    x62 = Dense(64, activation='relu', name='x62fc2')(x62)
    x62 = Dropout(0.5)(x62)
    x62 = Dense(2, activation='softmax', name='x62_out')(x62)

    x63 = Dense(64, activation='relu', name='x63fc1')(x)
    x63 = Dropout(0.5)(x63)
    x63 = Dense(64, activation='relu', name='x63fc2')(x63)
    x63 = Dropout(0.5)(x63)
    x63 = Dense(2, activation='softmax', name='x63_out')(x63)

    x64 = Dense(64, activation='relu', name='x64fc1')(x)
    x64 = Dropout(0.5)(x64)
    x64 = Dense(64, activation='relu', name='x64fc2')(x64)
    x64 = Dropout(0.5)(x64)
    x64 = Dense(2, activation='softmax', name='x64_out')(x64)

    x65 = Dense(64, activation='relu', name='x65fc1')(x)
    x65 = Dropout(0.5)(x65)
    x65 = Dense(64, activation='relu', name='x65fc2')(x65)
    x65 = Dropout(0.5)(x65)
    x65 = Dense(2, activation='softmax', name='x65_out')(x65)

    x66 = Dense(64, activation='relu', name='x66fc1')(x)
    x66 = Dropout(0.5)(x66)
    x66 = Dense(64, activation='relu', name='x66fc2')(x66)
    x66 = Dropout(0.5)(x66)
    x66 = Dense(2, activation='softmax', name='x66_out')(x66)

    x67 = Dense(64, activation='relu', name='x67fc1')(x)
    x67 = Dropout(0.5)(x67)
    x67 = Dense(64, activation='relu', name='x67fc2')(x67)
    x67 = Dropout(0.5)(x67)
    x67 = Dense(2, activation='softmax', name='x67_out')(x67)

    x68 = Dense(64, activation='relu', name='x68fc1')(x)
    x68 = Dropout(0.5)(x68)
    x68 = Dense(64, activation='relu', name='x68fc2')(x68)
    x68 = Dropout(0.5)(x68)
    x68 = Dense(2, activation='softmax', name='x68_out')(x68)

    x69 = Dense(64, activation='relu', name='x69fc1')(x)
    x69 = Dropout(0.5)(x69)
    x69 = Dense(64, activation='relu', name='x69fc2')(x69)
    x69 = Dropout(0.5)(x69)
    x69 = Dense(2, activation='softmax', name='x69_out')(x69)

    x70 = Dense(64, activation='relu', name='x70fc1')(x)
    x70 = Dropout(0.5)(x70)
    x70 = Dense(64, activation='relu', name='x70fc2')(x70)
    x70 = Dropout(0.5)(x70)
    x70 = Dense(2, activation='softmax', name='x70_out')(x70)

    x71 = Dense(64, activation='relu', name='x71fc1')(x)
    x71 = Dropout(0.5)(x71)
    x71 = Dense(64, activation='relu', name='x71fc2')(x71)
    x71 = Dropout(0.5)(x71)
    x71 = Dense(2, activation='softmax', name='x71_out')(x71)

    x72 = Dense(64, activation='relu', name='x72fc1')(x)
    x72 = Dropout(0.5)(x72)
    x72 = Dense(64, activation='relu', name='x72fc2')(x72)
    x72 = Dropout(0.5)(x72)
    x72 = Dense(2, activation='softmax', name='x72_out')(x72)

    x73 = Dense(64, activation='relu', name='x73fc1')(x)
    x73 = Dropout(0.5)(x73)
    x73 = Dense(64, activation='relu', name='x73fc2')(x73)
    x73 = Dropout(0.5)(x73)
    x73 = Dense(2, activation='softmax', name='x73_out')(x73)

    x74 = Dense(64, activation='relu', name='x74fc1')(x)
    x74 = Dropout(0.5)(x74)
    x74 = Dense(64, activation='relu', name='x74fc2')(x74)
    x74 = Dropout(0.5)(x74)
    x74 = Dense(2, activation='softmax', name='x74_out')(x74)

    x75 = Dense(64, activation='relu', name='x75fc1')(x)
    x75 = Dropout(0.5)(x75)
    x75 = Dense(64, activation='relu', name='x75fc2')(x75)
    x75 = Dropout(0.5)(x75)
    x75 = Dense(2, activation='softmax', name='x75_out')(x75)

    x76 = Dense(64, activation='relu', name='x76fc1')(x)
    x76 = Dropout(0.5)(x76)
    x76 = Dense(64, activation='relu', name='x76fc2')(x76)
    x76 = Dropout(0.5)(x76)
    x76 = Dense(2, activation='softmax', name='x76_out')(x76)

    x77 = Dense(64, activation='relu', name='x77fc1')(x)
    x77 = Dropout(0.5)(x77)
    x77 = Dense(64, activation='relu', name='x77fc2')(x77)
    x77 = Dropout(0.5)(x77)
    x77 = Dense(2, activation='softmax', name='x77_out')(x77)

    x78 = Dense(64, activation='relu', name='x78fc1')(x)
    x78 = Dropout(0.5)(x78)
    x78 = Dense(64, activation='relu', name='x78fc2')(x78)
    x78 = Dropout(0.5)(x78)
    x78 = Dense(2, activation='softmax', name='x78_out')(x78)

    x79 = Dense(64, activation='relu', name='x79fc1')(x)
    x79 = Dropout(0.5)(x79)
    x79 = Dense(64, activation='relu', name='x79fc2')(x79)
    x79 = Dropout(0.5)(x79)
    x79 = Dense(2, activation='softmax', name='x79_out')(x79)

    x80 = Dense(64, activation='relu', name='x80fc1')(x)
    x80 = Dropout(0.5)(x80)
    x80 = Dense(64, activation='relu', name='x80fc2')(x80)
    x80 = Dropout(0.5)(x80)
    x80 = Dense(2, activation='softmax', name='x80_out')(x80)

    x81 = Dense(64, activation='relu', name='x81fc1')(x)
    x81 = Dropout(0.5)(x81)
    x81 = Dense(64, activation='relu', name='x81fc2')(x81)
    x81 = Dropout(0.5)(x81)
    x81 = Dense(2, activation='softmax', name='x81_out')(x81)

    x82 = Dense(64, activation='relu', name='x82fc1')(x)
    x82 = Dropout(0.5)(x82)
    x82 = Dense(64, activation='relu', name='x82fc2')(x82)
    x82 = Dropout(0.5)(x82)
    x82 = Dense(2, activation='softmax', name='x82_out')(x82)

    x83 = Dense(64, activation='relu', name='x83fc1')(x)
    x83 = Dropout(0.5)(x83)
    x83 = Dense(64, activation='relu', name='x83fc2')(x83)
    x83 = Dropout(0.5)(x83)
    x83 = Dense(2, activation='softmax', name='x83_out')(x83)

    x84 = Dense(64, activation='relu', name='x84fc1')(x)
    x84 = Dropout(0.5)(x84)
    x84 = Dense(64, activation='relu', name='x84fc2')(x84)
    x84 = Dropout(0.5)(x84)
    x84 = Dense(2, activation='softmax', name='x84_out')(x84)

    x85 = Dense(64, activation='relu', name='x85fc1')(x)
    x85 = Dropout(0.5)(x85)
    x85 = Dense(64, activation='relu', name='x85fc2')(x85)
    x85 = Dropout(0.5)(x85)
    x85 = Dense(2, activation='softmax', name='x85_out')(x85)

    x86 = Dense(64, activation='relu', name='x86fc1')(x)
    x86 = Dropout(0.5)(x86)
    x86 = Dense(64, activation='relu', name='x86fc2')(x86)
    x86 = Dropout(0.5)(x86)
    x86 = Dense(2, activation='softmax', name='x86_out')(x86)

    x87 = Dense(64, activation='relu', name='x87fc1')(x)
    x87 = Dropout(0.5)(x87)
    x87 = Dense(64, activation='relu', name='x87fc2')(x87)
    x87 = Dropout(0.5)(x87)
    x87 = Dense(2, activation='softmax', name='x87_out')(x87)

    x88 = Dense(64, activation='relu', name='x88fc1')(x)
    x88 = Dropout(0.5)(x88)
    x88 = Dense(64, activation='relu', name='x88fc2')(x88)
    x88 = Dropout(0.5)(x88)
    x88 = Dense(2, activation='softmax', name='x88_out')(x88)

    x89 = Dense(64, activation='relu', name='x89fc1')(x)
    x89 = Dropout(0.5)(x89)
    x89 = Dense(64, activation='relu', name='x89fc2')(x89)
    x89 = Dropout(0.5)(x89)
    x89 = Dense(2, activation='softmax', name='x89_out')(x89)

    x90 = Dense(64, activation='relu', name='x90fc1')(x)
    x90 = Dropout(0.5)(x90)
    x90 = Dense(64, activation='relu', name='x90fc2')(x90)
    x90 = Dropout(0.5)(x90)
    x90 = Dense(2, activation='softmax', name='x90_out')(x90)

    x91 = Dense(64, activation='relu', name='x91fc1')(x)
    x91 = Dropout(0.5)(x91)
    x91 = Dense(64, activation='relu', name='x91fc2')(x91)
    x91 = Dropout(0.5)(x91)
    x91 = Dense(2, activation='softmax', name='x91_out')(x91)

    x92 = Dense(64, activation='relu', name='x92fc1')(x)
    x92 = Dropout(0.5)(x92)
    x92 = Dense(64, activation='relu', name='x92fc2')(x92)
    x92 = Dropout(0.5)(x92)
    x92 = Dense(2, activation='softmax', name='x92_out')(x92)

    x93 = Dense(64, activation='relu', name='x93fc1')(x)
    x93 = Dropout(0.5)(x93)
    x93 = Dense(64, activation='relu', name='x93fc2')(x93)
    x93 = Dropout(0.5)(x93)
    x93 = Dense(2, activation='softmax', name='x93_out')(x93)

    x94 = Dense(64, activation='relu', name='x94fc1')(x)
    x94 = Dropout(0.5)(x94)
    x94 = Dense(64, activation='relu', name='x94fc2')(x94)
    x94 = Dropout(0.5)(x94)
    x94 = Dense(2, activation='softmax', name='x94_out')(x94)

    x95 = Dense(64, activation='relu', name='x95fc1')(x)
    x95 = Dropout(0.5)(x95)
    x95 = Dense(64, activation='relu', name='x95fc2')(x95)
    x95 = Dropout(0.5)(x95)
    x95 = Dense(2, activation='softmax', name='x95_out')(x95)

    x96 = Dense(64, activation='relu', name='x96fc1')(x)
    x96 = Dropout(0.5)(x96)
    x96 = Dense(64, activation='relu', name='x96fc2')(x96)
    x96 = Dropout(0.5)(x96)
    x96 = Dense(2, activation='softmax', name='x96_out')(x96)

    x97 = Dense(64, activation='relu', name='x97fc1')(x)
    x97 = Dropout(0.5)(x97)
    x97 = Dense(64, activation='relu', name='x97fc2')(x97)
    x97 = Dropout(0.5)(x97)
    x97 = Dense(2, activation='softmax', name='x97_out')(x97)

    x98 = Dense(64, activation='relu', name='x98fc1')(x)
    x98 = Dropout(0.5)(x98)
    x98 = Dense(64, activation='relu', name='x98fc2')(x98)
    x98 = Dropout(0.5)(x98)
    x98 = Dense(2, activation='softmax', name='x98_out')(x98)

    x99 = Dense(64, activation='relu', name='x99fc1')(x)
    x99 = Dropout(0.5)(x99)
    x99 = Dense(64, activation='relu', name='x99fc2')(x99)
    x99 = Dropout(0.5)(x99)
    x99 = Dense(2, activation='softmax', name='x99_out')(x99)

    x100 = Dense(64, activation='relu', name='x100fc1')(x)
    x100 = Dropout(0.5)(x100)
    x100 = Dense(64, activation='relu', name='x100fc2')(x100)
    x100 = Dropout(0.5)(x100)
    x100 = Dense(2, activation='softmax', name='x100_out')(x100)

    x101 = Dense(64, activation='relu', name='x101fc1')(x)
    x101 = Dropout(0.5)(x101)
    x101 = Dense(64, activation='relu', name='x101fc2')(x101)
    x101 = Dropout(0.5)(x101)
    x101 = Dense(2, activation='softmax', name='x101_out')(x101)

    x102 = Dense(64, activation='relu', name='x102fc1')(x)
    x102 = Dropout(0.5)(x102)
    x102 = Dense(64, activation='relu', name='x102fc2')(x102)
    x102 = Dropout(0.5)(x102)
    x102 = Dense(2, activation='softmax', name='x102_out')(x102)

    x103 = Dense(64, activation='relu', name='x103fc1')(x)
    x103 = Dropout(0.5)(x103)
    x103 = Dense(64, activation='relu', name='x103fc2')(x103)
    x103 = Dropout(0.5)(x103)
    x103 = Dense(2, activation='softmax', name='x103_out')(x103)

    x104 = Dense(64, activation='relu', name='x104fc1')(x)
    x104 = Dropout(0.5)(x104)
    x104 = Dense(64, activation='relu', name='x104fc2')(x104)
    x104 = Dropout(0.5)(x104)
    x104 = Dense(2, activation='softmax', name='x104_out')(x104)

    x105 = Dense(64, activation='relu', name='x105fc1')(x)
    x105 = Dropout(0.5)(x105)
    x105 = Dense(64, activation='relu', name='x105fc2')(x105)
    x105 = Dropout(0.5)(x105)
    x105 = Dense(2, activation='softmax', name='x105_out')(x105)

    x106 = Dense(64, activation='relu', name='x106fc1')(x)
    x106 = Dropout(0.5)(x106)
    x106 = Dense(64, activation='relu', name='x106fc2')(x106)
    x106 = Dropout(0.5)(x106)
    x106 = Dense(2, activation='softmax', name='x106_out')(x106)

    x107 = Dense(64, activation='relu', name='x107fc1')(x)
    x107 = Dropout(0.5)(x107)
    x107 = Dense(64, activation='relu', name='x107fc2')(x107)
    x107 = Dropout(0.5)(x107)
    x107 = Dense(2, activation='softmax', name='x107_out')(x107)

    x108 = Dense(64, activation='relu', name='x108fc1')(x)
    x108 = Dropout(0.5)(x108)
    x108 = Dense(64, activation='relu', name='x108fc2')(x108)
    x108 = Dropout(0.5)(x108)
    x108 = Dense(2, activation='softmax', name='x108_out')(x108)

    x109 = Dense(64, activation='relu', name='x109fc1')(x)
    x109 = Dropout(0.5)(x109)
    x109 = Dense(64, activation='relu', name='x109fc2')(x109)
    x109 = Dropout(0.5)(x109)
    x109 = Dense(2, activation='softmax', name='x109_out')(x109)

    x110 = Dense(64, activation='relu', name='x110fc1')(x)
    x110 = Dropout(0.5)(x110)
    x110 = Dense(64, activation='relu', name='x110fc2')(x110)
    x110 = Dropout(0.5)(x110)
    x110 = Dense(2, activation='softmax', name='x110_out')(x110)

    x111 = Dense(64, activation='relu', name='x111fc1')(x)
    x111 = Dropout(0.5)(x111)
    x111 = Dense(64, activation='relu', name='x111fc2')(x111)
    x111 = Dropout(0.5)(x111)
    x111 = Dense(2, activation='softmax', name='x111_out')(x111)

    x112 = Dense(64, activation='relu', name='x112fc1')(x)
    x112 = Dropout(0.5)(x112)
    x112 = Dense(64, activation='relu', name='x112fc2')(x112)
    x112 = Dropout(0.5)(x112)
    x112 = Dense(2, activation='softmax', name='x112_out')(x112)

    x113 = Dense(64, activation='relu', name='x113fc1')(x)
    x113 = Dropout(0.5)(x113)
    x113 = Dense(64, activation='relu', name='x113fc2')(x113)
    x113 = Dropout(0.5)(x113)
    x113 = Dense(2, activation='softmax', name='x113_out')(x113)

    x114 = Dense(64, activation='relu', name='x114fc1')(x)
    x114 = Dropout(0.5)(x114)
    x114 = Dense(64, activation='relu', name='x114fc2')(x114)
    x114 = Dropout(0.5)(x114)
    x114 = Dense(2, activation='softmax', name='x114_out')(x114)

    x115 = Dense(64, activation='relu', name='x115fc1')(x)
    x115 = Dropout(0.5)(x115)
    x115 = Dense(64, activation='relu', name='x115fc2')(x115)
    x115 = Dropout(0.5)(x115)
    x115 = Dense(2, activation='softmax', name='x115_out')(x115)

    x116 = Dense(64, activation='relu', name='x116fc1')(x)
    x116 = Dropout(0.5)(x116)
    x116 = Dense(64, activation='relu', name='x116fc2')(x116)
    x116 = Dropout(0.5)(x116)
    x116 = Dense(2, activation='softmax', name='x116_out')(x116)

    x117 = Dense(64, activation='relu', name='x117fc1')(x)
    x117 = Dropout(0.5)(x117)
    x117 = Dense(64, activation='relu', name='x117fc2')(x117)
    x117 = Dropout(0.5)(x117)
    x117 = Dense(2, activation='softmax', name='x117_out')(x117)

    x118 = Dense(64, activation='relu', name='x118fc1')(x)
    x118 = Dropout(0.5)(x118)
    x118 = Dense(64, activation='relu', name='x118fc2')(x118)
    x118 = Dropout(0.5)(x118)
    x118 = Dense(2, activation='softmax', name='x118_out')(x118)

    x119 = Dense(64, activation='relu', name='x119fc1')(x)
    x119 = Dropout(0.5)(x119)
    x119 = Dense(64, activation='relu', name='x119fc2')(x119)
    x119 = Dropout(0.5)(x119)
    x119 = Dense(2, activation='softmax', name='x119_out')(x119)

    x120 = Dense(64, activation='relu', name='x120fc1')(x)
    x120 = Dropout(0.5)(x120)
    x120 = Dense(64, activation='relu', name='x120fc2')(x120)
    x120 = Dropout(0.5)(x120)
    x120 = Dense(2, activation='softmax', name='x120_out')(x120)

    x121 = Dense(64, activation='relu', name='x121fc1')(x)
    x121 = Dropout(0.5)(x121)
    x121 = Dense(64, activation='relu', name='x121fc2')(x121)
    x121 = Dropout(0.5)(x121)
    x121 = Dense(2, activation='softmax', name='x121_out')(x121)

    x122 = Dense(64, activation='relu', name='x122fc1')(x)
    x122 = Dropout(0.5)(x122)
    x122 = Dense(64, activation='relu', name='x122fc2')(x122)
    x122 = Dropout(0.5)(x122)
    x122 = Dense(2, activation='softmax', name='x122_out')(x122)

    x123 = Dense(64, activation='relu', name='x123fc1')(x)
    x123 = Dropout(0.5)(x123)
    x123 = Dense(64, activation='relu', name='x123fc2')(x123)
    x123 = Dropout(0.5)(x123)
    x123 = Dense(2, activation='softmax', name='x123_out')(x123)

    x124 = Dense(64, activation='relu', name='x124fc1')(x)
    x124 = Dropout(0.5)(x124)
    x124 = Dense(64, activation='relu', name='x124fc2')(x124)
    x124 = Dropout(0.5)(x124)
    x124 = Dense(2, activation='softmax', name='x124_out')(x124)

    x125 = Dense(64, activation='relu', name='x125fc1')(x)
    x125 = Dropout(0.5)(x125)
    x125 = Dense(64, activation='relu', name='x125fc2')(x125)
    x125 = Dropout(0.5)(x125)
    x125 = Dense(2, activation='softmax', name='x125_out')(x125)

    x126 = Dense(64, activation='relu', name='x126fc1')(x)
    x126 = Dropout(0.5)(x126)
    x126 = Dense(64, activation='relu', name='x126fc2')(x126)
    x126 = Dropout(0.5)(x126)
    x126 = Dense(2, activation='softmax', name='x126_out')(x126)

    x127 = Dense(64, activation='relu', name='x127fc1')(x)
    x127 = Dropout(0.5)(x127)
    x127 = Dense(64, activation='relu', name='x127fc2')(x127)
    x127 = Dropout(0.5)(x127)
    x127 = Dense(2, activation='softmax', name='x127_out')(x127)

    x128 = Dense(64, activation='relu', name='x128fc1')(x)
    x128 = Dropout(0.5)(x128)
    x128 = Dense(64, activation='relu', name='x128fc2')(x128)
    x128 = Dropout(0.5)(x128)
    x128 = Dense(2, activation='softmax', name='x128_out')(x128)

    x129 = Dense(64, activation='relu', name='x129fc1')(x)
    x129 = Dropout(0.5)(x129)
    x129 = Dense(64, activation='relu', name='x129fc2')(x129)
    x129 = Dropout(0.5)(x129)
    x129 = Dense(2, activation='softmax', name='x129_out')(x129)

    x130 = Dense(64, activation='relu', name='x130fc1')(x)
    x130 = Dropout(0.5)(x130)
    x130 = Dense(64, activation='relu', name='x130fc2')(x130)
    x130 = Dropout(0.5)(x130)
    x130 = Dense(2, activation='softmax', name='x130_out')(x130)

    x131 = Dense(64, activation='relu', name='x131fc1')(x)
    x131 = Dropout(0.5)(x131)
    x131 = Dense(64, activation='relu', name='x131fc2')(x131)
    x131 = Dropout(0.5)(x131)
    x131 = Dense(2, activation='softmax', name='x131_out')(x131)

    x132 = Dense(64, activation='relu', name='x132fc1')(x)
    x132 = Dropout(0.5)(x132)
    x132 = Dense(64, activation='relu', name='x132fc2')(x132)
    x132 = Dropout(0.5)(x132)
    x132 = Dense(2, activation='softmax', name='x132_out')(x132)

    x133 = Dense(64, activation='relu', name='x133fc1')(x)
    x133 = Dropout(0.5)(x133)
    x133 = Dense(64, activation='relu', name='x133fc2')(x133)
    x133 = Dropout(0.5)(x133)
    x133 = Dense(2, activation='softmax', name='x133_out')(x133)

    x134 = Dense(64, activation='relu', name='x134fc1')(x)
    x134 = Dropout(0.5)(x134)
    x134 = Dense(64, activation='relu', name='x134fc2')(x134)
    x134 = Dropout(0.5)(x134)
    x134 = Dense(2, activation='softmax', name='x134_out')(x134)

    x135 = Dense(64, activation='relu', name='x135fc1')(x)
    x135 = Dropout(0.5)(x135)
    x135 = Dense(64, activation='relu', name='x135fc2')(x135)
    x135 = Dropout(0.5)(x135)
    x135 = Dense(2, activation='softmax', name='x135_out')(x135)

    x136 = Dense(64, activation='relu', name='x136fc1')(x)
    x136 = Dropout(0.5)(x136)
    x136 = Dense(64, activation='relu', name='x136fc2')(x136)
    x136 = Dropout(0.5)(x136)
    x136 = Dense(2, activation='softmax', name='x136_out')(x136)

    x137 = Dense(64, activation='relu', name='x137fc1')(x)
    x137 = Dropout(0.5)(x137)
    x137 = Dense(64, activation='relu', name='x137fc2')(x137)
    x137 = Dropout(0.5)(x137)
    x137 = Dense(2, activation='softmax', name='x137_out')(x137)

    x138 = Dense(64, activation='relu', name='x138fc1')(x)
    x138 = Dropout(0.5)(x138)
    x138 = Dense(64, activation='relu', name='x138fc2')(x138)
    x138 = Dropout(0.5)(x138)
    x138 = Dense(2, activation='softmax', name='x138_out')(x138)

    x139 = Dense(64, activation='relu', name='x139fc1')(x)
    x139 = Dropout(0.5)(x139)
    x139 = Dense(64, activation='relu', name='x139fc2')(x139)
    x139 = Dropout(0.5)(x139)
    x139 = Dense(2, activation='softmax', name='x139_out')(x139)

    x140 = Dense(64, activation='relu', name='x140fc1')(x)
    x140 = Dropout(0.5)(x140)
    x140 = Dense(64, activation='relu', name='x140fc2')(x140)
    x140 = Dropout(0.5)(x140)
    x140 = Dense(2, activation='softmax', name='x140_out')(x140)

    x141 = Dense(64, activation='relu', name='x141fc1')(x)
    x141 = Dropout(0.5)(x141)
    x141 = Dense(64, activation='relu', name='x141fc2')(x141)
    x141 = Dropout(0.5)(x141)
    x141 = Dense(2, activation='softmax', name='x141_out')(x141)

    x142 = Dense(64, activation='relu', name='x142fc1')(x)
    x142 = Dropout(0.5)(x142)
    x142 = Dense(64, activation='relu', name='x142fc2')(x142)
    x142 = Dropout(0.5)(x142)
    x142 = Dense(2, activation='softmax', name='x142_out')(x142)

    x143 = Dense(64, activation='relu', name='x143fc1')(x)
    x143 = Dropout(0.5)(x143)
    x143 = Dense(64, activation='relu', name='x143fc2')(x143)
    x143 = Dropout(0.5)(x143)
    x143 = Dense(2, activation='softmax', name='x143_out')(x143)

    x144 = Dense(64, activation='relu', name='x144fc1')(x)
    x144 = Dropout(0.5)(x144)
    x144 = Dense(64, activation='relu', name='x144fc2')(x144)
    x144 = Dropout(0.5)(x144)
    x144 = Dense(2, activation='softmax', name='x144_out')(x144)

    x145 = Dense(64, activation='relu', name='x145fc1')(x)
    x145 = Dropout(0.5)(x145)
    x145 = Dense(64, activation='relu', name='x145fc2')(x145)
    x145 = Dropout(0.5)(x145)
    x145 = Dense(2, activation='softmax', name='x145_out')(x145)

    x146 = Dense(64, activation='relu', name='x146fc1')(x)
    x146 = Dropout(0.5)(x146)
    x146 = Dense(64, activation='relu', name='x146fc2')(x146)
    x146 = Dropout(0.5)(x146)
    x146 = Dense(2, activation='softmax', name='x146_out')(x146)

    x147 = Dense(64, activation='relu', name='x147fc1')(x)
    x147 = Dropout(0.5)(x147)
    x147 = Dense(64, activation='relu', name='x147fc2')(x147)
    x147 = Dropout(0.5)(x147)
    x147 = Dense(2, activation='softmax', name='x147_out')(x147)

    x148 = Dense(64, activation='relu', name='x148fc1')(x)
    x148 = Dropout(0.5)(x148)
    x148 = Dense(64, activation='relu', name='x148fc2')(x148)
    x148 = Dropout(0.5)(x148)
    x148 = Dense(2, activation='softmax', name='x148_out')(x148)

    x149 = Dense(64, activation='relu', name='x149fc1')(x)
    x149 = Dropout(0.5)(x149)
    x149 = Dense(64, activation='relu', name='x149fc2')(x149)
    x149 = Dropout(0.5)(x149)
    x149 = Dense(2, activation='softmax', name='x149_out')(x149)

    x150 = Dense(64, activation='relu', name='x150fc1')(x)
    x150 = Dropout(0.5)(x150)
    x150 = Dense(64, activation='relu', name='x150fc2')(x150)
    x150 = Dropout(0.5)(x150)
    x150 = Dense(2, activation='softmax', name='x150_out')(x150)

    x151 = Dense(64, activation='relu', name='x151fc1')(x)
    x151 = Dropout(0.5)(x151)
    x151 = Dense(64, activation='relu', name='x151fc2')(x151)
    x151 = Dropout(0.5)(x151)
    x151 = Dense(2, activation='softmax', name='x151_out')(x151)

    x152 = Dense(64, activation='relu', name='x152fc1')(x)
    x152 = Dropout(0.5)(x152)
    x152 = Dense(64, activation='relu', name='x152fc2')(x152)
    x152 = Dropout(0.5)(x152)
    x152 = Dense(2, activation='softmax', name='x152_out')(x152)

    x153 = Dense(64, activation='relu', name='x153fc1')(x)
    x153 = Dropout(0.5)(x153)
    x153 = Dense(64, activation='relu', name='x153fc2')(x153)
    x153 = Dropout(0.5)(x153)
    x153 = Dense(2, activation='softmax', name='x153_out')(x153)

    x154 = Dense(64, activation='relu', name='x154fc1')(x)
    x154 = Dropout(0.5)(x154)
    x154 = Dense(64, activation='relu', name='x154fc2')(x154)
    x154 = Dropout(0.5)(x154)
    x154 = Dense(2, activation='softmax', name='x154_out')(x154)

    x155 = Dense(64, activation='relu', name='x155fc1')(x)
    x155 = Dropout(0.5)(x155)
    x155 = Dense(64, activation='relu', name='x155fc2')(x155)
    x155 = Dropout(0.5)(x155)
    x155 = Dense(2, activation='softmax', name='x155_out')(x155)

    x156 = Dense(64, activation='relu', name='x156fc1')(x)
    x156 = Dropout(0.5)(x156)
    x156 = Dense(64, activation='relu', name='x156fc2')(x156)
    x156 = Dropout(0.5)(x156)
    x156 = Dense(2, activation='softmax', name='x156_out')(x156)

    x157 = Dense(64, activation='relu', name='x157fc1')(x)
    x157 = Dropout(0.5)(x157)
    x157 = Dense(64, activation='relu', name='x157fc2')(x157)
    x157 = Dropout(0.5)(x157)
    x157 = Dense(2, activation='softmax', name='x157_out')(x157)

    x158 = Dense(64, activation='relu', name='x158fc1')(x)
    x158 = Dropout(0.5)(x158)
    x158 = Dense(64, activation='relu', name='x158fc2')(x158)
    x158 = Dropout(0.5)(x158)
    x158 = Dense(2, activation='softmax', name='x158_out')(x158)

    x159 = Dense(64, activation='relu', name='x159fc1')(x)
    x159 = Dropout(0.5)(x159)
    x159 = Dense(64, activation='relu', name='x159fc2')(x159)
    x159 = Dropout(0.5)(x159)
    x159 = Dense(2, activation='softmax', name='x159_out')(x159)

    x160 = Dense(64, activation='relu', name='x160fc1')(x)
    x160 = Dropout(0.5)(x160)
    x160 = Dense(64, activation='relu', name='x160fc2')(x160)
    x160 = Dropout(0.5)(x160)
    x160 = Dense(2, activation='softmax', name='x160_out')(x160)

    x161 = Dense(64, activation='relu', name='x161fc1')(x)
    x161 = Dropout(0.5)(x161)
    x161 = Dense(64, activation='relu', name='x161fc2')(x161)
    x161 = Dropout(0.5)(x161)
    x161 = Dense(2, activation='softmax', name='x161_out')(x161)

    x162 = Dense(64, activation='relu', name='x162fc1')(x)
    x162 = Dropout(0.5)(x162)
    x162 = Dense(64, activation='relu', name='x162fc2')(x162)
    x162 = Dropout(0.5)(x162)
    x162 = Dense(2, activation='softmax', name='x162_out')(x162)

    x163 = Dense(64, activation='relu', name='x163fc1')(x)
    x163 = Dropout(0.5)(x163)
    x163 = Dense(64, activation='relu', name='x163fc2')(x163)
    x163 = Dropout(0.5)(x163)
    x163 = Dense(2, activation='softmax', name='x163_out')(x163)

    x164 = Dense(64, activation='relu', name='x164fc1')(x)
    x164 = Dropout(0.5)(x164)
    x164 = Dense(64, activation='relu', name='x164fc2')(x164)
    x164 = Dropout(0.5)(x164)
    x164 = Dense(2, activation='softmax', name='x164_out')(x164)

    x165 = Dense(64, activation='relu', name='x165fc1')(x)
    x165 = Dropout(0.5)(x165)
    x165 = Dense(64, activation='relu', name='x165fc2')(x165)
    x165 = Dropout(0.5)(x165)
    x165 = Dense(2, activation='softmax', name='x165_out')(x165)

    x166 = Dense(64, activation='relu', name='x166fc1')(x)
    x166 = Dropout(0.5)(x166)
    x166 = Dense(64, activation='relu', name='x166fc2')(x166)
    x166 = Dropout(0.5)(x166)
    x166 = Dense(2, activation='softmax', name='x166_out')(x166)

    x167 = Dense(64, activation='relu', name='x167fc1')(x)
    x167 = Dropout(0.5)(x167)
    x167 = Dense(64, activation='relu', name='x167fc2')(x167)
    x167 = Dropout(0.5)(x167)
    x167 = Dense(2, activation='softmax', name='x167_out')(x167)

    x168 = Dense(64, activation='relu', name='x168fc1')(x)
    x168 = Dropout(0.5)(x168)
    x168 = Dense(64, activation='relu', name='x168fc2')(x168)
    x168 = Dropout(0.5)(x168)
    x168 = Dense(2, activation='softmax', name='x168_out')(x168)

    x169 = Dense(64, activation='relu', name='x169fc1')(x)
    x169 = Dropout(0.5)(x169)
    x169 = Dense(64, activation='relu', name='x169fc2')(x169)
    x169 = Dropout(0.5)(x169)
    x169 = Dense(2, activation='softmax', name='x169_out')(x169)

    x170 = Dense(64, activation='relu', name='x170fc1')(x)
    x170 = Dropout(0.5)(x170)
    x170 = Dense(64, activation='relu', name='x170fc2')(x170)
    x170 = Dropout(0.5)(x170)
    x170 = Dense(2, activation='softmax', name='x170_out')(x170)

    x171 = Dense(64, activation='relu', name='x171fc1')(x)
    x171 = Dropout(0.5)(x171)
    x171 = Dense(64, activation='relu', name='x171fc2')(x171)
    x171 = Dropout(0.5)(x171)
    x171 = Dense(2, activation='softmax', name='x171_out')(x171)

    x172 = Dense(64, activation='relu', name='x172fc1')(x)
    x172 = Dropout(0.5)(x172)
    x172 = Dense(64, activation='relu', name='x172fc2')(x172)
    x172 = Dropout(0.5)(x172)
    x172 = Dense(2, activation='softmax', name='x172_out')(x172)

    x173 = Dense(64, activation='relu', name='x173fc1')(x)
    x173 = Dropout(0.5)(x173)
    x173 = Dense(64, activation='relu', name='x173fc2')(x173)
    x173 = Dropout(0.5)(x173)
    x173 = Dense(2, activation='softmax', name='x173_out')(x173)

    x174 = Dense(64, activation='relu', name='x174fc1')(x)
    x174 = Dropout(0.5)(x174)
    x174 = Dense(64, activation='relu', name='x174fc2')(x174)
    x174 = Dropout(0.5)(x174)
    x174 = Dense(2, activation='softmax', name='x174_out')(x174)

    x175 = Dense(64, activation='relu', name='x175fc1')(x)
    x175 = Dropout(0.5)(x175)
    x175 = Dense(64, activation='relu', name='x175fc2')(x175)
    x175 = Dropout(0.5)(x175)
    x175 = Dense(2, activation='softmax', name='x175_out')(x175)

    x176 = Dense(64, activation='relu', name='x176fc1')(x)
    x176 = Dropout(0.5)(x176)
    x176 = Dense(64, activation='relu', name='x176fc2')(x176)
    x176 = Dropout(0.5)(x176)
    x176 = Dense(2, activation='softmax', name='x176_out')(x176)

    x177 = Dense(64, activation='relu', name='x177fc1')(x)
    x177 = Dropout(0.5)(x177)
    x177 = Dense(64, activation='relu', name='x177fc2')(x177)
    x177 = Dropout(0.5)(x177)
    x177 = Dense(2, activation='softmax', name='x177_out')(x177)

    x178 = Dense(64, activation='relu', name='x178fc1')(x)
    x178 = Dropout(0.5)(x178)
    x178 = Dense(64, activation='relu', name='x178fc2')(x178)
    x178 = Dropout(0.5)(x178)
    x178 = Dense(2, activation='softmax', name='x178_out')(x178)

    x179 = Dense(64, activation='relu', name='x179fc1')(x)
    x179 = Dropout(0.5)(x179)
    x179 = Dense(64, activation='relu', name='x179fc2')(x179)
    x179 = Dropout(0.5)(x179)
    x179 = Dense(2, activation='softmax', name='x179_out')(x179)

    x180 = Dense(64, activation='relu', name='x180fc1')(x)
    x180 = Dropout(0.5)(x180)
    x180 = Dense(64, activation='relu', name='x180fc2')(x180)
    x180 = Dropout(0.5)(x180)
    x180 = Dense(2, activation='softmax', name='x180_out')(x180)

    x181 = Dense(64, activation='relu', name='x181fc1')(x)
    x181 = Dropout(0.5)(x181)
    x181 = Dense(64, activation='relu', name='x181fc2')(x181)
    x181 = Dropout(0.5)(x181)
    x181 = Dense(2, activation='softmax', name='x181_out')(x181)

    x182 = Dense(64, activation='relu', name='x182fc1')(x)
    x182 = Dropout(0.5)(x182)
    x182 = Dense(64, activation='relu', name='x182fc2')(x182)
    x182 = Dropout(0.5)(x182)
    x182 = Dense(2, activation='softmax', name='x182_out')(x182)

    x183 = Dense(64, activation='relu', name='x183fc1')(x)
    x183 = Dropout(0.5)(x183)
    x183 = Dense(64, activation='relu', name='x183fc2')(x183)
    x183 = Dropout(0.5)(x183)
    x183 = Dense(2, activation='softmax', name='x183_out')(x183)

    x184 = Dense(64, activation='relu', name='x184fc1')(x)
    x184 = Dropout(0.5)(x184)
    x184 = Dense(64, activation='relu', name='x184fc2')(x184)
    x184 = Dropout(0.5)(x184)
    x184 = Dense(2, activation='softmax', name='x184_out')(x184)

    x185 = Dense(64, activation='relu', name='x185fc1')(x)
    x185 = Dropout(0.5)(x185)
    x185 = Dense(64, activation='relu', name='x185fc2')(x185)
    x185 = Dropout(0.5)(x185)
    x185 = Dense(2, activation='softmax', name='x185_out')(x185)

    x186 = Dense(64, activation='relu', name='x186fc1')(x)
    x186 = Dropout(0.5)(x186)
    x186 = Dense(64, activation='relu', name='x186fc2')(x186)
    x186 = Dropout(0.5)(x186)
    x186 = Dense(2, activation='softmax', name='x186_out')(x186)

    x187 = Dense(64, activation='relu', name='x187fc1')(x)
    x187 = Dropout(0.5)(x187)
    x187 = Dense(64, activation='relu', name='x187fc2')(x187)
    x187 = Dropout(0.5)(x187)
    x187 = Dense(2, activation='softmax', name='x187_out')(x187)

    x188 = Dense(64, activation='relu', name='x188fc1')(x)
    x188 = Dropout(0.5)(x188)
    x188 = Dense(64, activation='relu', name='x188fc2')(x188)
    x188 = Dropout(0.5)(x188)
    x188 = Dense(2, activation='softmax', name='x188_out')(x188)

    x189 = Dense(64, activation='relu', name='x189fc1')(x)
    x189 = Dropout(0.5)(x189)
    x189 = Dense(64, activation='relu', name='x189fc2')(x189)
    x189 = Dropout(0.5)(x189)
    x189 = Dense(2, activation='softmax', name='x189_out')(x189)

    x190 = Dense(64, activation='relu', name='x190fc1')(x)
    x190 = Dropout(0.5)(x190)
    x190 = Dense(64, activation='relu', name='x190fc2')(x190)
    x190 = Dropout(0.5)(x190)
    x190 = Dense(2, activation='softmax', name='x190_out')(x190)

    x191 = Dense(64, activation='relu', name='x191fc1')(x)
    x191 = Dropout(0.5)(x191)
    x191 = Dense(64, activation='relu', name='x191fc2')(x191)
    x191 = Dropout(0.5)(x191)
    x191 = Dense(2, activation='softmax', name='x191_out')(x191)

    x192 = Dense(64, activation='relu', name='x192fc1')(x)
    x192 = Dropout(0.5)(x192)
    x192 = Dense(64, activation='relu', name='x192fc2')(x192)
    x192 = Dropout(0.5)(x192)
    x192 = Dense(2, activation='softmax', name='x192_out')(x192)

    x193 = Dense(64, activation='relu', name='x193fc1')(x)
    x193 = Dropout(0.5)(x193)
    x193 = Dense(64, activation='relu', name='x193fc2')(x193)
    x193 = Dropout(0.5)(x193)
    x193 = Dense(2, activation='softmax', name='x193_out')(x193)

    x194 = Dense(64, activation='relu', name='x194fc1')(x)
    x194 = Dropout(0.5)(x194)
    x194 = Dense(64, activation='relu', name='x194fc2')(x194)
    x194 = Dropout(0.5)(x194)
    x194 = Dense(2, activation='softmax', name='x194_out')(x194)

    x195 = Dense(64, activation='relu', name='x195fc1')(x)
    x195 = Dropout(0.5)(x195)
    x195 = Dense(64, activation='relu', name='x195fc2')(x195)
    x195 = Dropout(0.5)(x195)
    x195 = Dense(2, activation='softmax', name='x195_out')(x195)

    x196 = Dense(64, activation='relu', name='x196fc1')(x)
    x196 = Dropout(0.5)(x196)
    x196 = Dense(64, activation='relu', name='x196fc2')(x196)
    x196 = Dropout(0.5)(x196)
    x196 = Dense(2, activation='softmax', name='x196_out')(x196)

    x197 = Dense(64, activation='relu', name='x197fc1')(x)
    x197 = Dropout(0.5)(x197)
    x197 = Dense(64, activation='relu', name='x197fc2')(x197)
    x197 = Dropout(0.5)(x197)
    x197 = Dense(2, activation='softmax', name='x197_out')(x197)

    x198 = Dense(64, activation='relu', name='x198fc1')(x)
    x198 = Dropout(0.5)(x198)
    x198 = Dense(64, activation='relu', name='x198fc2')(x198)
    x198 = Dropout(0.5)(x198)
    x198 = Dense(2, activation='softmax', name='x198_out')(x198)

    x199 = Dense(64, activation='relu', name='x199fc1')(x)
    x199 = Dropout(0.5)(x199)
    x199 = Dense(64, activation='relu', name='x199fc2')(x199)
    x199 = Dropout(0.5)(x199)
    x199 = Dense(2, activation='softmax', name='x199_out')(x199)

    x200 = Dense(64, activation='relu', name='x200fc1')(x)
    x200 = Dropout(0.5)(x200)
    x200 = Dense(64, activation='relu', name='x200fc2')(x200)
    x200 = Dropout(0.5)(x200)
    x200 = Dense(2, activation='softmax', name='x200_out')(x200)

    x201 = Dense(64, activation='relu', name='x201fc1')(x)
    x201 = Dropout(0.5)(x201)
    x201 = Dense(64, activation='relu', name='x201fc2')(x201)
    x201 = Dropout(0.5)(x201)
    x201 = Dense(2, activation='softmax', name='x201_out')(x201)

    x202 = Dense(64, activation='relu', name='x202fc1')(x)
    x202 = Dropout(0.5)(x202)
    x202 = Dense(64, activation='relu', name='x202fc2')(x202)
    x202 = Dropout(0.5)(x202)
    x202 = Dense(2, activation='softmax', name='x202_out')(x202)

    x203 = Dense(64, activation='relu', name='x203fc1')(x)
    x203 = Dropout(0.5)(x203)
    x203 = Dense(64, activation='relu', name='x203fc2')(x203)
    x203 = Dropout(0.5)(x203)
    x203 = Dense(2, activation='softmax', name='x203_out')(x203)

    x204 = Dense(64, activation='relu', name='x204fc1')(x)
    x204 = Dropout(0.5)(x204)
    x204 = Dense(64, activation='relu', name='x204fc2')(x204)
    x204 = Dropout(0.5)(x204)
    x204 = Dense(2, activation='softmax', name='x204_out')(x204)

    x205 = Dense(64, activation='relu', name='x205fc1')(x)
    x205 = Dropout(0.5)(x205)
    x205 = Dense(64, activation='relu', name='x205fc2')(x205)
    x205 = Dropout(0.5)(x205)
    x205 = Dense(2, activation='softmax', name='x205_out')(x205)

    x206 = Dense(64, activation='relu', name='x206fc1')(x)
    x206 = Dropout(0.5)(x206)
    x206 = Dense(64, activation='relu', name='x206fc2')(x206)
    x206 = Dropout(0.5)(x206)
    x206 = Dense(2, activation='softmax', name='x206_out')(x206)

    x207 = Dense(64, activation='relu', name='x207fc1')(x)
    x207 = Dropout(0.5)(x207)
    x207 = Dense(64, activation='relu', name='x207fc2')(x207)
    x207 = Dropout(0.5)(x207)
    x207 = Dense(2, activation='softmax', name='x207_out')(x207)

    x208 = Dense(64, activation='relu', name='x208fc1')(x)
    x208 = Dropout(0.5)(x208)
    x208 = Dense(64, activation='relu', name='x208fc2')(x208)
    x208 = Dropout(0.5)(x208)
    x208 = Dense(2, activation='softmax', name='x208_out')(x208)

    x209 = Dense(64, activation='relu', name='x209fc1')(x)
    x209 = Dropout(0.5)(x209)
    x209 = Dense(64, activation='relu', name='x209fc2')(x209)
    x209 = Dropout(0.5)(x209)
    x209 = Dense(2, activation='softmax', name='x209_out')(x209)

    x210 = Dense(64, activation='relu', name='x210fc1')(x)
    x210 = Dropout(0.5)(x210)
    x210 = Dense(64, activation='relu', name='x210fc2')(x210)
    x210 = Dropout(0.5)(x210)
    x210 = Dense(2, activation='softmax', name='x210_out')(x210)

    x211 = Dense(64, activation='relu', name='x211fc1')(x)
    x211 = Dropout(0.5)(x211)
    x211 = Dense(64, activation='relu', name='x211fc2')(x211)
    x211 = Dropout(0.5)(x211)
    x211 = Dense(2, activation='softmax', name='x211_out')(x211)

    x212 = Dense(64, activation='relu', name='x212fc1')(x)
    x212 = Dropout(0.5)(x212)
    x212 = Dense(64, activation='relu', name='x212fc2')(x212)
    x212 = Dropout(0.5)(x212)
    x212 = Dense(2, activation='softmax', name='x212_out')(x212)

    x213 = Dense(64, activation='relu', name='x213fc1')(x)
    x213 = Dropout(0.5)(x213)
    x213 = Dense(64, activation='relu', name='x213fc2')(x213)
    x213 = Dropout(0.5)(x213)
    x213 = Dense(2, activation='softmax', name='x213_out')(x213)

    x214 = Dense(64, activation='relu', name='x214fc1')(x)
    x214 = Dropout(0.5)(x214)
    x214 = Dense(64, activation='relu', name='x214fc2')(x214)
    x214 = Dropout(0.5)(x214)
    x214 = Dense(2, activation='softmax', name='x214_out')(x214)

    x215 = Dense(64, activation='relu', name='x215fc1')(x)
    x215 = Dropout(0.5)(x215)
    x215 = Dense(64, activation='relu', name='x215fc2')(x215)
    x215 = Dropout(0.5)(x215)
    x215 = Dense(2, activation='softmax', name='x215_out')(x215)

    x216 = Dense(64, activation='relu', name='x216fc1')(x)
    x216 = Dropout(0.5)(x216)
    x216 = Dense(64, activation='relu', name='x216fc2')(x216)
    x216 = Dropout(0.5)(x216)
    x216 = Dense(2, activation='softmax', name='x216_out')(x216)

    x217 = Dense(64, activation='relu', name='x217fc1')(x)
    x217 = Dropout(0.5)(x217)
    x217 = Dense(64, activation='relu', name='x217fc2')(x217)
    x217 = Dropout(0.5)(x217)
    x217 = Dense(2, activation='softmax', name='x217_out')(x217)

    x218 = Dense(64, activation='relu', name='x218fc1')(x)
    x218 = Dropout(0.5)(x218)
    x218 = Dense(64, activation='relu', name='x218fc2')(x218)
    x218 = Dropout(0.5)(x218)
    x218 = Dense(2, activation='softmax', name='x218_out')(x218)

    x219 = Dense(64, activation='relu', name='x219fc1')(x)
    x219 = Dropout(0.5)(x219)
    x219 = Dense(64, activation='relu', name='x219fc2')(x219)
    x219 = Dropout(0.5)(x219)
    x219 = Dense(2, activation='softmax', name='x219_out')(x219)

    x220 = Dense(64, activation='relu', name='x220fc1')(x)
    x220 = Dropout(0.5)(x220)
    x220 = Dense(64, activation='relu', name='x220fc2')(x220)
    x220 = Dropout(0.5)(x220)
    x220 = Dense(2, activation='softmax', name='x220_out')(x220)

    x221 = Dense(64, activation='relu', name='x221fc1')(x)
    x221 = Dropout(0.5)(x221)
    x221 = Dense(64, activation='relu', name='x221fc2')(x221)
    x221 = Dropout(0.5)(x221)
    x221 = Dense(2, activation='softmax', name='x221_out')(x221)

    x222 = Dense(64, activation='relu', name='x222fc1')(x)
    x222 = Dropout(0.5)(x222)
    x222 = Dense(64, activation='relu', name='x222fc2')(x222)
    x222 = Dropout(0.5)(x222)
    x222 = Dense(2, activation='softmax', name='x222_out')(x222)

    x223 = Dense(64, activation='relu', name='x223fc1')(x)
    x223 = Dropout(0.5)(x223)
    x223 = Dense(64, activation='relu', name='x223fc2')(x223)
    x223 = Dropout(0.5)(x223)
    x223 = Dense(2, activation='softmax', name='x223_out')(x223)

    x224 = Dense(64, activation='relu', name='x224fc1')(x)
    x224 = Dropout(0.5)(x224)
    x224 = Dense(64, activation='relu', name='x224fc2')(x224)
    x224 = Dropout(0.5)(x224)
    x224 = Dense(2, activation='softmax', name='x224_out')(x224)

    x225 = Dense(64, activation='relu', name='x225fc1')(x)
    x225 = Dropout(0.5)(x225)
    x225 = Dense(64, activation='relu', name='x225fc2')(x225)
    x225 = Dropout(0.5)(x225)
    x225 = Dense(2, activation='softmax', name='x225_out')(x225)

    x226 = Dense(64, activation='relu', name='x226fc1')(x)
    x226 = Dropout(0.5)(x226)
    x226 = Dense(64, activation='relu', name='x226fc2')(x226)
    x226 = Dropout(0.5)(x226)
    x226 = Dense(2, activation='softmax', name='x226_out')(x226)

    x227 = Dense(64, activation='relu', name='x227fc1')(x)
    x227 = Dropout(0.5)(x227)
    x227 = Dense(64, activation='relu', name='x227fc2')(x227)
    x227 = Dropout(0.5)(x227)
    x227 = Dense(2, activation='softmax', name='x227_out')(x227)

    x228 = Dense(64, activation='relu', name='x228fc1')(x)
    x228 = Dropout(0.5)(x228)
    x228 = Dense(64, activation='relu', name='x228fc2')(x228)
    x228 = Dropout(0.5)(x228)
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

def Create_Ouput_List(IMG_SIZE, checkpoint_path):
    model = VGG16()
    model.load_weights(checkpoint_path)

    for file in tqdm(TEST_LIST):
        path = os.path.join(TEST_DIR, file)
        try:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            input_img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
            input_img /= 255.0
            model_out = model.predict(input_img)
            for i in range(len(model_out)):
                if np.argmax(model_out[i]) == 1:
                    TEST_DICT[file].append(i + 1)
        except:
            print('Exception ', path)
    clear_session()


def Build_Output_Val():

    for file in TEST_DICT:
        id = file.split('.')[0]
        itemlist = []
        for item in TEST_DICT[file]:
            itemlist.append(item)
        Output_List.append([id, itemlist])

        # for item in range(len(TEST_DICT[file])):
        #     Output_List[.append([id, TEST_DICT[file]])

    Output_List.sort()
    my_df = pd.DataFrame(Output_List)
    my_df.to_csv('iMaterialistSubmissionOutput.csv', index=False, header=False)


# os.environ["CUDA_VISIBLE_DEVICES"]="0"     #1080
#os.environ["CUDA_VISIBLE_DEVICES"]="1"      #680


TEST_DIR = 'D:PythonData/iMaterialist/Train/'
#TEST_DIR = 'D:PythonData/iMaterialist/Val/'
#TEST_DIR = 'D:PythonData/iMaterialist/Test/'
LR = 1e-4
IMG_SIZE = 50
NUM_MODELS = 1

model1 = 'C:\\Users\\Stefan\\Desktop\\PythonScripts\\Kaggle\\iMaterialistChallenge\\Keras\\' \
         'VGG16_saved_models\\1xdif-50\\weights.120-72.56.hdf5'

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




















