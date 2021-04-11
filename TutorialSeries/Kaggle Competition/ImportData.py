import cv2
import numpy as np
import dicom
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'G:/PythonScripts/PythonData/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)

labels_df.head()

for patient in patients[:10]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    #print(len(slices), label)
    #print(len(slices), slices[0].pixel_array.shape)
    #print(slices[0])
    #print(len(patients))

IMG_PX_SIZE = 50
HM_SLICES = 20

def chunks(l, n):
    for i in range(0, len(l), n): yield l[i:i +n]

def mean(l):
    return sum(l)/len(l)

def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize = False):
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES]])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES]])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    print(len(new_slices))

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num+1)
            y.imshow(each_slice)
        plt.show()

    # fig = plt.figure()
    # for num,each_slice in enumerate(slices[:12]):
    #     y = fig.add_subplot(3, 4, num+1)
    #     new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE))
    #     plt.imshow(slices[0].pixel_array)
    #     y.imshow(new_image)
    # plt.show()

    if label == 1: label = np.array([0, 1])
    elif label == 0: label = np.array([1, 0])

    return np.array(new_slices), label

much_data = []

for num, patient in enumerate(patients[:10]):
    if num % 100 == 0: print(num)

    try:
        img_data, label = process_data(patient, labels_df, img_px_size = IMG_PX_SIZE, hm_slices = HM_SLICES)
        much_data.append([img_data, label])
    except KeyError as e:
        print('This in unlabeled Data')

np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES), much_data)









