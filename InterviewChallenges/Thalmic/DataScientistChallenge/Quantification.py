import numpy as np
import os

gaitfiles = os.listdir('WalkingActivity')
gaitdict = {}
gaitlenlist = []
roundgaitlenlist = []


for file in gaitfiles:
    gaitname = 'person' + file.split('.')[0]
    gaitdict[gaitname] = np.genfromtxt('WalkingActivity/{}'.format(file), delimiter=',')
    gaitlenlist.append(len(gaitdict[gaitname]))
    roundgaitlenlist.append(round(len(gaitdict[gaitname])/1000)*1000)



print('Min Length Time Series Entries:  ', min(gaitlenlist))
print('Max Length Time Series Entries:  ', max(gaitlenlist))



print('Woah! Huge disparity between the length of each sample')
print('As a first step, let\'s see if we can determine if we can relibly ID the person after 1 second of movement')

print(np.abs(gaitdict['person1'][:, 0] - 1).argmin())

print(gaitdict['person1'][70][0])

print('Let\'s see how well we can performif we consider 200 time steps or {} seconds'.format(gaitdict['person1'][200][0]))



for key in gaitdict:
    if 'train_data' not in locals():
        train_data = np.append(gaitdict[key][:min(gaitlenlist), 1], gaitdict[key][:min(gaitlenlist), 2])
        train_data = np.append(train_data, gaitdict[key][:min(gaitlenlist), 3])
        # np.array(gaitdict[gaitname][])
    else:
        new_data = np.append(gaitdict[key][:min(gaitlenlist), 1], gaitdict[key][:min(gaitlenlist), 2])
        new_data = np.append(new_data, gaitdict[key][:min(gaitlenlist), 3])
        train_data = np.vstack((train_data, new_data))





for key in gaitdict:
    if filename['category_id'] == 0: label = [1, 0]
    elif filename['category_id'] == 1: label = [0, 1]
    path = os.path.join(VAL_DIR, filename['image_id']+'.jpg')
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
    validation_data.append([np.array(img), label])
    counter += 1
    if counter == counterbreak: break




for key in gaitdict:
    if 'test_data' not in locals():
        test_data = np.append(gaitdict[key][min(gaitlenlist):min(gaitlenlist)*2, 1],
                              gaitdict[key][min(gaitlenlist):min(gaitlenlist)*2, 2])
        test_data = np.append(test_data, gaitdict[key][min(gaitlenlist):min(gaitlenlist*2), 3])
        # np.array(gaitdict[gaitname][])
    else:
        new_data = np.append(gaitdict[key][min(gaitlenlist):min(gaitlenlist)*2, 1],
                             gaitdict[key][min(gaitlenlist):min(gaitlenlist)*2, 2])
        new_data = np.append(new_data, gaitdict[key][min(gaitlenlist):min(gaitlenlist)*2, 3])
        test_data = np.vstack((test_data, new_data))

print(train_data)
print(test_data)
