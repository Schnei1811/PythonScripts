import json
import cv2
import shutil

directory = 'D:PythonData/iWildCam/Annotations/'

with open(directory + 'val_annotations.json') as json_data:
    d = json.load(json_data)

maxlocation = 0

print(d)
# print(type(d))
# print(d['info'])
print(d['images'])
# print(d['annotations'])
# print(d['categories'])
locationdict = {}

for name in d['images']:
    if name['location'] in locationdict: locationdict[name['location']] += 1
    else: locationdict[name['location']] = 0


    if name['location'] > maxlocation:
        maxlocation = name['location']

print(maxlocation)
print(sorted(locationdict))

# counter = 0
# for name in d['annotations']:
#     filename = 'D:PythonData/iWildCam/LowRes/train_val/' + name['id']+'.jpg'
#     counter += 1
#     print(name)
#
# print(counter)




# with open(directory + 'test_information.json') as json_data:
#     d = json.load(json_data)
#
#
# print(d)
# print(type(d))
# #print(d['info'])
# print(d['images'])
# #print(d['categories'])
#
# counter = 0
# for name in d:
#     #filename = 'D:PythonData/iWildCam/LowRes/train_val/' + name['id']+'.jpg'
#     counter += 1
#     print(name)
#
# print(counter)















