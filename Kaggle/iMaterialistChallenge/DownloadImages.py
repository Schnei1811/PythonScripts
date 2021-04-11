import json
import requests
from tqdm import tqdm
import os

directory = 'D:PythonData/iMaterialist/Annotations/'

with open(directory + 'validation.json') as json_data:
    d = json.load(json_data)

for name in tqdm(d['images']):
    try:
        img_data = requests.get(name['url']).content
        with open('D:PythonData/iMaterialist/Val/{}.jpg'.format(name['imageId']), 'wb') as handler:
            handler.write(img_data)
    except:
        print('Exception {}'.format(name['url']))
#
with open(directory + 'train.json') as json_data:
    d = json.load(json_data)

for name in tqdm(d['images']):
    try:
        if os.path.isfile('D:PythonData/iMaterialist/Train/{}.jpg'.format(name['imageId'])) == True:
            pass
        else:
            img_data = requests.get(name['url']).content
            with open('D:PythonData/iMaterialist/Train/{}.jpg'.format(name['imageId']), 'wb') as handler:
                handler.write(img_data)
    except:
        print('Exception {}'.format(name['url']))

with open(directory + 'test.json') as json_data:
    d = json.load(json_data)

for name in tqdm(d['images']):
    try:
        img_data = requests.get(name['url']).content
        with open('D:PythonData/iMaterialist/Test/{}.jpg'.format(name['imageId']), 'wb') as handler:
            handler.write(img_data)
    except:
        print('Exception {}'.format(name['url']))











