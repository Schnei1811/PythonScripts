import json

directory = 'D:PythonData/iMaterialist/Annotations/'



with open(directory + 'train.json') as json_data:
    d = json.load(json_data)

maxval = 0

print(d['annotations'])
for name in d['annotations']:
    if name['imageId'] == '10':
        print(name)

    # for val in name['labelId']:
    #     if int(val) > maxval: maxval = int(val)


print(maxval)








# with open(directory + 'validation.json') as json_data:
#     d = json.load(json_data)
#
# maxval = 0
#
# print(d['annotations'])
# for name in d['annotations']:
#     print(name)
#     print(name['labelId'])
#     for val in name['labelId']:
#         if int(val) > maxval: maxval = int(val)
#
#
# print(maxval)
# print(type(d))
# print(d['info'])
# print(d['images'])
# print(d['annotations'])
# print(d['categories'])
# locationdict = {}
#
# for name in d['images']:
#     if name['location'] in locationdict: locationdict[name['location']] += 1
#     else: locationdict[name['location']] = 0
#
#
#     if name['location'] > maxlocation:
#         maxlocation = name['location']