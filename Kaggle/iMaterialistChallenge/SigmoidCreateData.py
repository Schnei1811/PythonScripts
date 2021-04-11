from random import shuffle
from tqdm import tqdm
import cv2
import os
import numpy as np
import json


def create_multitask_train_data():
    with open(anno_dir + 'train.json') as json_data: anno_train = json.load(json_data)
    training_data = []
    counter, counterbreak = 0, 800000
    for filename in tqdm(anno_train['annotations']):
        d = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,
                11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,
                21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0,
                31:0,32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:0,40:0,
                41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0,49:0,50:0,
                51:0,52:0,53:0,54:0,55:0,56:0,57:0,58:0,59:0,60:0,
                61:0,62:0,63:0,64:0,65:0,66:0,67:0,68:0,69:0,70:0,
                71:0,72:0,73:0,74:0,75:0,76:0,77:0,78:0,79:0,80:0,
                81:0,82:0,83:0,84:0,85:0,86:0,87:0,88:0,89:0,90:0,
                91:0,92:0,93:0,94:0,95:0,96:0,97:0,98:0,99:0,100:0,
                101:0,102:0,103:0,104:0,105:0,106:0,107:0,108:0,109:0,110:0,
                111:0,112:0,113:0,114:0,115:0,116:0,117:0,118:0,119:0,120:0,
                121:0,122:0,123:0,124:0,125:0,126:0,127:0,128:0,129:0,130:0,
                131:0,132:0,133:0,134:0,135:0,136:0,137:0,138:0,139:0,140:0,
                141:0,142:0,143:0,144:0,145:0,146:0,147:0,148:0,149:0,150:0,
                151:0,152:0,153:0,154:0,155:0,156:0,157:0,158:0,159:0,160:0,
                161:0,162:0,163:0,164:0,165:0,166:0,167:0,168:0,169:0,170:0,
                171:0,172:0,173:0,174:0,175:0,176:0,177:0,178:0,179:0,180:0,
                181:0,182:0,183:0,184:0,185:0,186:0,187:0,188:0,189:0,190:0,
                191:0,192:0,193:0,194:0,195:0,196:0,197:0,198:0,199:0,200:0,
                201:0,202:0,203:0,204:0,205:0,206:0,207:0,208:0,209:0,210:0,
                211:0,212:0,213:0,214:0,215:0,216:0,217:0,218:0,219:0,220:0,
                221:0,222:0,223:0,224:0,225:0,226:0,227:0,228:0}

        for i in range(len(filename['labelId'])): d[int(filename['labelId'][i])] = 1

        path = os.path.join(TRAIN_DIR, filename['imageId'] + '.jpg')
        try:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],
                                    d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
                                    d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30],
                                    d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40],
                                    d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50],
                                    d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60],
                                    d[61], d[62], d[63], d[64], d[65], d[66], d[67], d[68], d[69], d[70],
                                    d[71], d[72], d[73], d[74], d[75], d[76], d[77], d[78], d[79], d[80],
                                    d[81], d[82], d[83], d[84], d[85], d[86], d[87], d[88], d[89], d[90],
                                    d[91], d[92], d[93], d[94], d[95], d[96], d[97], d[98], d[99], d[100],
                                    d[101], d[102], d[103], d[104], d[105], d[106], d[107], d[108], d[109], d[110],
                                    d[111], d[112], d[113], d[114], d[115], d[116], d[117], d[118], d[119], d[120],
                                    d[121], d[122], d[123], d[124], d[125], d[126], d[127], d[128], d[129], d[130],
                                    d[131], d[132], d[133], d[134], d[135], d[136], d[137], d[138], d[139], d[140],
                                    d[141], d[142], d[143], d[144], d[145], d[146], d[147], d[148], d[149], d[150],
                                    d[151], d[152], d[153], d[154], d[155], d[156], d[157], d[158], d[159], d[160],
                                    d[161], d[162], d[163], d[164], d[165], d[166], d[167], d[168], d[169], d[170],
                                    d[171], d[172], d[173], d[174], d[175], d[176], d[177], d[178], d[179], d[180],
                                    d[181], d[182], d[183], d[184], d[185], d[186], d[187], d[188], d[189], d[190],
                                    d[191], d[192], d[193], d[194], d[195], d[196], d[197], d[198], d[199], d[200],
                                    d[201], d[202], d[203], d[204], d[205], d[206], d[207], d[208], d[209], d[210],
                                    d[211], d[212], d[213], d[214], d[215], d[216], d[217], d[218], d[219], d[220],
                                    d[221], d[222], d[223], d[224], d[225], d[226], d[227], d[228]])
            counter += 1
        except:
            print('Exception ', path)
            pass

        if counter == counterbreak: break
    np.save('data/Binary_{}_pixel_train_multitask_data{}.npy'.format(IMG_SIZE, counter), training_data)
    shuffle(training_data)
    if counter == 0: np.save('data/Binary_{}_pixel_train_multitask_data.npy'.format(IMG_SIZE), training_data)
    return training_data

def create_multitask_val_data():
    with open(anno_dir + 'validation.json') as json_data: anno_val = json.load(json_data)
    validation_data = []
    counter, counterbreak = 0, 200
    for filename in tqdm(anno_val['annotations']):
        d = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,
                11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,
                21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0,
                31:0,32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:0,40:0,
                41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0,49:0,50:0,
                51:0,52:0,53:0,54:0,55:0,56:0,57:0,58:0,59:0,60:0,
                61:0,62:0,63:0,64:0,65:0,66:0,67:0,68:0,69:0,70:0,
                71:0,72:0,73:0,74:0,75:0,76:0,77:0,78:0,79:0,80:0,
                81:0,82:0,83:0,84:0,85:0,86:0,87:0,88:0,89:0,90:0,
                91:0,92:0,93:0,94:0,95:0,96:0,97:0,98:0,99:0,100:0,
                101:0,102:0,103:0,104:0,105:0,106:0,107:0,108:0,109:0,110:0,
                111:0,112:0,113:0,114:0,115:0,116:0,117:0,118:0,119:0,120:0,
                121:0,122:0,123:0,124:0,125:0,126:0,127:0,128:0,129:0,130:0,
                131:0,132:0,133:0,134:0,135:0,136:0,137:0,138:0,139:0,140:0,
                141:0,142:0,143:0,144:0,145:0,146:0,147:0,148:0,149:0,150:0,
                151:0,152:0,153:0,154:0,155:0,156:0,157:0,158:0,159:0,160:0,
                161:0,162:0,163:0,164:0,165:0,166:0,167:0,168:0,169:0,170:0,
                171:0,172:0,173:0,174:0,175:0,176:0,177:0,178:0,179:0,180:0,
                181:0,182:0,183:0,184:0,185:0,186:0,187:0,188:0,189:0,190:0,
                191:0,192:0,193:0,194:0,195:0,196:0,197:0,198:0,199:0,200:0,
                201:0,202:0,203:0,204:0,205:0,206:0,207:0,208:0,209:0,210:0,
                211:0,212:0,213:0,214:0,215:0,216:0,217:0,218:0,219:0,220:0,
                221:0,222:0,223:0,224:0,225:0,226:0,227:0,228:0}

        for i in range(len(filename['labelId'])): d[int(filename['labelId'][i])] = 1

        path = os.path.join(VAL_DIR, filename['imageId'] + '.jpg')
        try:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            validation_data.append([np.array(img), d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],
                                    d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
                                    d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30],
                                    d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40],
                                    d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50],
                                    d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60],
                                    d[61], d[62], d[63], d[64], d[65], d[66], d[67], d[68], d[69], d[70],
                                    d[71], d[72], d[73], d[74], d[75], d[76], d[77], d[78], d[79], d[80],
                                    d[81], d[82], d[83], d[84], d[85], d[86], d[87], d[88], d[89], d[90],
                                    d[91], d[92], d[93], d[94], d[95], d[96], d[97], d[98], d[99], d[100],
                                    d[101], d[102], d[103], d[104], d[105], d[106], d[107], d[108], d[109], d[110],
                                    d[111], d[112], d[113], d[114], d[115], d[116], d[117], d[118], d[119], d[120],
                                    d[121], d[122], d[123], d[124], d[125], d[126], d[127], d[128], d[129], d[130],
                                    d[131], d[132], d[133], d[134], d[135], d[136], d[137], d[138], d[139], d[140],
                                    d[141], d[142], d[143], d[144], d[145], d[146], d[147], d[148], d[149], d[150],
                                    d[151], d[152], d[153], d[154], d[155], d[156], d[157], d[158], d[159], d[160],
                                    d[161], d[162], d[163], d[164], d[165], d[166], d[167], d[168], d[169], d[170],
                                    d[171], d[172], d[173], d[174], d[175], d[176], d[177], d[178], d[179], d[180],
                                    d[181], d[182], d[183], d[184], d[185], d[186], d[187], d[188], d[189], d[190],
                                    d[191], d[192], d[193], d[194], d[195], d[196], d[197], d[198], d[199], d[200],
                                    d[201], d[202], d[203], d[204], d[205], d[206], d[207], d[208], d[209], d[210],
                                    d[211], d[212], d[213], d[214], d[215], d[216], d[217], d[218], d[219], d[220],
                                    d[221], d[222], d[223], d[224], d[225], d[226], d[227], d[228]])
        except:
            print('Exception')
    #     counter += 1
    #     if counter == counterbreak: break
    # np.save('data/Binary_{}_pixel_val_multitask_data{}.npy'.format(IMG_SIZE, counter), validation_data)
    shuffle(validation_data)
    if counter == 0: np.save('data/Binary_Trial{}_pixel_val_multitask_data.npy'.format(IMG_SIZE), validation_data)
    return validation_data




# def create_mock_train_val_multitask_data():
#     with open(anno_dir + 'train_annotations.json') as json_data: anno_train = json.load(json_data)
#     training_data = []
#     val_data = []
#     counter, counterbreak = 0, 7500
#     for filename in tqdm(anno_train['annotations']):
#         if filename['category_id'] == 0: label = [1, 0]
#         elif filename['category_id'] == 1: label = [0, 1]
#         path = os.path.join(TRAIN_DIR, filename['image_id']+'.jpg')
#         img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#         for filename2 in (anno_train['images']):
#             if filename2['id'] == filename['image_id']:
#                 if np.random.random() <= 0.1: val_data.append([np.array(img), np.array(label), filename2['location']])
#                 else: training_data.append([np.array(img), np.array(label), filename2['location']])
#                 break
#         # counter += 1
#         # if counter == counterbreak: break
#     counter = 0
#     with open(anno_dir + 'val_annotations.json') as json_data: anno_val = json.load(json_data)
#     for filename in tqdm(anno_val['annotations']):
#         if filename['category_id'] == 0: label = [1, 0]
#         elif filename['category_id'] == 1: label = [0, 1]
#         path = os.path.join(VAL_DIR, filename['image_id']+'.jpg')
#         img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#         for filename2 in (anno_val['images']):
#             if filename2['id'] == filename['image_id']:
#                 if np.random.random() <= 0.1: val_data.append([np.array(img), np.array(label), filename2['location']])
#                 else: training_data.append([np.array(img), np.array(label), filename2['location']])
#                 break
#     #     counter += 1
#     #     if counter == counterbreak: break
#     # np.save('data/{}_pixel_mock_train_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), training_data)
#     # np.save('data/{}_pixel_mock_val_multitask_data{}.npy'.format(IMG_SIZE, counterbreak), val_data)
#     if counter == 0:
#         np.save('data/{}_pixel_mock_train_multitask_data.npy'.format(IMG_SIZE), training_data)
#         np.save('data/{}_pixel_mock_val_multitask_data.npy'.format(IMG_SIZE), val_data)
#     return training_data

anno_dir = 'D:PythonData/iMaterialist/Annotations/'
TRAIN_DIR = 'D:PythonData/iMaterialist/Train/'
VAL_DIR = 'D:PythonData/iMaterialist/Val/'
IMG_SIZE = 50


create_multitask_train_data()
create_multitask_val_data()

#create_mock_train_val_multitask_data()
