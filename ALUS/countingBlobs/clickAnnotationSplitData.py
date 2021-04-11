import os
import shutil

cp_img_dir = "G:\\PythonData\\ALUS\\ALUS_Data\\images_div1"
train_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\full_data"
test_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\full_data_test"

file_lst = os.listdir(cp_img_dir)

counter = 0

for i, file in enumerate(file_lst):
    counter += 1
    cp_dir = train_dir
    if i > 0:
        if not file_lst[i].split("_")[0] == file_lst[i-1].split("_")[0]:
            counter = 0
        if counter > 10:
            cp_dir = test_dir
    shutil.copy(os.path.join(cp_img_dir, file), os.path.join(cp_dir, file))