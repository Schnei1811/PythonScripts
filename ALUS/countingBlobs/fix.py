from glob import glob
import os
from shutil import move

img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\210126_images"



div_lst = [5,6,7,8,9,10]


for div in div_lst:
    file_lst = glob(os.path.join(img_dir, f"full_data_div{div}\\*"))
    for i, file in enumerate(file_lst):
        print(file)
        if ".JPG" in file:
            # import ipdb;ipdb.set_trace()
            move(file_lst[i+5], file_lst[i+2])
            move(file_lst[i+4], file_lst[i+1])