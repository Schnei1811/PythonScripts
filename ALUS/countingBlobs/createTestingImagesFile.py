import os
from glob import glob
from tqdm import tqdm
import cv2


# img_dir = "G:\\PythonData\\ALUS\\ALUS_Data\\"
# img_dir = "G:\\PythonData\\ALUS\\ALUS_Mixed_Test_Set\\"

img_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\"

# sizes = [1, 2, 3, 4, 5]

sizes = [5]

for div in sizes:
    if not os.path.exists(img_dir + "full_data_test_div{}".format(div)):
        os.makedirs(img_dir + "full_data_test_div{}".format(div))

    div_path = img_dir + "full_data_test_div{}\\".format(div)

    path_lst = []

    #import ipdb;ipdb.set_trace()
    for img_path in tqdm(glob(img_dir + "full_data_test\\*")):
        img_name = div_path + img_path.split("\\")[-1][:-4] + ".JPG"
        img = cv2.imread(img_path)
        h, w, c = img.shape
        resized_img = cv2.resize(img, (int(w / div), int(h / div)))
        cv2.imwrite(img_name, resized_img)
        path_lst.append(img_name)

    with open(img_dir + "test_div{}.txt".format(div), "w") as f:
        for path in path_lst:
            f.write("%s\n" % path)