from glob import glob
import os


train_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\full_data_div5\\*"
test_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\full_data_test\\*"
image_set_dir = "C:\\Users\\Stefan\\Desktop\\CountBlobs\\LCFCN-master\\ALUS_BL\\image_sets\\"

train_path_lst = []
val_path_lst = []
test_path_lst = []


for i, path in enumerate(glob(train_dir)):
    if ".txt" not in path and "dots" not in path:
        if i % 2 == 0:
            train_path_lst.append(path)
        else:
            val_path_lst.append(path)

for path in glob(test_dir):
    test_path_lst.append(path)

with open(os.path.join(image_set_dir, "training.txt"), "w") as f:
    for path in train_path_lst:
        f.write("%s\n" % path)

with open(os.path.join(image_set_dir, "validation.txt"), "w") as f:
    for path in val_path_lst:
        f.write("%s\n" % path)

with open(os.path.join(image_set_dir, "trainval.txt"), "w") as f:
    for path in train_path_lst:
        f.write("%s\n" % path)
    for path in val_path_lst:
        f.write("%s\n" % path)

with open(os.path.join(image_set_dir, "test.txt"), "w") as f:
    for path in test_path_lst:
        f.write("%s\n" % path)
