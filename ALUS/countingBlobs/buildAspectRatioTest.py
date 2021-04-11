import cv2
from glob import glob
import numpy as np

img_size = 224

def buildImageAspectRatio(X_path):
    img = cv2.imread(X_path)

    resize_x = int(img.shape[1] * img_size/max(img.shape))
    resize_y = int(img.shape[0] * img_size/max(img.shape))

    push_x = (img_size - resize_x) // 2
    push_y = (img_size - resize_y) // 2

    canvas = np.zeros((img_size, img_size, 3)).astype("uint8") + 255
    resized_img = cv2.resize(img, (resize_x, resize_y))
    import ipdb;ipdb.set_trace()
    canvas[push_y:resized_img.shape[0] + push_y, push_x:resized_img.shape[1] + push_x, :] = resized_img
    cv2.imshow("win", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return canvas


img_dir = "G:\\PythonData\\ALUS\\20.12.06_Alus_Added_Aranea_NonMite\\ALUS_Classifications\\*"

for path in glob(img_dir):
    buildImageAspectRatio(path)