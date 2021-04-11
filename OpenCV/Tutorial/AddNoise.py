import cv2
import numpy as np

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return temp_image.astype(np.uint8)


#img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#img_hor_mirror = cv2.flip(img, 1)
#noisy_img = convert_to_uint8(np.float64(np.copy(img)) + np.random.randn(img.shape[0], img.shape[1]) * sigma)
#noisy_img_mirror = convert_to_uint8(np.float64(np.copy(img_hor_mirror)) + np.random.randn(img.shape[0], img.shape[1]) * sigma)