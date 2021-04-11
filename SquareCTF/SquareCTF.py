import os
import numpy as np
import cv2
from random import shuffle
import matplotlib as plt

listdir = os.listdir('D:PythonData/Shredded/')


for i in range(20):
    shuffle(listdir)

    for img in listdir:
        if img == listdir[0]:
            img1 = cv2.imread('D:PythonData/Shredded/{}'.format(img))
            img2 = cv2.imread('D:PythonData/Shredded/{}'.format(img))
            vis = np.concatenate((img1, img2), axis=1)
        elif img == listdir[1]: pass
        else:
            image = cv2.imread('D:PythonData/Shredded/{}'.format(img))
            vis = np.concatenate((vis, image), axis=1)

    cv2.imwrite('D:PythonData/Shredded/Random{}.png'.format(i), vis)

# cv2.imshow('img', vis)
# cv2.waitKey()
# cv2.destroyAllWindows()





