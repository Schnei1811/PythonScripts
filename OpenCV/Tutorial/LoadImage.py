import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Files/thumbsup.jpg', cv2.IMREAD_GRAYSCALE)           #IMREAD_GRAYSCALE (0) IMREAD_COLOR (1)  IMREAD_UNCHANGED (-1)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')            #cv uses BGR and matplotlib uses RBG
# plt.plot([1xdif-50,1xdif-100], [80,1xdif-100], 'c', linewidth=5)
# plt.show()

cv2.imwrite('Files/thumbsupgray.png',img)