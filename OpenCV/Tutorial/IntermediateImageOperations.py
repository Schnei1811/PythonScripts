import cv2
import numpy as np

img1 = cv2.imread('Files/3D-Matplotlib.png')
img2 = cv2.imread('Files/mainsvmimage.png')

#add = img1 + img2
#add = cv2.add(img1,img2)            #added all the pixel values together
#weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
#cv2.imshow('add', weighted)

img3 = cv2.imread('Files/mainlogo.png')

rows, cols, channels = img3.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)    #if pixel value above 220, it'll be converted to 255. If below, to black. Inverted

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

dst = cv2.add(img1_bg, img3_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img3_fg', img3_fg)
cv2.imshow('dst', dst)

#cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()