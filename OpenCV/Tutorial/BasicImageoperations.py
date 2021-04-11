import numpy as np
import cv2

img = cv2.imread('Files/thumbsup.jpg', cv2.IMREAD_COLOR)
#img = cv2.imread('Files/thumbsup.jpg', cv2.IMREAD_GRAYSCALE)

print(img)

px = img[55, 55]         #colour value for that pixel
print(px)
img[100, 50] = [0, 0, 0]
print(px)

print(img)

# Region of Image
roi = img[100:150, 100:150]

img[50:150, 100:200] = [255, 255, 255]

# face = img[37:111, 107:194]
# img[0:74, 0:87] = face

resized_image = cv2.resize(img, (150, 150))


cv2.imshow('image', img)
#cv2.imshow('resizedimage', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()