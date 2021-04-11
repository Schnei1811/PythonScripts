import cv2
import numpy as np

IMG_SIZE = 200
ROI_SIZE = 40
PIXEL_SKIP = 20

img = cv2.imread('Files/OctopusImages/OctopusOriginals/3.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
i = 0

for x in range(0, IMG_SIZE - ROI_SIZE, PIXEL_SKIP):
    for y in range(0, IMG_SIZE - ROI_SIZE, PIXEL_SKIP):
        if i == 0:
            totalrois = np.ravel(img[x:ROI_SIZE + x, y:ROI_SIZE + y])
            i += 1
        else:
            totalrois = np.vstack((totalrois, np.ravel(img[x:ROI_SIZE + x, y:ROI_SIZE + y])))

i = 0
for counter in totalrois:
    img = np.reshape(totalrois[i], (ROI_SIZE, ROI_SIZE))
    i += 1
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()