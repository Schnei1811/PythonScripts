import cv2


img = cv2.imread('Files/OctopusImages/Octopus200x200LineMantle/1.png', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

M0 = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
M90 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
M180 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
M270 = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
dst0 = cv2.warpAffine(img,M0,(cols,rows))
dst90 = cv2.warpAffine(img,M90,(cols,rows))
dst180 = cv2.warpAffine(img,M180,(cols,rows))
dst270 = cv2.warpAffine(img,M270,(cols,rows))

M305 = cv2.getRotationMatrix2D((cols/2,rows/2),305,1)
dst305 = cv2.warpAffine(img,M305,(cols,rows))

cv2.imshow('0', dst0)
cv2.imshow('90', dst90)
cv2.imshow('180', dst180)
cv2.imshow('270', dst270)
cv2.imshow('305', dst305)
cv2.waitKey()
cv2.destroyAllWindows()