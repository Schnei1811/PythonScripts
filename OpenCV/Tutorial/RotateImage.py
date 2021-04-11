import cv2

img = cv2.imread('Files/thumbsup.jpg', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

M0 = cv2.getRotationMatrix2D((cols/2, rows/2), 0, 1)
M90 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
M180 = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
M270 = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
dst0 = cv2.warpAffine(img, M0, (cols, rows))
dst90 = cv2.warpAffine(img, M90, (cols, rows))
dst180 = cv2.warpAffine(img, M180, (cols, rows))
dst270 = cv2.warpAffine(img, M270, (cols, rows))

M305 = cv2.getRotationMatrix2D((cols/2, rows/2), 305, 1)
dst305 = cv2.warpAffine(img, M305, (cols, rows))

cv2.imshow('0', dst0)
cv2.imshow('90', dst90)
cv2.imshow('180', dst180)
cv2.imshow('270', dst270)
cv2.imshow('305', dst305)
cv2.waitKey()
cv2.destroyAllWindows()



# def segment_mutiple_dot_images(COORDINATES, SAVEDIR):
#     dotsdict = np.load(COORDINATES).item()
#     imgcounter = 0
#     frameaddition = int(ROI_SIZE / 2)
#     for imgname in os.listdir(ORIGINAL_IMAGES_DIR):
#         path = os.path.join(ORIGINAL_IMAGES_DIR, imgname)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         rows, cols = img.shape
#         points = dotsdict[imgcounter]
#         print('image', imgcounter)
#         for imgangle in range(0, 360, 5):
#             Mod = cv2.getRotationMatrix2D((cols / 2, rows / 2), imgangle, 1)
#             RotatedImage = cv2.warpAffine(img, Mod, (cols, rows))
#
#             ones = np.ones(shape=(len(points), 1))
#             points_ones = np.hstack([points, ones])
#             transformed_points = Mod.dot(points_ones.T).T
#
#             for transformed_counter in range(len(transformed_points)):
#                 x, y = int(transformed_points[transformed_counter][0]), int(transformed_points[transformed_counter][1])
#
#                 if y <= frameaddition: y += frameaddition - y
#                 elif y >= IMG_SIZE - frameaddition: y -= frameaddition - (IMG_SIZE - y)
#                 if x <= frameaddition: x += frameaddition - x
#                 elif x >= IMG_SIZE - frameaddition: x -= frameaddition - (IMG_SIZE - x)
#
#                 ROI = RotatedImage[y - frameaddition:y + frameaddition, x - frameaddition:x + frameaddition]
#
#         cv2.imwrite(SAVEDIR + imgname + 'PD' + str(transformed_counter) + 'ANG' + str(imgangle) + '.png', ROI)
#         imgcounter += 1
