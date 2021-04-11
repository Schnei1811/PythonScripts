import cv2
import numpy as np
import os

img_array = []
image_dir = 'D:PythonData/Peru/ProcessedVideo/003Razor Billed Curassow Mitu tuberoso11/'

for file in os.listdir(image_dir):
    if len(file.split('.')[0]) == 1:
        os.rename(image_dir + file, image_dir + '00' + file)
    if len(file.split('.')[0]) == 2:
        os.rename(image_dir + file, image_dir + '0' + file)




for filename in os.listdir('D:PythonData/Peru/ProcessedVideo/003Razor Billed Curassow Mitu tuberoso11/'):
    print(filename)
    img = cv2.imread('D:PythonData/Peru/ProcessedVideo/003Razor Billed Curassow Mitu tuberoso11/' + filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('Boxes-003Razor Billed Burassow Mitu tuberoso11.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()