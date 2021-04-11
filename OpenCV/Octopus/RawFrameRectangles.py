import os
import numpy as np
import cv2
import tqdm

boxes = np.loadtxt('OctoCoordinatesVideoOne.txt', delimiter=',').astype(int)

for img in tqdm.tqdm(os.listdir('Frames/RawFrames')):
    path = os.path.join('Frames/RawFrames', img)
    framenum = int(img[:-4])
    img = cv2.imread(path)
    for i, j in enumerate(boxes):
        if boxes[i, 0] == framenum:
            cv2.rectangle(img, (boxes[i, 3], boxes[i, 2]), (boxes[i, 5], boxes[i, 4]), 255, 2)
    cv2.imwrite('Frames/BoxFrames/{}.jpg'.format(framenum), img)