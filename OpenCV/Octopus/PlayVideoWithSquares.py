import numpy as np
import cv2

threshold = 8        # Lower = less difference from mean value
blur = 100           # After blurring, pixel value cut off
squaresize = 0       # Square Sizes

videoname = '1chargingbehaviour'
#videoname = '2corrallingbehaviour'
#videoname = '3corrallingfromanotherangle'
#videoname = '4successfulescape'
#videoname = '5shortclip'
#videoname = 'GO010141'
#videoname = 'test'

capture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
try: squares = np.loadtxt('Files/{}/{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)
except FileNotFoundError: squares = np.loadtxt('Files/{}/temp{}{}{}VideoBoxesArray.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)


framenum = 0
while True:
    ret, frame = capture.read()
    for i, j in enumerate(squares):
        if squares[i, 0] == framenum and (squares[i, 5] - squares[i, 3]) * (squares[i, 4] - squares[i, 2]) > 1000:
            cv2.rectangle(frame, (squares[i, 2], squares[i, 3]), (squares[i, 4], squares[i, 5]), 255, 2)
    cv2.imshow('original', frame)
    framenum += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27: break

capture.release()
cv2.destroyAllWindows()














