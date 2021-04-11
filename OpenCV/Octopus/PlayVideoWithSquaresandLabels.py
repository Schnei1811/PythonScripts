import numpy as np
import cv2

threshold = 8       # Lower = less difference from mean value
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
squares = np.loadtxt('Files/{}/{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)

font = cv2.FONT_HERSHEY_COMPLEX
framenum = 0
while True:
    ret, frame = capture.read()
    for i,j in enumerate(squares):
        if squares[i, 0] == framenum:
            cv2.rectangle(frame, (squares[i, 2], squares[i, 3]), (squares[i,4], squares[i, 5]), 255, 2)
            if squares[i, 6] == 0: str_label = 'Diver'
            elif squares[i, 6] == 1: str_label = 'Fish'
            elif squares[i, 6] == 2: str_label = 'Kelp'
            else: str_label = 'Octopus'
            cv2.putText(frame, '{}'.format(str_label), (int(squares[i, 2] + 0.2*(squares[i, 4] - squares[i, 2])), squares[i, 5]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('original', frame)
    framenum += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()














