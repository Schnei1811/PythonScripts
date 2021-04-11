import numpy as np
import cv2
import tqdm

videoname = '1chargingbehaviour'

#Full Length Videos
videoname = 'GO011675'

capture = cv2.VideoCapture('D:PythonData/Octopus/RawVideos/{}.mp4'.format(videoname))
maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(capture.get(3)), int(capture.get(4))

squares = np.loadtxt('D:PythonData/Octopus/OutputCSV/{}frcnnboxes.csv'.format(videoname), delimiter=',').astype(int)

#squares = y2, x1, x2, y1, framenum
framenum = 0
for framenum in tqdm.trange(0, maxnumframes):
    ret, frame = capture.read()
    cv2.imwrite('Frames/{}.jpg'.format(framenum), frame)
    for i, j in enumerate(squares):
        if squares[i, 4] == framenum:
            y1, x1, y2, x2 = squares[i,0], squares[i,1], squares[i,2], squares[i,3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)
    cv2.imshow('original', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: break

capture.release()
cv2.destroyAllWindows()