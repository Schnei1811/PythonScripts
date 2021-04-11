import cv2
import numpy as np
import tqdm


def video(videoname):
    capture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
    maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for framenum in tqdm.trange(0, maxnumframes):
        ret, frame = capture.read()
        if framenum % 500 == 0: cv2.imwrite('ObjectDetectionFrames/{}{}.jpg'.format(videoname, framenum), frame)


#video('1chargingbehaviour')
#video('2corrallingbehaviour')
#video('3corrallingfromanotherangle')
#video('4successfulescape')
video('GO010141')
video('GO020142')
video('GO040170')