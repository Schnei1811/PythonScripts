import cv2
import tqdm
import os
import numpy as np


# def determine_rectangles():
#     training_data = []
#     for img in os.listdir(TRAIN_DIR):
#         print(img)
#         path = os.path.join(TRAIN_DIR, img)
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)
#         cv2.imshow('img', img)
#         cv2.waitKey()
#         cv2.destroyAllWindows
#     return

#determine_rectangles()

videoname = '1chargingbehaviour'
TRAIN_DIR = 'Images/'

capture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))

imgnum = 0
maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
for framenum in tqdm.trange(0, maxnumframes):
    ret, frame = capture.read()
    #BottomLeftOcto
    if 150 < framenum < 211: x1, y1, x2, y2 = 500, 650, 700, 900
    elif 210 < framenum < 250: x1, y1, x2, y2 = 500, 690, 700, 900
    else: x1, y1, x2, y2 = 500, 730, 700, 900
    img = frame[y1:y2, x1:x2]
    cv2.imwrite('Images/Octo1.{}.jpg'.format(imgnum), img)
    if 'octocoordinates' not in locals(): octocoordinates = np.array([framenum, y1, x1, y2, x2, 0])
    else: octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 0]), octocoordinates))
    imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

    #SmallOcto
    if 1340 < framenum < 1400: x1, y1, x2, y2 = 890, 520, 980, 610
    else: x1, y1, x2, y2 = 890, 550, 950, 610
    img = frame[y1:y2, x1:x2]
    cv2.imwrite('Images/Octo2.{}.jpg'.format(imgnum), img)
    octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 1]), octocoordinates))
    imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

    #BackRightOcto
    if 80 < framenum < 250: x1, y1, x2, y2 = 1120, 470, 1320, 600
    elif 720 < framenum < 871: x1, y1, x2, y2 = 1120, 440, 1290, 600
    elif 870 < framenum < 1250: x1, y1, x2, y2 = 1120, 500, 1330, 600
    else: x1, y1, x2, y2 = 1120, 500, 1230, 600
    img = frame[y1:y2, x1:x2]
    cv2.imwrite('Images/Octo3.{}.jpg'.format(imgnum), img)
    octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 2]), octocoordinates))
    imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

    #LargeArmWavingOcto
    if framenum < 105: x1, y1, x2, y2 = 745, 640, 880, 730
    elif 250 < framenum < 320: x1, y1, x2, y2 = 745, 580, 920, 730
    elif 590 < framenum < 650: x1, y1, x2, y2 = 745, 560, 900, 730
    elif 760 < framenum < 811: x1, y1, x2, y2 = 745, 580, 900, 730
    elif 810 < framenum < 1141: x1, y1, x2, y2 = 745, 640, 870, 730
    elif 1140 < framenum < 1210: x1, y1, x2, y2 = 745, 500, 950, 730
    elif 1235 < framenum < 1341: x1, y1, x2, y2 = 700, 520, 950, 730
    elif 1420 < framenum < 2708: x1, y1, x2, y2 = 745, 640, 870, 730
    else: x1, y1, x2, y2 = 745, 640, 940, 730
    img = frame[y1:y2, x1:x2]
    cv2.imwrite('Images/Octo4.{}.jpg'.format(imgnum), img)
    octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 3]), octocoordinates))
    imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

    #ChargingOcto
    if 150 < framenum < 401: x1, y1, x2, y2 = 780, 520, 940, 650
    elif 590 < framenum < 631: x1, y1, x2, y2 = 840, 520, 1000, 680
    elif 630 < framenum < 671: x1, y1, x2, y2 = 880, 520, 1050, 690
    elif 670 < framenum < 701: x1, y1, x2, y2 = 900, 520, 1080, 690
    elif 700 < framenum < 761: x1, y1, x2, y2 = 940, 520, 1130, 690
    elif 760 < framenum < 821: x1, y1, x2, y2 = 960, 490, 1200, 700
    elif 820 < framenum < 831: x1, y1, x2, y2 = 1000, 480, 1200, 700
    elif 830 < framenum < 851: x1, y1, x2, y2 = 1040, 480, 1300, 700
    elif 850 < framenum < 861: x1, y1, x2, y2 = 1080, 480, 1340, 700
    elif 860 < framenum < 881: x1, y1, x2, y2 = 1140, 520, 1440, 700
    elif 880 < framenum < 891: x1, y1, x2, y2 = 1140, 540, 1480, 700
    elif 890 < framenum < 911: x1, y1, x2, y2 = 1140, 540, 1550, 740
    elif 910 < framenum < 921: x1, y1, x2, y2 = 1180, 560, 1640, 760
    elif 920 < framenum < 951: x1, y1, x2, y2 = 1180, 620, 1640, 800
    elif 950 < framenum < 981: x1, y1, x2, y2 = 1260, 600, 1620, 820
    elif 980 < framenum < 1001: x1, y1, x2, y2 = 1240, 600, 1600, 800
    elif 1000 < framenum < 1031: x1, y1, x2, y2 = 1240, 600, 1600, 800
    elif 1030 < framenum < 1041: x1, y1, x2, y2 = 1200, 600, 1560, 780
    elif 1040 < framenum < 1051: x1, y1, x2, y2 = 1180, 590, 1520, 760
    elif 1050 < framenum < 1061: x1, y1, x2, y2 = 1140, 570, 1480, 740
    elif 1060 < framenum < 1091: x1, y1, x2, y2 = 1120, 560, 1440, 720
    elif 1090 < framenum < 1111: x1, y1, x2, y2 = 1100, 550, 1400, 700
    elif 1110 < framenum < 1151: x1, y1, x2, y2 = 1080, 550, 1360, 700
    elif 1150 < framenum < 1231: x1, y1, x2, y2 = 1040, 530, 1300, 700
    elif 1230 < framenum < 1261: x1, y1, x2, y2 = 1000, 530, 1240, 700
    elif 1260 < framenum < 1301: x1, y1, x2, y2 = 960, 530, 1200, 680
    elif 1300 < framenum < 1341: x1, y1, x2, y2 = 920, 530, 1150, 650
    elif 1340 < framenum < 1381: x1, y1, x2, y2 = 880, 530, 1110, 650
    elif 1380 < framenum < 1421: x1, y1, x2, y2 = 840, 530, 1070, 650
    elif 1420 < framenum < 1461: x1, y1, x2, y2 = 800, 530, 1030, 650
    elif 1460 < framenum < 2701: x1, y1, x2, y2 = 780, 560, 940, 650
    else: x1, y1, x2, y2 = 800, 545, 940, 650
    img = frame[y1:y2, x1:x2]
    cv2.imwrite('Images/Octo5.{}.jpg'.format(imgnum), img)
    octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 4]), octocoordinates))
    imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

    #IntrudingOcto
    if 130 < framenum < 820:
        x1, y1, x2, y2 = 1550, 740, 1820, 940
        img = frame[y1:y2, x1:x2]
        cv2.imwrite('Images/Octo6.{}.jpg'.format(imgnum), img)
        octocoordinates = np.vstack((np.array([framenum, y1, x1, y2, x2, 5]), octocoordinates))
        imgnum += 1
    #cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)
    #cv2.imshow('original', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: break

octocoordinates = octocoordinates[np.argsort(octocoordinates[:, 0])]
np.savetxt('OctoCoordinatesVideoOne.txt', octocoordinates, delimiter=',', fmt='%i')