import cv2
import tqdm

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
SAVE_DIR = 'D:PythonData/Octopus/Octo5Frame/'

capture = cv2.VideoCapture('D:PythonData/Octopus/RawVideos/{}.mp4'.format(videoname))

#img = frame[y1:y2, x1:x2]
#cv2.rectangle(x1, y1, x2, y2)

maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
for framenum in tqdm.trange(0, maxnumframes):
    ret, frame = capture.read()
    Octo1 = 'Octo1'
    #BottomLeftOcto
    if 150 < framenum < 211:
        x1, y1, x2, y2 = 500, 650, 700, 900
        #cv2.rectangle(frame, (500, 650), (700, 900), 255, 2)
    elif 210 < framenum < 250:
        x1, y1, x2, y2 = 500, 690, 700, 900
        #cv2.rectangle(frame, (500, 690), (700, 900), 255, 2)
    else:
        x1, y1, x2, y2 = 500, 730, 700, 900
        #cv2.rectangle(frame, (500, 730), (700, 900), 255, 2)
    img1 = frame[y1:y2, x1:x2]

    #SmallOcto
    Octo2 = 'Octo2'
    if 1340 < framenum < 1400:
        x1, y1, x2, y2 = 890, 520, 980, 610
        #cv2.rectangle(frame, (890, 520), (980, 610), 255, 2)
    else:
        x1, y1, x2, y2 = 890, 550, 950, 610
        #cv2.rectangle(frame, (890, 550), (950, 610), 255, 2)
    img2 = frame[y1:y2, x1:x2]

    #BackRightOcto
    Octo3 = 'Octo3'
    if 80 < framenum < 250:
        x1, y1, x2, y2 = 1120, 470, 1320, 600
        #cv2.rectangle(frame, (1120, 470), (1320, 600), 255, 2)
    elif 720 < framenum < 871:
        x1, y1, x2, y2 = 1120, 440, 1290, 600
        #cv2.rectangle(frame, (1120, 440), (1290, 600), 255, 2)
    elif 870 < framenum < 1250:
        x1, y1, x2, y2 = 1120, 500, 1330, 600
        #cv2.rectangle(frame, (1120, 500), (1330, 600), 255, 2)
    else:
        x1, y1, x2, y2 = 1120, 500, 1230, 600
        #cv2.rectangle(frame, (1120, 500), (1230, 600), 255, 2)
    img3 = frame[y1:y2, x1:x2]

    #LargeArmWavingOcto
    Octo4 = 'Octo4'
    if framenum < 105:
        x1, y1, x2, y2 = 745, 640, 880, 731
        #cv2.rectangle(frame, (745, 640), (880, 731), 255, 2)
    elif 250 < framenum < 320:
        x1, y1, x2, y2 = 745, 580, 920, 730
        #cv2.rectangle(frame, (745, 580), (920, 730), 255, 2)
    elif 590 < framenum < 650:
        x1, y1, x2, y2 = 745, 560, 900, 730
        #cv2.rectangle(frame, (745, 560), (900, 730), 255, 2)
    elif 760 < framenum < 811:
        x1, y1, x2, y2 = 745, 580, 900, 730
        #cv2.rectangle(frame, (745, 580), (900, 730), 255, 2)
    elif 810 < framenum < 1141:
        x1, y1, x2, y2 = 745, 640, 870, 730
        #cv2.rectangle(frame, (745, 640), (870, 730), 255, 2)
    elif 1140 < framenum < 1210:
        x1, y1, x2, y2 = 745, 500, 950, 730
        #cv2.rectangle(frame, (745, 500), (950, 730), 255, 2)
    elif 1235 < framenum < 1341:
        x1, y1, x2, y2 = 700, 520, 950, 730
        #cv2.rectangle(frame, (700, 520), (950, 730), 255, 2)
    elif 1420 < framenum < 2708:
        x1, y1, x2, y2 = 745, 640, 870, 730
        #cv2.rectangle(frame, (745, 640), (870, 730), 255, 2)
    else:
        x1, y1, x2, y2 = 745, 640, 940, 730
        #cv2.rectangle(frame, (745, 640), (940, 730), 255, 2)
    img4 = frame[y1:y2, x1:x2]

    #ChargingOcto
    Octo5 = 'Octo5'
    if 150 < framenum < 401:
        x1, y1, x2, y2 = 780, 520, 940, 650
        #cv2.rectangle(frame, (780, 520), (940, 650), 255, 2)
    elif 590 < framenum < 631:
        x1, y1, x2, y2 = 840, 520, 1000, 680
        #cv2.rectangle(frame, (840, 520), (1000, 680), 255, 2)
    elif 630 < framenum < 671:
        x1, y1, x2, y2 = 880, 520, 1050, 690
        #cv2.rectangle(frame, (880, 520), (1050, 690), 255, 2)
    elif 670 < framenum < 701:
        x1, y1, x2, y2 = 900, 520, 1080, 690
        #cv2.rectangle(frame, (900, 520), (1080, 690), 255, 2)
    elif 700 < framenum < 761:
        x1, y1, x2, y2 = 940, 520, 1130, 690
        #cv2.rectangle(frame, (940, 520), (1130, 690), 255, 2)
    elif 760 < framenum < 821:
        x1, y1, x2, y2 = 960, 490, 1200, 700
        #cv2.rectangle(frame, (960, 490), (1200, 700), 255, 2)
    elif 820 < framenum < 831:
        x1, y1, x2, y2 = 1000, 480, 1200, 700
        #cv2.rectangle(frame, (1000, 480), (1200, 700), 255, 2)
    elif 830 < framenum < 851:
        x1, y1, x2, y2 = 1040, 480, 1300, 700
        #cv2.rectangle(frame, (1040, 480), (1300, 700), 255, 2)
    elif 850 < framenum < 861:
        x1, y1, x2, y2 = 1080, 480, 1340, 700
        #cv2.rectangle(frame, (1080, 480), (1340, 700), 255, 2)
    elif 860 < framenum < 881:
        x1, y1, x2, y2 = 1140, 520, 1440, 700
        #cv2.rectangle(frame, (1140, 520), (1440, 700), 255, 2)
    elif 880 < framenum < 891:
        x1, y1, x2, y2 = 1140, 540, 1480, 700
        #cv2.rectangle(frame, (1140, 540), (1480, 700), 255, 2)
    elif 890 < framenum < 911:
        x1, y1, x2, y2 = 1140, 540, 1550, 740
        #cv2.rectangle(frame, (1140, 540), (1550, 740), 255, 2)
    elif 910 < framenum < 921:
        x1, y1, x2, y2 = 1180, 560, 1640, 760
        #cv2.rectangle(frame, (1180, 560), (1640, 760), 255, 2)
    elif 920 < framenum < 951:
        x1, y1, x2, y2 = 1180, 620, 1640, 800
        #cv2.rectangle(frame, (1180, 620), (1640, 800), 255, 2)
    elif 950 < framenum < 981:
        x1, y1, x2, y2 = 1260, 600, 1620, 820
        #cv2.rectangle(frame, (1260, 600), (1620, 820), 255, 2)
    elif 980 < framenum < 1001:
        x1, y1, x2, y2 = 1240, 600, 1600, 800
        #cv2.rectangle(frame, (1240, 600), (1600, 800), 255, 2)
    elif 1000 < framenum < 1031:
        x1, y1, x2, y2 = 1240, 600, 1600, 800
        #cv2.rectangle(frame, (1240, 600), (1600, 800), 255, 2)
    elif 1030 < framenum < 1041:
        x1, y1, x2, y2 = 1200, 600, 1560, 780
        #cv2.rectangle(frame, (1200, 600), (1560, 780), 255, 2)
    elif 1040 < framenum < 1051:
        x1, y1, x2, y2 = 1180, 590, 1520, 760
        #cv2.rectangle(frame, (1180, 590), (1520, 760), 255, 2)
    elif 1050 < framenum < 1061:
        x1, y1, x2, y2 = 1140, 570, 1480, 740
        #cv2.rectangle(frame, (1140, 570), (1480, 740), 255, 2)
    elif 1060 < framenum < 1091:
        x1, y1, x2, y2 = 1120, 560, 1440, 720
        #cv2.rectangle(frame, (1120, 560), (1440, 720), 255, 2)
    elif 1090 < framenum < 1111:
        x1, y1, x2, y2 = 1100, 550, 1400, 700
        #cv2.rectangle(frame, (1100, 550), (1400, 700), 255, 2)
    elif 1110 < framenum < 1151:
        x1, y1, x2, y2 = 1080, 550, 1360, 700
        #cv2.rectangle(frame, (1080, 550), (1360, 700), 255, 2)
    elif 1150 < framenum < 1231:
        x1, y1, x2, y2 = 1040, 530, 1300, 700
        #cv2.rectangle(frame, (1040, 530), (1300, 700), 255, 2)
    elif 1230 < framenum < 1261:
        x1, y1, x2, y2 = 1000, 530, 1240, 700
        #cv2.rectangle(frame, (1000, 530), (1240, 700), 255, 2)
    elif 1260 < framenum < 1301:
        x1, y1, x2, y2 = 960, 530, 1200, 680
        #cv2.rectangle(frame, (960, 530), (1200, 680), 255, 2)
    elif 1300 < framenum < 1341:
        x1, y1, x2, y2 = 920, 530, 1150, 650
        #cv2.rectangle(frame, (920, 530), (1150, 650), 255, 2)
    elif 1340 < framenum < 1381:
        x1, y1, x2, y2 = 880, 530, 1110, 650
        #cv2.rectangle(frame, (880, 530), (1110, 650), 255, 2)
    elif 1380 < framenum < 1421:
        x1, y1, x2, y2 = 840, 530, 1070, 650
        #cv2.rectangle(frame, (840, 530), (1070, 650), 255, 2)
    elif 1420 < framenum < 1461:
        x1, y1, x2, y2 = 800, 530, 1030, 650
        #cv2.rectangle(frame, (800, 530), (1030, 650), 255, 2)
    elif 1460 < framenum < 2701:
        x1, y1, x2, y2 = 780, 560, 940, 650
        #cv2.rectangle(frame, (780, 560), (940, 650), 255, 2)
    else:
        x1, y1, x2, y2 = 800, 545, 940, 650
        #cv2.rectangle(frame, (800, 545), (940, 650), 255, 2)
    img5 = frame[y1:y2, x1:x2]

    #IntrudingOcto
    Octo6 = 'Octo6'
    if 130 < framenum < 820:
        x1, y1, x2, y2 = 1550, 738, 1820, 940
        #cv2.rectangle(frame, (1550, 738), (1820, 940), 255, 2)
        img6 = frame[y1:y2, x1:x2]
        if framenum % 5 == 0:
            cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo6, framenum), img6)

    if framenum % 5 == 0:
        cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo1, framenum), img1)
        cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo2, framenum), img2)
        cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo3, framenum), img3)
        cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo4, framenum), img4)
        cv2.imwrite(SAVE_DIR + '/{}.{}.jpg'.format(Octo5, framenum), img5)


    #cv2.imshow('original', frame)
    #k = cv2.waitKey(30) & 0xff
    #if k == 27: break