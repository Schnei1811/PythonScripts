import numpy as np
from scipy.spatial import distance
import cv2

videoname = '1chargingbehaviour'
#videoname = '2corrallingbehaviour'
# videoname = '3corrallingfromanotherangle'
# videoname = '4successfulescape'


data = np.loadtxt('D:PythonData/Octopus/CSV/{}frcnnboxes.csv'.format(videoname), delimiter=',')
data = data.astype(int)

maxnumframes = data[len(data)-1][4]



i = 0
frame = 0

octodict = {}
tenativeoctodict = {}


while frame < maxnumframes:
    while frame == data[i][4]:
        if frame == 0: octodict[i] = data[i]
        else:
            for octo in range(len(octodict)):
                mindist = 10000
                dist = distance.euclidean(octodict[octo][0:2], data[i][0:2]) + \
                       distance.euclidean(octodict[octo][2:4], data[i][2:4])
                if dist < mindist:
                    mindist = dist
                    minocto = data[i]
                if mindist < 65:
                    octodict[octo] = minocto
        i += 1
    for octo in octodict:
        octodict[octo][4] = frame
        if 'outputdata' not in locals(): outputdata = np.append([octodict[octo]], [octo])
        else: outputdata = np.vstack((outputdata, np.append([octodict[octo]], [octo])))
    frame += 1



# while frame < maxnumframes:
#     middledata = []
#     while frame == data[i][4]:
#         middledata.append(data[i])
#         i += 1
#
#     if frame == 0:
#         for k in range(len(middledata)):
#             octodict[k] = middledata[k]
#     else:
#         for octo in range(len(octodict)):
#             for k in range(len(middledata)):
#                 mindist = 10000
#                 dist = distance.euclidean(octodict[octo][0:2], middledata[k][0:2]) + \
#                        distance.euclidean(octodict[octo][2:4], middledata[k][2:4])
#                 if dist < mindist:
#                     mindist = dist
#                     minocto = data[i][0:5]
#                 if mindist < 65:
#                     del middledata[k]
#                     octodict[octo] = minocto
#
#     for octo in octodict:
#         if 'outputdata' not in locals():
#             outputdata = np.append([octodict[octo]], [octo])
#         else:
#             outputdata = np.vstack((outputdata, np.append([octodict[octo]], [octo])))
#     frame += 1

np.set_printoptions(threshold=np.nan)
print(outputdata)


capture = cv2.VideoCapture('D:PythonData/Octopus/RawVideos/{}.mp4'.format(videoname))

framenum = 0
while True:
    ret, frame = capture.read()
    for i, j in enumerate(outputdata):
        if outputdata[i, 4] == framenum:
            # print(outputdata[i])
            if outputdata[i, 5] == 0:
                cv2.rectangle(frame, (outputdata[i, 1], outputdata[i, 0]), (outputdata[i, 3], outputdata[i, 2]), (255, 255, 255), 2)
            if outputdata[i, 5] == 1:
                cv2.rectangle(frame, (outputdata[i, 1], outputdata[i, 0]), (outputdata[i, 3], outputdata[i, 2]), (0, 255, 255), 2)
            if outputdata[i, 5] == 2:
                cv2.rectangle(frame, (outputdata[i, 1], outputdata[i, 0]), (outputdata[i, 3], outputdata[i, 2]), (255, 0, 255), 2)
            if outputdata[i, 5] == 3:
                cv2.rectangle(frame, (outputdata[i, 1], outputdata[i, 0]), (outputdata[i, 3], outputdata[i, 2]), (255, 255, 0), 2)
            if outputdata[i, 5] == 4:
                cv2.rectangle(frame, (outputdata[i, 1], outputdata[i, 0]), (outputdata[i, 3], outputdata[i, 2]), (0, 0, 255), 2)
    cv2.imshow('original', frame)
    framenum += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27: break

capture.release()
cv2.destroyAllWindows()















