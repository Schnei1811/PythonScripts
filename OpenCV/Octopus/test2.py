import numpy as np
import cv2
import scipy.sparse

def image_expand(y1, y2, x1, x2):
    if y2 - y1 > x2 - x1:
        dimensiondiff = (y2 - y1) - (x2 - x1)
        x1 -= int(dimensiondiff / 2)
        x2 += int(dimensiondiff / 2)
        if (y2 - y1) - (x2 - x1) != 0: x2 += 1
    elif x2 - x1 > y2 - y1:
        dimensiondiff = (x2 - x1) - (y2 - y1)
        y1 -= int(dimensiondiff / 2)
        y2 += int(dimensiondiff / 2)
        if (x2 - x1) - (y2 - y1) != 0: y2 += 1
    if y2 > height: y2 = height
    if y1 < 0: y1 = 0
    if x2 > width: x2 = width
    if x1 < 0: x1 = 0
    y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
    return y1, y2, x1, x2

def post_process_sqaures(squares):
    i = 0
    while True:
        if i == 0: stationarysquares = squares[i]
        elif squares[i, 0] == 0: stationarysquares = np.vstack((squares[i], stationarysquares))
        if squares[i, 0] == 1: break
        i += 1

    framenum = 0
    while framenum < maxnumframes:
        print(framenum)
        for i, j in enumerate(squares):
            if squares[i, 0] == framenum:
                for k, l in enumerate(squares):
                    if squares[k, 0] == framenum + 1:
                        totalx1 = squares[k, 2] - squares[i, 2]
                        totaly1 = squares[k, 3] - squares[i, 3]
                        totalx2 = squares[k, 4] - squares[i, 4]
                        totaly2 = squares[k, 5] - squares[i, 5]
                        totalsum = abs(totalx1 + totaly1 + totalx2 + totaly2)
                        if totalsum < 100:
                            squares[i, 2] = (squares[i, 2] + squares[k, 2]) / 2
                            squares[i, 3] = (squares[i, 3] + squares[k, 3]) / 2
                            squares[i, 4] = (squares[i, 4] + squares[k, 4]) / 2
                            squares[i, 5] = (squares[i, 5] + squares[k, 5]) / 2
        framenum += 1
    np.savetxt('Files/{}/{}{}{}{}{}TESTVideoBoxID.txt'.format(videoname, threshold, blur, squaresize, INITIAL_SCAN_SIZE, STEP_SIZE), squares, fmt='%i', delimiter=',')
    return squares

def stationary_squares(squares):
    i = 0
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        if i == 0:
            stationarysquares = squares[i]
        elif squares[i, 0] == 0:
            stationarysquares = np.vstack((squares[i], stationarysquares))
        if squares[i, 0] == 1: break
        i += 1

    newsquarescounter = 0
    for i, j in enumerate(squares):
        #print(i)
        squarefound = False
        for k, l in enumerate(stationarysquares):
            totalx1 = stationarysquares[k, 2] - squares[i, 2]
            totaly1 = stationarysquares[k, 3] - squares[i, 3]
            totalx2 = stationarysquares[k, 4] - squares[i, 4]
            totaly2 = stationarysquares[k, 5] - squares[i, 5]
            totalx = abs(totalx1 + totalx2)
            totaly = abs(totaly1 + totaly2)
            if totalx < 30 and totaly < 30 and (squares[i, 5] - squares[i, 3]) * (squares[i, 4] - squares[i, 2]) < 75000:
                stationarysquares[k, 1] += 10
                squarefound = True
            if stationarysquares[k, 1] > 250:
                stationarysquares[k, 0] = squares[i, 0]
                # y1, y2, x1, x2 = image_expand(squares[i, 3], squares[i, 5], squares[i, 2], squares[i, 4])
                # img = gray[y1:y2, x1:x2]
                # roi = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(IMG_SIZE, IMG_SIZE, 1)
                # init_model_out = model.predict([roi])[0]
                # print(init_model_out, np.argmax(init_model_out))
                if newsquarescounter == 0:
                    newsquares = stationarysquares[k]
                    newsquarescounter += 1
                else: newsquares = np.vstack((stationarysquares[k], newsquares))

        stationarysquares = stationarysquares[stationarysquares.min(axis=1) >= -20, :]
        if squarefound == False: stationarysquares = np.vstack((squares[i], stationarysquares))
        if i == 4000: break
        if squares[i + 1, 0] - squares[i, 0] > 0: stationarysquares[:, 1] -= 1

        if squares[i + 1, 0] - squares[i, 0] > 0 and 'newsquares' in locals():
            intermediatesquares = scipy.delete(newsquares, 1, 1)
            intermediatesquares = np.vstack({tuple(row) for row in intermediatesquares})

    intermediatesquares = intermediatesquares[np.argsort(intermediatesquares[:, 0])]

    for i, j in enumerate(intermediatesquares):
        if i == len(intermediatesquares) - 1: break
        if i == 0: consideredsquares = intermediatesquares[i]
        elif intermediatesquares[i, 0] - intermediatesquares[i - 1, 0] == 0:
            consideredsquares = np.vstack((intermediatesquares[i], consideredsquares))
        if intermediatesquares[i, 0] - intermediatesquares[i - 1, 0] > 0:
            if consideredsquares.ndim == 1 and 'finalsquares' not in locals(): finalfinalsqaures = consideredsquares
            elif consideredsquares.ndim == 1 and 'finalsquares' in locals(): finalfinalsqaures = np.vstacks((consideredsquares, finalfinalsqaures))
            elif consideredsquares.ndim == 2:
                # Create Adjacency Matrix
                framenum = consideredsquares[0, 0]
                consideredsquares = consideredsquares[:, 1:]

                intersectiongraph = np.zeros((len(consideredsquares), len(consideredsquares)))
                for q in range(0, len(consideredsquares)):
                    for r in range(0, len(consideredsquares)):
                        if q == r: intersectiongraph[q, r] = 0
                        elif (consideredsquares[q, 2] < consideredsquares[r, 0] or consideredsquares[r, 2] < consideredsquares[q, 0]
                              or consideredsquares[q, 3] < consideredsquares[r, 1] or consideredsquares[r, 3] < consideredsquares[q, 1]):
                            intersectiongraph[q, r] = 0
                        else: intersectiongraph[q, r] = 1

                # Determine Strongly Connected Components
                uniquesquares = np.array([scipy.sparse.csgraph.connected_components(intersectiongraph, directed=False, connection='weak', return_labels=True)[1]])
                consideredsquares = np.concatenate((uniquesquares.T, consideredsquares), axis=1)
                consideredsquares = consideredsquares[np.argsort(consideredsquares[:, 0])]

                # Determine Min/Max of SCCs
                squarecount = 0
                minx, miny, maxx, maxy = width, height, 0, 0
                for q, r in enumerate(consideredsquares):
                    # print(squarecount)
                    if consideredsquares[q, 1] < minx: minx = consideredsquares[q, 1]
                    if consideredsquares[q, 2] < miny: miny = consideredsquares[q, 2]
                    if consideredsquares[q, 3] > maxx: maxx = consideredsquares[q, 3]
                    if consideredsquares[q, 4] > maxy: maxy = consideredsquares[q, 4]
                    # print(minx, miny, maxx, maxy)
                    try:
                        if consideredsquares[q + 1, 0] - consideredsquares[q, 0] > 0:
                            if squarecount == 0: finalsquares = np.array([framenum, 0, minx, miny, maxx, maxy])
                            else: finalsquares = np.vstack((np.array([framenum, 0, minx, miny, maxx, maxy]), finalsquares))
                            squarecount += 1
                            minx, miny, maxx, maxy = width, height, 0, 0
                    except:
                        if squarecount == 0: finalsquares = np.array([framenum, 0, minx, miny, maxx, maxy])
                        else: finalsquares = np.vstack((np.array([framenum, 0, minx, miny, maxx, maxy]), finalsquares))
                        squarecount += 1
                        minx, miny, maxx, maxy = width, height, 0, 0
                squares = np.vstack((finalsquares, squares))
            consideredsquares = intermediatesquares[i]
    return squares

def show_octopus_id(squares):
    framenum = 0
    while framenum < maxnumframes:
        ret, frame = capture.read()
        for i, j in enumerate(squares):
            if squares[i, 0] == framenum:
                cv2.rectangle(frame, (squares[i, 2], squares[i, 3]), (squares[i, 4], squares[i, 5]), 255, 2)
        cv2.imshow('original', frame)
        framenum += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

threshold = 8               # Lower = less difference from mean value
blur = 100                  # After blurring, pixel value cut off
squaresize = 12             # Square Sizes
INITIAL_SCAN_SIZE = 150
STEP_SIZE = 20

videoname = '1chargingbehaviour'
#videoname = '2corrallingbehaviour'
#videoname = '3corrallingfromanotherangle'
#videoname = '4successfulescape'
#videoname = '5shortclip'
#videoname = 'GO010141'
#videoname = 'test'


IMG_SIZE = 150
LR = 0.0001
NUM_CLASSIFICATIONS = 6
MODEL_NAME = '{}-{}.model'.format(LR, 'VGG16')
#MODEL_NAME = 'Files/Models/6ClassVGG16.model'.format(LR, 'VGG16')


# Building 'VGG Network'
# network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# network = conv_2d(network, 64, 3, activation='relu')
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
# network = conv_2d(network, 128, 3, activation='relu')
# network = conv_2d(network, 128, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
# network = conv_2d(network, 256, 3, activation='relu')
# network = conv_2d(network, 256, 3, activation='relu')
# network = conv_2d(network, 256, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
# network = fully_connected(network, 4096, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 4096, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, NUM_CLASSIFICATIONS, activation='softmax')
# network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='targets')
# model = tflearn.DNN(network, tensorboard_dir='log')





capture = cv2.VideoCapture('Files/Videos/{}.mp4'.format(videoname))
maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(capture.get(3)), int(capture.get(4))
#squares = np.loadtxt('Files/{}/{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)
#squares = np.loadtxt('Files/{}/{}{}{}TestVideoBoxID.txt'.format(videoname, threshold, blur, squaresize), delimiter=',').astype(int)
squares = np.loadtxt('Files/{}/{}{}{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize, INITIAL_SCAN_SIZE, STEP_SIZE), delimiter=',').astype(int)
#squares = np.loadtxt('Files/{}/{}{}{}{}{}TESTVideoBoxID.txt'.format(videoname, threshold, blur, squaresize, INITIAL_SCAN_SIZE, STEP_SIZE), delimiter=',').astype(int)


#squares = post_process_sqaures(squares)
squares = stationary_squares(squares)

show_octopus_id(squares)

capture.release()
cv2.destroyAllWindows()
