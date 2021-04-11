import numpy as np
np.set_printoptions(threshold=np.nan)

threshold = 8               # Lower = less difference from mean value
blur = 100                  # After blurring, pixel value cut off
squaresize = 12             # Square Sizes
INITIAL_SCAN_SIZE = 150
STEP_SIZE = 20
videoname = '1chargingbehaviour'

squares = np.loadtxt('Files/{}/{}{}{}{}{}VideoBoxID.txt'.format(videoname, threshold, blur, squaresize, INITIAL_SCAN_SIZE, STEP_SIZE), delimiter=',').astype(int)

i = 0
while True:
    if i == 0: stationarysquares = squares[i]
    elif squares[i, 0] == 0: stationarysquares = np.vstack((squares[i], stationarysquares))
    if squares[i, 0] == 1: break
    i += 1

for i, j in enumerate(squares):
    squarefound = False
    for k, l in enumerate(stationarysquares):
        totalx1 = stationarysquares[k, 2] - squares[i, 2]
        totaly1 = stationarysquares[k, 3] - squares[i, 3]
        totalx2 = stationarysquares[k, 4] - squares[i, 4]
        totaly2 = stationarysquares[k, 5] - squares[i, 5]
        totalsum = abs(totalx1 + totaly1 + totalx2 + totaly2)
        if totalsum < 30:
            stationarysquares[k, 1] += 3
            squarefound = True
        if stationarysquares[k, 1] > 30:

            squares = np.vstack((stationarysquares[k], squares))
    stationarysquares = stationarysquares[stationarysquares.min(axis=1) >= -20, :]
    if squarefound == False: stationarysquares = np.vstack((squares[i], stationarysquares))
    if i == 5600: break
    if squares[i+1, 0] - squares[i,0] > 0: stationarysquares[:, 1] -= 1


print(stationarysquares)
