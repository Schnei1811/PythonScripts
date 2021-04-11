def boxpixeldata(boxdata):
    print('\nGathering Pixel Data of Boxes')
    roiw, roih = uniformboxsize, uniformboxsize
    finalsquarearray = np.zeros([roiw * roih])

    # Resize Boxes by Expanding to Form Square. Check Bounds. Store in Array for Unsupervised Learning
    for framenum in tqdm.trange(0, maxnumframes):
        newframecounter = 0
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, j in enumerate(boxdata):
            if boxdata[i, 0] == framenum and (boxdata[i, 5] - boxdata[i, 3]) * (boxdata[i, 4] - boxdata[i, 2]) > IMG_SIZE_MINIMUM:
                squarearray, squarecounter = np.zeros([roiw * roih]), 0
                y1, y2, x1, x2 = image_expand(boxdata[i, 3], boxdata[i, 5], boxdata[i, 2], boxdata[i, 4])
                roi = gray[y1: y2, x1: x2]
                roi = cv2.resize(roi, (roiw, roih))
                for y in range(0, roih):
                    for x in range(0, roiw):
                        squarearray[squarecounter] = roi[y, x]  # works [height,width]
                        squarecounter += 1
                if newframecounter == 0:
                    intermediatesquarearray = squarearray
                    newframecounter += 1
                else: intermediatesquarearray = np.vstack((intermediatesquarearray, squarearray))
        if framenum == 0: finalsquarearray = intermediatesquarearray
        else: finalsquarearray = np.vstack((finalsquarearray, intermediatesquarearray))

        if framenum % savestate == 0:
            np.savetxt('Files/{}/temp{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize), finalsquarearray, fmt='%i', delimiter=',')
        if framenum == maxnumframes - 1:
            np.savetxt('Files/{}/{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize), finalsquarearray, fmt='%i', delimiter=',')
            os.remove('Files/{}/temp{}{}{}{}UnsupervisedData.txt'.format(videoname, videoname, threshold, blur, squaresize))
        framenum += 1
    return finalsquarearray

def KMeansfunc(squares):
    print('\nRunning KMeans on Identified Boxes')
    clf = KMeans(n_clusters=numclusters)
    clf.fit(squares)
    labels = clf.labels_
    np.savetxt('Files/{}/{}x{}Labels.txt'.format(videoname, uniformboxsize, uniformboxsize), labels, fmt='%i', delimiter=',')
    return labels

def saveimages(labels, imgdata):
    print('\nSaving Images')
    for dir in tqdm.trange(0, numclusters):
        if not os.path.exists('Files/{}/Clusters/{}'.format(videoname, dir)): os.makedirs('Files/{}/Clusters/{}'.format(videoname, dir))
        for i, j in enumerate(imgdata):
            if labels[i] == dir:
                roi = imgdata[i, :].reshape((uniformboxsize, uniformboxsize))
                cv2.imwrite('Files/{}/Clusters/{}/{}{}.jpg'.format(videoname, dir, i, videoname), roi)
