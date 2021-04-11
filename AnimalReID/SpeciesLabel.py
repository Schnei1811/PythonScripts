import cv2
import numpy as np
import os
import time
import tqdm
import pandas as pd

def write_to_file(specieslist):
    while True:
        if input('\nWould you like to add species expected in the video? (y/n): ') == 'y':
            species = input('Enter species name: ')
            ans = input('{}? (y/n)'.format(species))
            if species in specieslist: print('\n{} already on the species list\n'.format(species))
            else:
                if ans == 'y':
                    specieslist.append(species)
                    specieslist.sort()
                print('Current species:\n')
            for species in specieslist: print(species)
        else: break
    with open(SAVE_DIR + 'SpeciesData.txt', 'w') as f:
        for item in specieslist:
            f.write("{}\n".format(item))
    f.close()
    return specieslist

def mouse_xy(event, x, y, flags, param):
    global ix, iy, frame, maxnumframes, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = -1, -1
    if event == cv2.EVENT_LBUTTONUP:
        ix, iy = x, y
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(frame, (x, y), 100, (255, 0, 0), -1)
        ix, iy = x, y
    if event == cv2.EVENT_RBUTTONDBLCLK:
        k = 27


def record_coordinates(videoname):
    global ix, iy, frame, maxnumframes, k
    ix, iy = -1, -1

    capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
    maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    objectcoordinates = np.array([-1, -1, -1])

    for framenum in range(maxnumframes):
        ret, frame = capture.read()
        cv2.namedWindow('original')
        cv2.moveWindow('original', 40, 30)
        cv2.setMouseCallback('original', mouse_xy)
        cv2.imshow('original', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: break
        if framenum == 0:
            print('Place mouse on video. Start in 3 seconds')
            time.sleep(3)
        objectcoordinates = np.vstack((np.array([framenum, ix, iy]), objectcoordinates))
    return objectcoordinates


def capture_object(objectnumber, SOD):
    print('Videos in Directory: {}\n'.format(VIDEO_DIR))

    for species in os.listdir(VIDEO_DIR):
        print(os.listdir(VIDEO_DIR).index(species), species)

    # User input for video
    while True:
        ans = input('\nSelect video by name or number: ')
        if ans in os.listdir(VIDEO_DIR):
            videoname = ans[:-4]
            break
        elif 0 <= int(ans) < len(os.listdir(VIDEO_DIR)):
            videoname = os.listdir(VIDEO_DIR)[int(ans)][:-4]
            break
        else:
            print('Enter valid videoname')

    # Check record of videos and raise flag if previously analyzed
    if os.path.exists(SAVE_DIR + 'VideosExamined.txt'):
        with open(SAVE_DIR + 'VideosExamined.txt', 'r') as f:
            videolist = f.read().splitlines()
        if videoname in videolist:
            print('This video has been previously examined. Duplicate data can lead to overfitting.')
            if not input('Are you sure you would like to continue? (y/n): ') == 'y': quit()

    print('Video name: {}'.format(videoname))
    print(
        '\nInstructions:\nFollow the top left corner of the object of interest with your mouse until the object leaves '
        'frame or the video ends. The video will restart. Then follow the bottom right corner. \nIf the animal becomes '
        'hidden, click and hold the left mouse button to no longer capture to individual.\nTo exit video early press '
        'escape \nRepeat as necessary until all objects are labeled')

    # Keep a list of examined species. If file doesn't exist, create it and ask for user input
    if not os.path.exists(SAVE_DIR + 'SpeciesData.txt'):
        specieslist = []
        write_to_file(specieslist)
    else:
        with open(SAVE_DIR + 'SpeciesData.txt', 'r') as f:
            specieslist = f.read().splitlines()
        f.close()
        # If file is empty, ask for user input
        if not specieslist:
            write_to_file(specieslist)
        else:
            print('Current species:\n')
            for species in specieslist: print(species)
            specieslist = write_to_file(specieslist)


    #Object Coordinates
    OCx1y1 = record_coordinates(videoname)
    OCx2y2 = record_coordinates(videoname)


    if OCx1y1.shape[0] < maxnumframes + 1:
        OCx1y1 = np.vstack((np.ones((maxnumframes - OCx1y1.shape[0], 3)) * -1, OCx1y1)).astype(int)

    if OCx2y2.shape[0] < maxnumframes + 1:
        OCx2y2 = np.vstack((np.ones((maxnumframes - OCx2y2.shape[0], 3)) * -1, OCx2y2)).astype(int)

    #Object Data
    OD = np.hstack((OCx1y1, OCx2y2[:, 1:])).astype(int)
    OD = np.flipud(OD)

    #Play video with box for review
    if input('Would you like to review this object? (y/n)') == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for i in range(maxnumframes):
            ret, frame = capture.read()
            if not np.isin(-1, OD[i]): cv2.rectangle(frame, (OD[i][1], OD[i][2]), (OD[i][3], OD[i][4]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break
            if i == 0: time.sleep(2)

    print('Species List: ')
    for species in specieslist: print(specieslist.index(species), species)

    #Add column representing species name
    while True:
        ans = input('Which species is this? (0-{}): '.format(len(specieslist)-1))
        if ans.isdigit() == True:
            if int(ans) in range(0, len(specieslist)):
                if not os.path.exists(SAVE_DIR + '{}/{}/'.format(specieslist[int(ans)], videoname)):
                    os.makedirs(SAVE_DIR + '{}/{}/'.format(specieslist[int(ans)], videoname))
                OD = np.hstack((OD, np.ones((OD.shape[0], 1))*int(ans))).astype(int)
                break
        else: print('Enter Valid Numeric Value')

    #If satisfied, add column representing object number (used for animal re-ID)
    if input('Are you satisfied with this object? (y/n)') == 'y':
        OD = np.hstack((OD, np.ones((OD.shape[0], 1)) * objectnumber)).astype(int)
        if objectnumber == 0: SOD = OD
        else: SOD = np.vstack((SOD, OD))
    else:
        capture_object(objectnumber, SOD)

    if input('Would you like to label more objects? (y/n)') == 'y':
        objectnumber += 1
        capture_object(objectnumber, SOD)

    #Review all bounding boxes
    if input('Review? (y/n)') == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for framenum in range(maxnumframes):
            ret, frame = capture.read()
            for i, j in enumerate(SOD):
                if SOD[i, 0] == framenum and not np.isin(-1, SOD[i][:]):
                    cv2.rectangle(frame, (SOD[i][1], SOD[i][2]), (SOD[i][3], SOD[i][4]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break

        if input('Would you like to label more objects? (y/n)') == 'y':
            objectnumber += 1
            capture_object(objectnumber, SOD)

    #Create output file array
    outputfilearray = np.array(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], dtype=str)

    if input('Save? (y/n)') == 'y':
        framesave = int(input('Save every how many frames? (0-20. 5 recommended. 1 is every frame): '))
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for framenum in tqdm.trange(0, maxnumframes):
            ret, frame = capture.read()
            if framenum % framesave == 0:
                if np.isin(-1, SOD[framenum]): pass
                else:
                    cv2.imwrite(SAVE_DIR + 'Frames/{}.{}.jpg'.format(videoname, framenum), frame)
                    for i, j in enumerate(SOD):
                        if framenum == SOD[i][0]:
                            if np.isin(-1, SOD[i]): pass
                            else:
                                img = frame[SOD[i][2]:SOD[i][4], SOD[i][1]:SOD[i][3]]
                                cv2.imwrite(SAVE_DIR + '{}/{}/{}.{}.{}.jpg'.format(specieslist[SOD[i][5]], videoname,
                                                                                specieslist[SOD[i][5]],
                                                                                SOD[i][6], framenum), img)
                                outputfilearray = np.vstack((outputfilearray,
                                                            ['{}.{}.jpg'.format(videoname, SOD[i][0]),
                                                             int(capture.get(3)), int(capture.get(4)),
                                                             specieslist[SOD[i][5]], SOD[i][1], SOD[i][2],
                                                             SOD[i][3], SOD[i][4]]))

        outputfilearray = outputfilearray.astype(str)

        if not os.path.exists(SAVE_DIR + 'ObjectDetectorTrainingFile.csv'):
            np.savetxt(SAVE_DIR + 'ObjectDetectorTrainingFile.csv', outputfilearray, delimiter=',', fmt='%s')
        else:
            np.savetxt(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv', outputfilearray[1:], delimiter=',', fmt='%s')
            df2 = pd.read_csv(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv')
            os.remove(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv')
            df2.to_csv(SAVE_DIR + 'ObjectDetectorTrainingFile.csv', mode='a', index=False)

        if not os.path.exists(SAVE_DIR + 'VideosExamined.txt'):
            with open(SAVE_DIR + 'VideosExamined.txt', 'w') as f:
                f.write("{}\n".format(videoname))
        else:
            with open(SAVE_DIR + 'VideosExamined.txt', 'r') as f:
                videolist = f.read().splitlines()
            f.close()
            videolist.append(videoname)
            with open(SAVE_DIR + 'VideosExamined.txt', 'w') as f:
                for item in sorted(videolist):
                    f.write("{}\n".format(item))

        if input('Saved!\n\nWould you like to annotate another video? (y/n)') == 'y':
            objectnumber = 0
            capture_object(objectnumber, SOD)
        else: quit()
    else:
        if input('Saved!\n\nWould you like to annotate another video? (y/n)') == 'y':
            objectnumber = 0
            capture_object(objectnumber, SOD)



#------------------------------------------------#
# Change Directory
# Folder containing raw videos
VIDEO_DIR = 'RawVideos/'

#Folder to save data
SAVE_DIR = 'Peru/'

#------------------------------------------------#
import cv2
import numpy as np
import os
import time
import tqdm
import pandas as pd

def write_to_file(specieslist):
    while True:
        if input('\nWould you like to add species expected in the video? (y/n): ') == 'y':
            species = input('Enter species name: ')
            ans = input('{}? (y/n)'.format(species))
            if species in specieslist: print('\n{} already on the species list\n'.format(species))
            else:
                if ans == 'y':
                    specieslist.append(species)
                    specieslist.sort()
                print('Current species:\n')
            for species in specieslist: print(species)
        else: break
    with open(SAVE_DIR + 'SpeciesData.txt', 'w') as f:
        for item in specieslist:
            f.write("{}\n".format(item))
    f.close()
    return specieslist

def mouse_xy(event, x, y, flags, param):
    global ix, iy, frame, maxnumframes, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = -1, -1
    if event == cv2.EVENT_LBUTTONUP:
        ix, iy = x, y
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(frame, (x, y), 100, (255, 0, 0), -1)
        ix, iy = x, y
    if event == cv2.EVENT_RBUTTONDBLCLK:
        k = 27


def record_coordinates(videoname):
    global ix, iy, frame, maxnumframes, k
    ix, iy = -1, -1

    capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
    maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    objectcoordinates = np.array([-1, -1, -1])

    for framenum in range(maxnumframes):
        ret, frame = capture.read()
        cv2.namedWindow('original')
        cv2.moveWindow('original', 40, 30)
        cv2.setMouseCallback('original', mouse_xy)
        cv2.imshow('original', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: break
        if framenum == 0:
            print('Place mouse on video. Start in 3 seconds')
            time.sleep(3)
        objectcoordinates = np.vstack((np.array([framenum, ix, iy]), objectcoordinates))
    cv2.destroyAllWindows()
    return objectcoordinates


def capture_object(objectnumber, SOD):
    print('Videos in Directory: {}\n'.format(VIDEO_DIR))

    for species in os.listdir(VIDEO_DIR):
        print(os.listdir(VIDEO_DIR).index(species), species)

    # User input for video
    while True:
        ans = input('\nSelect video by name or number: ')
        if ans in os.listdir(VIDEO_DIR):
            videoname = ans[:-4]
            break
        elif 0 <= int(ans) < len(os.listdir(VIDEO_DIR)):
            videoname = os.listdir(VIDEO_DIR)[int(ans)][:-4]
            break
        else:
            print('Enter valid videoname')

    # Check record of videos and raise flag if previously analyzed
    if os.path.exists(SAVE_DIR + 'VideosExamined.txt'):
        with open(SAVE_DIR + 'VideosExamined.txt', 'r') as f:
            videolist = f.read().splitlines()
        if videoname in videolist:
            print('This video has been previously examined. Duplicate data can lead to overfitting.')
            if not input('Are you sure you would like to continue? (y/n): ') == 'y': capture_object(objectnumber, SOD)

    print('Video name: {}'.format(videoname))
    print(
        '\nInstructions:\nFollow the top left corner of the object of interest with your mouse until the object leaves '
        'frame or the video ends. The video will restart. Then follow the bottom right corner. \nIf the animal becomes '
        'hidden, click and hold the left mouse button to no longer capture to individual.\nTo exit video early press '
        'escape \nRepeat as necessary until all objects are labeled')

    # Keep a list of examined species. If file doesn't exist, create it and ask for user input
    if not os.path.exists(SAVE_DIR + 'SpeciesData.txt'):
        specieslist = []
        write_to_file(specieslist)
    else:
        with open(SAVE_DIR + 'SpeciesData.txt', 'r') as f:
            specieslist = f.read().splitlines()
        f.close()
        # If file is empty, ask for user input
        if not specieslist:
            write_to_file(specieslist)
        else:
            print('\nCurrent species:\n')
            for species in specieslist: print(species)
            specieslist = write_to_file(specieslist)


    #Object Coordinates
    OCx1y1 = record_coordinates(videoname)
    OCx2y2 = record_coordinates(videoname)


    if OCx1y1.shape[0] < maxnumframes + 1:
        OCx1y1 = np.vstack((np.ones((maxnumframes - OCx1y1.shape[0], 3)) * -1, OCx1y1)).astype(int)

    if OCx2y2.shape[0] < maxnumframes + 1:
        OCx2y2 = np.vstack((np.ones((maxnumframes - OCx2y2.shape[0], 3)) * -1, OCx2y2)).astype(int)

    #Object Data
    OD = np.hstack((OCx1y1, OCx2y2[:, 1:])).astype(int)
    OD = np.flipud(OD)

    #Play video with box for review
    if input('\nWould you like to review this object? (y/n): ') == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for i in range(maxnumframes):
            ret, frame = capture.read()
            if not np.isin(-1, OD[i]): cv2.rectangle(frame, (OD[i][1], OD[i][2]), (OD[i][3], OD[i][4]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break
            if i == 0: time.sleep(2)
        cv2.destroyAllWindows()

    print('\nSpecies List: ')
    for species in specieslist: print(specieslist.index(species), species)

    #Add column representing species name
    while True:
        ans = input('\nWhich species is this? (0-{}): '.format(len(specieslist)-1))
        if ans.isdigit() == True:
            if int(ans) in range(0, len(specieslist)):
                if not os.path.exists(SAVE_DIR + '{}/{}/'.format(specieslist[int(ans)], videoname)):
                    os.makedirs(SAVE_DIR + '{}/{}/'.format(specieslist[int(ans)], videoname))
                OD = np.hstack((OD, np.ones((OD.shape[0], 1))*int(ans))).astype(int)
                break
        else: print('Enter Valid Numeric Value')

    #If satisfied, add column representing object number (used for animal re-ID)
    if input('\nAre you satisfied with this object? (y/n): ') == 'y':
        OD = np.hstack((OD, np.ones((OD.shape[0], 1)) * objectnumber)).astype(int)
        if objectnumber == 0: SOD = OD
        else: SOD = np.vstack((SOD, OD))
    else:
        capture_object(objectnumber, SOD)

    if input('\nWould you like to label more objects? (y/n): ') == 'y':
        objectnumber += 1
        capture_object(objectnumber, SOD)

    #Review all bounding boxes
    if input('\nReview? (y/n): ' ) == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for framenum in range(maxnumframes):
            ret, frame = capture.read()
            for i, j in enumerate(SOD):
                if SOD[i, 0] == framenum and not np.isin(-1, SOD[i][:]):
                    cv2.rectangle(frame, (SOD[i][1], SOD[i][2]), (SOD[i][3], SOD[i][4]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break
        cv2.destroyAllWindows()

        if input('\nWould you like to label more objects? (y/n): ') == 'y':
            objectnumber += 1
            capture_object(objectnumber, SOD)

    #Create output file array
    outputfilearray = np.array(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], dtype=str)

    if input('\nSave? (y/n): ') == 'y':
        framesave = int(input('\nSave every how many frames? (0-20. 5 recommended. 1 is every frame): '))
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for framenum in tqdm.trange(0, maxnumframes):
            ret, frame = capture.read()
            if framenum % framesave == 0:
                if np.isin(-1, SOD[framenum]): pass
                else:
                    cv2.imwrite(SAVE_DIR + 'Frames/{}.{}.jpg'.format(videoname, framenum), frame)
                    for i, j in enumerate(SOD):
                        if framenum == SOD[i][0]:
                            if np.isin(-1, SOD[i]): pass
                            else:
                                img = frame[SOD[i][2]:SOD[i][4], SOD[i][1]:SOD[i][3]]
                                cv2.imwrite(SAVE_DIR + '{}/{}/{}.{}.{}.jpg'.format(specieslist[SOD[i][5]], videoname,
                                                                                specieslist[SOD[i][5]],
                                                                                SOD[i][6], framenum), img)
                                outputfilearray = np.vstack((outputfilearray,
                                                            ['{}.{}.jpg'.format(videoname, SOD[i][0]),
                                                             int(capture.get(3)), int(capture.get(4)),
                                                             specieslist[SOD[i][5]], SOD[i][1], SOD[i][2],
                                                             SOD[i][3], SOD[i][4]]))

        outputfilearray = outputfilearray.astype(str)

        if not os.path.exists(SAVE_DIR + 'ObjectDetectorTrainingFile.csv'):
            np.savetxt(SAVE_DIR + 'ObjectDetectorTrainingFile.csv', outputfilearray, delimiter=',', fmt='%s')
        else:
            np.savetxt(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv', outputfilearray[1:], delimiter=',', fmt='%s')
            df2 = pd.read_csv(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv')
            os.remove(SAVE_DIR + 'ObjectDetectorTrainingFile2.csv')
            df2.to_csv(SAVE_DIR + 'ObjectDetectorTrainingFile.csv', mode='a', index=False)

        if not os.path.exists(SAVE_DIR + 'VideosExamined.txt'):
            with open(SAVE_DIR + 'VideosExamined.txt', 'w') as f:
                f.write("{}\n".format(videoname))
        else:
            with open(SAVE_DIR + 'VideosExamined.txt', 'r') as f:
                videolist = f.read().splitlines()
            f.close()
            videolist.append(videoname)
            with open(SAVE_DIR + 'VideosExamined.txt', 'w') as f:
                for item in sorted(videolist):
                    f.write("{}\n".format(item))

        if input('Saved!\n\nWould you like to annotate another video? (y/n): ') == 'y':
            objectnumber = 0
            capture_object(objectnumber, SOD)
        else: quit()
    else:
        if input('Saved!\n\nWould you like to annotate another video? (y/n): ') == 'y':
            objectnumber = 0
            capture_object(objectnumber, SOD)
        else: quit()



#------------------------------------------------#
# Change Directory
# Folder containing raw videos
VIDEO_DIR = 'RawVideos/'

#Folder to save data
SAVE_DIR = 'Peru/'

#------------------------------------------------#


if not os.path.exists(SAVE_DIR + '/Frames'):
    os.makedirs(SAVE_DIR + '/Frames')

#initialze global variables

#Saved Object Data. Final Data Array
SOD = np.array([])
#object number used for keeping track of multiple individuals for animal re-ID
objectnumber = 0


capture_object(objectnumber, SOD)

if not os.path.exists(SAVE_DIR + '/Frames'):
    os.makedirs(SAVE_DIR + '/Frames')

#initialze global variables

#Saved Object Data. Final Data Array
SOD = np.array([])
#object number used for keeping track of multiple individuals for animal re-ID
objectnumber = 0


capture_object(objectnumber, SOD)