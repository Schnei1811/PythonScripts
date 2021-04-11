import cv2
import numpy as np
import os
import time
import tqdm

# Track mouse movements
def mouse_xy(event, x, y, flags, param):
    global ix, iy
    #if left click, convert x, y values to -1
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = -1, -1

    #lift left click, return to x,y values
    if event == cv2.EVENT_LBUTTONUP:
        ix, iy = x, y

    #if mouse moves, record x,y
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(frame, (x, y), 100, (255, 0, 0), -1)
        ix, iy = x, y


def record_coordinates():
    global ix, iy
    ix, iy = -1, -1
    global frame
    global maxnumframes

    #load video
    capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))

    #get max number of frames
    maxnumframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 2

    #initialize object coordinates
    objectcoordinates = np.array([-1, -1, -1])

    #Play video and record coordinates
    for framenum in range(maxnumframes):
        ret, frame = capture.read()
        #Calls Mouse recording function
        cv2.setMouseCallback('original', mouse_xy)
        cv2.imshow('original', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: break
        if framenum == 0: time.sleep(2)
        objectcoordinates = np.vstack((np.array([framenum, iy, ix]), objectcoordinates))
    return objectcoordinates


def capture_object(objectnumber, savedobjectdata):

    #gather x1, y1 data
    OCx1y1 = record_coordinates()

    #gather x2, y2 data
    OCx2y2 = record_coordinates()

    #If video exited early and coordinate length uneven, add -1 values to make even
    if OCx1y1.shape[0] > OCx2y2.shape[0]:
        OCx2y2 = np.vstack((OCx2y2, np.ones((OCx1y1.shape[0] - OCx2y2.shape[0], 3)) * -1))
    else:
        OCx1y1 = np.vstack((OCx1y1, np.ones((OCx2y2.shape[0] - OCx1y1.shape[0], 3)) * -1))


    # Add a column relevant to object number. Used for for animal re-ID
    OCx2y2 = np.hstack((OCx2y2[:, 0:], np.ones((OCx2y2.shape[0], 1)) * objectnumber))

    #Join columns to create framenum, y1, x1, y2, x2, objectnumber and reverses order to make chronological
    objectdata = np.flip(np.hstack((OCx1y1, OCx2y2[:,0:])).astype(int))

    #Play video with bounding boxes
    ans = input('Would you like to review this object? (y/n)')
    if ans == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for i in range(maxnumframes):
            ret, frame = capture.read()
            cv2.rectangle(frame, (objectdata[i][2], objectdata[i][1]), (objectdata[i][4], objectdata[i][3]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break
            if i == 0: time.sleep(2)

    #Ask if satisfied with bounding boxes. If so, store object data in final variable savedobjectdata
    ans = input('Are you satisfied with this object? (y/n)')
    if ans == 'y':
        if objectnumber == 0: savedobjectdata = objectdata
        else: savedobjectdata = np.vstack((savedobjectdata, objectdata))
    else:
        capture_object(objectnumber, savedobjectdata)

    ans = input('Would you like to name this object? Numeric value if no. (y/n)')
    if ans == 'y':
        name = input('Name: ')
        namedict[objectnumber] = name
    else:
        namedict[objectnumber] = 'ObjectNumber' + str(objectnumber)

    ans = input('Would you like to label more objects? (y/n)')
    if ans == 'y':
        objectnumber += 1
        capture_object(objectnumber, savedobjectdata)

    ans = input('Review? (y/n)')
    if ans == 'y':
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        for framenum in range(maxnumframes):
            ret, frame = capture.read()
            for i, j in enumerate(savedobjectdata):
                if savedobjectdata[i, 0] == framenum:
                    cv2.rectangle(frame, (savedobjectdata[i][2], savedobjectdata[i][1]),
                                  (savedobjectdata[i][4], savedobjectdata[i][3]), 255, 2)
            cv2.imshow('original', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27: break

        print('Objects: \n')
        for key in namedict:
            print(namedict[key])
        print('\n')

        ans = input('Would you like to label more objects? (y/n)')
        if ans == 'y':
            objectnumber += 1
            capture_object(objectnumber, savedobjectdata)

    ans = input('Save? (y/n)')
    if ans == 'y':
        framesave = input('Save every how many frames? (Numeric value): ')
        capture = cv2.VideoCapture(VIDEO_DIR + '{}.mp4'.format(videoname))
        if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

        for framenum in tqdm.trange(0, maxnumframes):
            ret, frame = capture.read()
            if framenum % framesave == 0:
                for i, j in enumerate(savedobjectdata):
                    if framenum == savedobjectdata[i][0]:
                        if not os.path.exists(SAVE_DIR + '/{}'.format(namedict[savedobjectdata[i][5]])):
                            os.makedirs(SAVE_DIR + '/{}'.format(namedict[savedobjectdata[i][5]]))
                        if np.isin(-1, savedobjectdata[i]): pass
                        else:
                            print(savedobjectdata[i])
                            img = frame[savedobjectdata[i][1]:savedobjectdata[i][3],
                                  savedobjectdata[i][2]:savedobjectdata[i][4]]
                            cv2.imwrite(SAVE_DIR + '{}/{}.{}.jpg'.format(namedict[savedobjectdata[i][5]],
                                                                         namedict[savedobjectdata[i][5]], framenum),img)
        np.savetxt(SAVE_DIR + '/ObjectCoordinates{}.txt'.format(videoname), savedobjectdata, delimiter=',', fmt='%i')
        quit()
    else: quit()





# videoname = 'reddit'
#videoname = 'GO010141. Octopus03. t13 41'

# videoname = 'GO020141. Octopus04. t26 39'
# videoname = 'GO030141. Octopus03. t03 12[2]'
# videoname = 'GO030141. Octopus08. t15 30'
# videoname = 'GO040141. Octopus08. t15 05'
# videoname = 'GO010142. Octopus14. t22 39'
# videoname = 'GO020142. Octopus13. t22 36'
# videoname = 'GO030142. Octopus09. t32 23'
# videoname = 'GO030142. Octopus09. t32 23'
# videoname = 'GO040142. Octopus09. t30 27'
# videoname = 'GO050142. Octopus10. t01 36'
# videoname = 'GOPR0142. Octopus09. t08 35'
# videoname = 'GO021681. Octopus16. t02 47'
# videoname = 'GO031681. Octopus11. t14 22'
# videoname = 'GO010143. Octopus22. t18 07'

#test
# videoname = 'GO021681. Octopus13. t16 51'
# videoname = 'GO010143. Octopus22. t 14 44'
# videoname = 'GO010143. Octopus22. t28 18'
# videoname = 'reddit'


# videoname = '007Racoon'



VIDEO_DIR = 'D:PythonData/Peru/RawVideos/'

print('Videos in Video Directory: {}'.format(VIDEO_DIR))

for item in os.listdir('D:PythonData/Peru/RawVideos'):
    if item[-4:] == '.mp4':
        print(item)



print('Video name: {}'.format(videoname))
print('\nInstructions:\nFollow the top left corner of the object of interest for the full length of the video.'
      'The video will restart. Then follow the bottom right corner.\n'
      'Click and hold if object becomes hidden to ignore creating labels\n'
      'Repeat as necessary until all objects labeled\n')

#VIDEO_DIR = 'D:PythonData/Octopus/RawVideos/'
SAVE_DIR = 'D:PythonData/Peru/'.format(videoname)
objectnumber = 0
savedobjectdata = np.array([])
namedict = {}


capture_object(objectnumber, savedobjectdata)