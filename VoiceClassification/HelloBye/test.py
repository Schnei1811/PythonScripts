import numpy as np
import scipy.io.wavfile as wavfile

np.set_printoptions(threshold=np.nan)

path = 'RawAudio/Are0.wav'

rate, data = wavfile.read(path)

i = 0
audiostart = 0
audioend = 0
audiostarttoggle = False

for sound in data:
    if -500 < sound < 500: pass
    else:
        if audiostarttoggle == False:
            audiostarttoggle = True
            audiostart = i
        audioend = i
    i += 1


print('len ', len(data))
print(audiostart)
print(audioend)
#print(Data[audiostart: audioend])

originaldata = data[audiostart: audioend]
data = originaldata

i = 0
while len(data) < 5:
    noise = np.random.normal(0, 3, len(originaldata)).astype(int)
    noisydata = originaldata + noise
    data = np.vstack((data, noisydata))

print(data)
print(data.shape)


    # if Data[i-4] == 0 and Data[i-3] == 0 and Data[i-2] == 0 and Data[i-1] == 0 and Data[i] == 0 and audiotoggle == True:
    #     audioend = i
    #     audiotoggle = False
    #     if len(Data[audiostart:audioend]) > len(outputdata): outputdata = Data[audiostart:audioend]
    # if Data[i-1] == 0 and Data[i] == 0 and audiotoggle == False: audiostartcheck = i
    # if i - audiostartcheck > 500:
    #     audiostart = audiostartcheck
    #     audiotoggle = True
    # i += 1


