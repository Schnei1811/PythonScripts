import numpy as np
import os
import scipy.io.wavfile as wavfile
from tqdm import tqdm
np.set_printoptions(threshold=np.nan)

#wavfile.write('testwrite{}.wav'.format(soundfile), int(rate/SamplingRate), wavdata)

def Process_Raw_Audio():
    maxlength = 0
    for soundfile in tqdm(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, soundfile)
        refinedpath = os.path.join(REFINED_DIR, soundfile)
        i, audiostart, audioend, audiostarttoggle = 0, 0, 0, False
        rate, wavdata = wavfile.read(path)
        wavdata = wavdata[:, 0]             #Consider only one channel
        wavdata = wavdata[0::SamplingRate]  #Reduce Data size
        for sound in wavdata:
            if -500 < sound < 500: pass
            else:
                if audiostarttoggle == False:
                    audiostarttoggle = True
                    audiostart = i
                audioend = i
            i += 1
        outputdata = wavdata[audiostart:audioend]
        print(soundfile, len(outputdata))
        if len(outputdata) > maxlength:
            maxlength = len(outputdata)
            print('New Max Length: ', soundfile, maxlength)
        np.savetxt(refinedpath, outputdata, fmt='%i', delimiter=',')
        i = 0
        while i < 10:
            noise = np.random.normal(0, GaussianNoise, len(outputdata)).astype(int)
            noisydata = outputdata + noise
            np.savetxt(refinedpath + str(i), noisydata, fmt='%i', delimiter=',')
            i += 1
    np.save('model/MLPModelInfo', [maxlength, SamplingRate, HiddenLayerSize])
    return maxlength

RAW_DIR = "RawAudio/"
REFINED_DIR = "RefinedData/"
SamplingRate = 4
GaussianNoise = 20
HiddenLayerSize = 1000

i = 0
fulldata = []

maxlength = Process_Raw_Audio()

print('Max Length: ', maxlength)

for soundfile in tqdm(os.listdir(REFINED_DIR)):
    path = os.path.join(REFINED_DIR, soundfile)
    data = np.loadtxt(path)
    data = np.lib.pad(data, (0, maxlength - len(data)), 'constant', constant_values=(0, 0))
    if "Aggressive" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 1))
    if "Demanding" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 2))
    if "Hiss" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 3))
    if "Kitten" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 4))
    if "Purring" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 5))

    if i == 0:
        fulldata = data
        i += 1
    else: fulldata = np.vstack((fulldata, data))

np.savetxt('FormattedData/fulldata.csv', fulldata, fmt='%i', delimiter=',')
np.save('FormattedData/fulldata', fulldata)

print('Data Shape ', fulldata.shape)