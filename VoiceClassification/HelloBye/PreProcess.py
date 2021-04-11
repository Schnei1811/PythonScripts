import numpy as np
import os
import scipy.io.wavfile as wavfile
from tqdm import tqdm

def Process_Raw_Audio():
    maxlength = 0
    for soundfile in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, soundfile)
        refinedpath = os.path.join(REFINED_DIR, soundfile)
        rate, wavdata = wavfile.read(path)
        i, audiostart, audioend, audiostarttoggle = 0, 0, 0, False
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
            print(maxlength)
        np.savetxt(refinedpath, outputdata, fmt='%i', delimiter=',')

        i = 0
        while i < 20:
            noise = np.random.normal(0, 3, len(outputdata)).astype(int)
            noisydata = outputdata + noise
            np.savetxt(refinedpath + str(i), noisydata, fmt='%i', delimiter=',')
            i += 1
    np.save('model/maxlength', [maxlength])
    return maxlength

RAW_DIR = "RawAudio/"
REFINED_DIR = "RefinedData/"
i = 0
fulldata = []

maxlength = Process_Raw_Audio()

for soundfile in tqdm(os.listdir(REFINED_DIR)):
    path = os.path.join(REFINED_DIR, soundfile)
    data = np.loadtxt(path)
    data = np.lib.pad(data, (0, maxlength - len(data)), 'constant', constant_values=(0, 0))
    if "Hello" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 1))
    if "Bye" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 2))
    if "How" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 3))
    if "Are" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 4))
    if "You" in soundfile: data = np.lib.pad(data, (0, 1), 'constant', constant_values=(0, 5))
    if i == 0:
        fulldata = data
        i += 1
    else: fulldata = np.vstack((fulldata, data))

np.savetxt('FormattedData/fulldata.csv', fulldata, fmt='%i', delimiter=',')
np.save('FormattedData/fulldata', fulldata)

print('Data Shape ', fulldata.shape)