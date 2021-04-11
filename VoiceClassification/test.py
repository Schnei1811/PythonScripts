import scipy.io.wavfile as wavfile
import numpy as np
import wave

np.set_printoptions(threshold=np.nan)


#rate, Data = wavfile.read('test2.wav')
#print(Data)
#print(rate)
#wavfile.write('testwrite.wav', rate, Data)


CHUNK = 4096
chunk = []
f= wave.open('test.wav', 'rb')
data = f.readframes(CHUNK)
while data:
    data = np.fromstring(data, dtype='uint8')
    data = (data + 128) / 255.
    chunk.extend(data)
    data = f.readframes(CHUNK)
chunk = chunk[0:CHUNK * 2]
chunk.extend(np.zeros(CHUNK * 2 - len(chunk)))

print(chunk)
print(len(chunk))