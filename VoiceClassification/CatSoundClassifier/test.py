import scipy.io.wavfile as wavfile
import numpy as np

def show_info(aname, a):
    print ("Array", aname)
    print ("shape:", a.shape)
    print ("dtype:", a.dtype)
    print ("min, max:", a.min(), a.max())


rate, data = wavfile.read('RawAudio/Aggressive0.wav')

show_info("Data", data)

# Take the sine of each element in `Data`.
# The np.sin function is "vectorized", so there is no need
# for a Python loop here.
sindata = np.sin(data)

show_info("sindata", sindata)

# Scale up the values to 16 bit integer range and round
# the value.
scaled = np.round(32767*sindata)

show_info("scaled", scaled)

# Cast `scaled` to an array with a 16 bit signed integer Data type.
newdata = scaled.astype(np.int16)
