import numpy as np

data = np.loadtxt('1chargingbehaviour101308UnsupervisedData50x50.txt', delimiter=',')
print(data.shape)
data = data[0:10000, :]
print(data.shape)

np.savetxt('toydata50x50.txt', data, fmt='%i', delimiter=',')