import numpy as np
data = np.array([2,43,2,1,5,4,0,2,1,45,56,23,24,1342,12,341,12,5342,24321,342,123,431])

i = 0
insertposition = 0

if data.shape == 1: print(data)
else:
    while i < len(data):
        j, insertposition = 0, 0
        for j in range(len(data)):
            if data[j] < data[i]: insertposition = j
        if insertposition > i:
            data = np.insert(data, insertposition + 1, data[i])
            data = np.delete(data, i)
        else: i += 1
print(data)

