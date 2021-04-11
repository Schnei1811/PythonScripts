import numpy as np

data = np.loadtxt('Sort.txt')

#Data = np.array([2, 43, 2, 1, 5, 4, 0, 2, 1, 45, 56, 23, 24, 1342, 12, 341, 12, 5342, 24321, 342, 123, 431])

for i in range(1, len(data)):
    val = data[i]
    j = i - 1
    while (j >= 0) and (data[j] > val):
        data[j + 1] = data[j]
        j = j - 1
    data[j + 1] = val
print(data)