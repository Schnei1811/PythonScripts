import numpy as np
import pandas as pd

text = np.loadtxt('OctoCoordinatesVideoOne.txt', delimiter=',').astype(int)


for i, j in enumerate(text):
    if i % 50 == 0:
        if 'csv' not in locals(): csv = np.array([(j[0]), 1920, 1080, j[5], j[2], j[1], j[4], j[3]])
        else: csv = np.vstack((csv, np.array([(j[0]), 1920, 1080, j[5], j[2], j[1], j[4], j[3]])))

for i, j in enumerate(csv):
    j[0] = ''.join((str(j[0]), '.jpg'))


np.savetxt('OctopusTFRecords.csv', csv, delimiter=',', fmt='%i')