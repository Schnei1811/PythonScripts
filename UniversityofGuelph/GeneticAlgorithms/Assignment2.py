import random
import statistics as stats
import numpy as np


randnumbers = []

for i in range(5000):
    randnumbers.append(random.randint(0, 100))

print(np.mean(randnumbers))
print(stats.stdev(randnumbers))

averagenumbers = []



for i in range(5000):
    randnumbers = []
    for k in range(100):
        randnumbers.append(random.randint(0, 100))
    averagenumbers.append(np.mean(randnumbers))

print(np.mean(averagenumbers))
print(stats.stdev(averagenumbers))

