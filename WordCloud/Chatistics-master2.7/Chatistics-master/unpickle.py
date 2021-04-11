import pickle
import sys
import pandas as pd

import re


with open('data/messenger.pkl', 'rb') as pickle_file:
        u = pickle._Unpickler(pickle_file)
        u.encoding = 'latin1'
        p = u.load()
        print(p)

print(p)


p = re.sub('[^0-9a-zA-Z]+', '', p)

p.to_csv('Output.txt', sep=',', encoding='utf-8')

