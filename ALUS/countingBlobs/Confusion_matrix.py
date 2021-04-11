import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import confusion_matrix



conf_mtrx = np.load("confusion_matrix.npy")

conf_mtrx_norm = conf_mtrx / conf_mtrx.max(axis=1)

cmap = colors.ListedColormap(['#ffffff', '#e8ffff', '#d9f1ff', '#bfe6ff', '#8cd3ff',
                              '#59bfff', '#26abff', '#0da2ff', '#009dff', '#0055ff'])

bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
plot = plt.imshow(conf_mtrx_norm, interpolation='nearest', cmap=cmap, norm=norm)

# plt.xticks([i for i in range(len(labels))], labels)
plt.xticks(rotation=80)
# plt.yticks([i for i in range(len(labels))], labels)

for i in range(conf_mtrx.shape[0]):
    for k in range(conf_mtrx.shape[1]):
        plt.text(k, i, str(conf_mtrx[i][k]), horizontalalignment='center', verticalalignment='center')

plt.savefig("confusion_matrix.jpg", pad_inches=0.5)
np.save("confusion_matrix.npy", conf_mtrx)


import ipdb;ipdb.set_trace()
