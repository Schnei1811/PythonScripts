# Unlike K-means where you specify #2 with mean shift the machine figures out how many there ought to be and where they are
# Every feature set is a cluster center.
# Radius/bandwidth  -> Radius distance around each datapoint. Bandwidth  everything within the radius
# ie. 3 features within the bandwidth, determined by the radius
# Take the mean of all the datapoints within the bandwidth. Assign new mean value.
# Convergence -> When that cluster center stops moving it is optimized.
# Can have tiered sizes of radius where points within a first tier are weighted higher, then second, third or fourth
# Mainly for research / structuring Data

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

centers = [[1,1,1],[5,5,5],[3,10,10]]

X, _ = make_blobs(n_samples = 1000, centers = centers, cluster_std = 1.5)

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()