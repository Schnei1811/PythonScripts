import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random

from sklearn.datasets.samples_generator import make_blobs       #can do both supervised and unsupervised

centers = random.randrange(2,5)

X,y = make_blobs(n_samples=25, centers=3, n_features=2)


#X = np.array([[1, 2],
#              [1.5, 1.8],
#              [5, 8],
#              [8, 8],
#              [1, 0.6],
#              [9, 11],
#              [8, 2],
#              [10, 2],
#              [9, 3], ])

#plt.scatter(X[:,0],X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]

#Every feature sets is a cluster center. Take the mean of all feature sets within that radius. Repeat until convergence
#Radius of 4 hard coded. Going to penalize points that are far away from the centroid
#Lots of radius steps. Closer the points are, higher the weight
class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data,axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]  # reverses list

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.00000000001       #when featureset compares distance to itself
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1     #if distance is greater then max distance(1xdif-100) steps. Weight is max(0)
                    to_add = (weights[weight_index]**2)*[featureset]    #making a very large list. Could be more efficient
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid)) #tuple and np arrays have different attributes. Later we will reference some things

            uniques = sorted(list(set(new_centroids))) #as we modify centroids it doesn't modify original ones

            to_pop = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:             #if identical
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:       #within one radius, remove
                        to_pop.append(ii)                   #can't modify list as you iterate through it
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)


    def predict(self,data):
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classificiation in clf.classifications:
    color = colors[classificiation]
    for featureset in clf.classifications[classificiation]:
        plt.scatter(featureset[0],featureset[1],marker='x',color=color,s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()

