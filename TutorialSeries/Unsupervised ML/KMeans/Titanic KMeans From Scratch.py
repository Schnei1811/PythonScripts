import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4],])

#plt.scatter(X[:,0],X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}
        # using first two Data points. Could random but doesn't matter
        for i in range(self.k):
            self.centroids[i] = data[i]

        #Change to numbers for each iterations
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            # 0th index in list will be distance to 0th centroid and for 1s
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            # finding the mean of the features. Redefines the centroid
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            optimized = True

            # any of these centroids move more than tolerance, no we are not optimized
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    # usually wouldn't predict on what you've trained, but with centroids it doesn't matter
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

#unknowns = np.array([[1,3],
#                     [8,9],
#                     [0,3],
#                     [5,4],
#                     [6,4],])
#
#for unknown in unknowns:
#    classification = clf.predict(unknown)
#    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


plt.show()

