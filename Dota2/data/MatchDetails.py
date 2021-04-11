import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import sys
import pickle
from sklearn import preprocessing, cross_validation,svm, neighbors,tree
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import time

df = pd.read_csv('e45g45.csv',delimiter=',')

df = df.as_matrix()

print(df)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample Data
centers = [[1, 1], [-1, -1], [1, -1]]

###############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(df, quantile=0.26, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(df)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

###############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(df[my_members, 0], df[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Match Detail Cluster Analysis')
plt.xlabel('XPM')
plt.ylabel('GPM')
plt.show()


import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

centers = [[1,1,1],[5,5,5],[3,10,10]]

X=df

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


#heroes = ['Abaddon','Alchemist','AncientApparition','AntiMage','ArcWarden','Axe','Bane','BatRider','Beastmaster','Bloodseeker','BountyHunter',\
#'BrewMaster','Bristleback','Broodmother','CentarWarrunner','ChaosKnight','Chen','Clinkz','Clockwerk','CrystalMaiden','DarkSeer','Dazzle','DeathProphet',\
#'Disruptor','Doom','DragonKnight','DrowRanger','EarthSpirit','EarthShaker','ElderTitan','EmberSpirit','Enchantress','Enigma','FacelessVoid','Gyro',\
#'Huskar','Invoker','Io','Jakiro','Juggernaut','KeeperoftheLight','Kunkaa','LegionCommander','Leshrac','Lich','Lifestealer','Lina','Lion','LoneDruid',\
#'Luna','Lycan','Magnus','Medusa','Meepo','Mirana','Morphling','NagaSiren','NaturesProphet','Necrophos','NightStalker','NyxAssassin','OrgeMagi',\
#'OmniKnight','Oracle','OutworldDevourer','PhantomAssassin','PhantomLancer','Pheonix','Puck','Pudge','Pugna','QueenofPain','Razor','Riki','Rubick',\
#'SandKing','ShadowDemon','ShadowFiend','ShadowShaman','Silencer','SkywrathMage','Slardar','Slark','Sniper','Spectre','SpiritBreaker','StormSpirit','Sven',\
#'Technies','TemplarAssassin','TerrorBlade','Tidehunter','Timbersaw','Tinker','Tiny','Treant','TrollWarlord','Tusk','Undying','Ursa','VengefulSpirit',\
#'Venomancer','Viper','Visage','Warlock','Weaver','Windranger','WinterWyvern','WitchDoctor','WraithKing']

#x = [0,5,10,15,20,25,30,35,40,45]

#for i in range(0,110):
#    plt.plot(x, df[i,12:22], label = heroes[i])
#plt.xlabel('Time')
#plt.ylabel('GPM')
#plt.title('Match Description')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

#for i in range(0,110):
#    plt.plot(x, df[i,1:11], label = heroes[i])
#plt.xlabel('Time')
#plt.ylabel('XPM')
#plt.title('Match Description')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()