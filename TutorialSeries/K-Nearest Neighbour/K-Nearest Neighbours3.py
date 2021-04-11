#Euclidean Distance. Measures the closeness of points

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

#plot1 = [1,3]
#plot2 = [2,5]
#euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}

new_features = [5,7]

#[[plt.scatter(ii[0],ii[1], s=1xdif-100, color=i) for ii in dataset[i]] for i in dataset]

#for i in dataset:
#    for ii in dataset[i]:
#        [plt.scatter(ii[0],ii[1],s=1xdif-100, color=i)]


#plt.scatter(new_features[0],new_features[1])
#plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
            for features in data[group]:
#               euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))    One possibility
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([euclidean_distance, group])       #list of lists where first item is distance, then group. Sort list, get the first list element

    votes = [i[1] for i in sorted(distances)[:k]]
#    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

#   print(vote_result, confidence)

    return vote_result, confidence

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

accuracies = []

for i in range(25):
    df = pd.read_csv('KNNcancerdata')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist()        #Double check to convert any string to a float
    random.shuffle(full_data)

    test_size= 0.3
    train_set = {1:[],2:[]}
    test_set = {1:[],2:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct +=1
 #           else:
 #               print(confidence)
            total +=1
#    print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))

    # Confidence: Vote for result
