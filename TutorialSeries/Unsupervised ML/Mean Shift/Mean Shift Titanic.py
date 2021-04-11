import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import pandas as pd
style.use('ggplot')

# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)             #copy dataframe
df.drop(['body', 'name'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
print(df.head())
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    # handling non-numerical Data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x += 1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)
#print(df.head())

# add/remove features just to see impact they have.
df.drop(['ticket', 'home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]        #iloc   references the row. first row under column cluster group, set value to what label i is. Same order as X

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']==float(i))]     #new temp df, original df where the cluster group is cluster group zero
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)

#print(original_df[(original_df['cluster_group']==0)])
#print(original_df[(original_df['cluster_group']==1)])
#print(original_df[(original_df['cluster_group']==2)])
#print(original_df[(original_df['cluster_group']==3)])

#print(original_df[(original_df['cluster_group']==0)].describe())
#print(original_df[(original_df['cluster_group']==1)].describe())
#print(original_df[(original_df['cluster_group']==2)].describe())
#print(original_df[(original_df['cluster_group']==3)].describe())

cluster_0 = original_df[(original_df['cluster_group']==0)]
cluster_0_fc = cluster_0[(cluster_0['pclass']==1)]
print(cluster_0_fc.describe())


