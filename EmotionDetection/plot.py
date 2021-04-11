from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
import ipdb

df = pd.read_csv('tweet_emotions.csv', delimiter=',')
df.head()


def basic_eda(df, row_limit=5, list_elements_limit=10):
    ### rows and columns
    print('Info : There are {} columns in the dataset'.format(df.shape[1]))
    print('Info : There are {} rows in the dataset'.format(df.shape[0]))

    print("==================================================")

    ## data types
    print("\nData type information of different columns")
    dtypes_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={0: 'dtype', 'index': 'column_name'})
    cat_df = dtypes_df[dtypes_df['dtype'] == 'object']
    num_df = dtypes_df[dtypes_df['dtype'] != 'object']
    print('Info : There are {} categorical columns'.format(len(cat_df)))
    print('Info : There are {} numerical columns'.format(len(dtypes_df) - len(cat_df)))

    if list_elements_limit >= len(cat_df):
        print("Categorical columns : ", list(cat_df['column_name']))
    else:
        print("Categorical columns : ", list(cat_df['column_name'])[:list_elements_limit])

    if list_elements_limit >= len(num_df):
        print("Numerical columns : ", list(num_df['column_name']))
    else:
        print("Numerical columns : ", list(num_df['column_name'])[:list_elements_limit])

    # dtypes_df['dtype'].value_counts().plot.bar()
    # display(dtypes_df.head(row_limit))

    print("==================================================")
    print("\nDescription of numerical variables")

    #### Describibg numerical columns
    desc_df_num = df[list(num_df['column_name'])].describe().T.reset_index().rename(columns={'index': 'column_name'})
    # display(desc_df_num.head(row_limit))

    print("==================================================")
    print("\nDescription of categorical variables")

    desc_df_cat = df[list(cat_df['column_name'])].describe().T.reset_index().rename(columns={'index': 'column_name'})
    # display(desc_df_cat.head(row_limit))

    return


basic_eda(df)

# Quickly check for mising values
total = df.isnull().sum()
print(total)



col = 'sentiment'
df_value_counts = df[col].value_counts()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
percents = list(df[col].value_counts(normalize=True))
labels = list(df_value_counts.index.values)
sizes = list(df_value_counts)
#ax.pie(sizes, explode=explode, colors=bo, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
ax2.pie(sizes, explode=percents, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.9, labeldistance=1.3)
ax2.add_artist(plt.Circle((0, 0), 0.6, fc='white'))
sns.countplot(y=col, data=df, ax=ax1)
ax1.set_title("Count of each emotion")
ax2.set_title("Percentage of each emotion")
plt.show()




df['sentiment'] = df['sentiment'].apply(lambda x : x if x in ['happiness', 'sadness', 'worry', 'neutral', 'love'] else "other")

df_value_counts = df[col].value_counts()

col = 'sentiment'
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
percents = list(df[col].value_counts(normalize=True))
labels = list(df_value_counts.index.values)
sizes = list(df_value_counts)
#ax.pie(sizes, explode=explode, colors=bo, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
ax2.pie(sizes,  explode=percents, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.9)
ax2.add_artist(plt.Circle((0,0),0.6,fc='white'))
sns.countplot(y =col, data = df, ax=ax1)
ax1.set_title("Count of each emotion")
ax2.set_title("Percentage of each emotion")
plt.show()

df['char_length'] = df['content'].apply(lambda x : len(x))
df['token_length'] = df['content'].apply(lambda x : len(x.split(" ")))






# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
# sns.distplot(df['char_length'], ax=ax1)
# sns.distplot(df['token_length'], ax=ax2)
# ax1.set_title('Number of characters in the tweet')
# ax2.set_title('Number of token(words) in the tweet')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(16,8))
# for sentiment in df['sentiment'].value_counts().sort_values()[-5:].index.tolist():
#     #print(sentiment)
#     sns.kdeplot(df[df['sentiment']==sentiment]['char_length'],ax=ax, label=sentiment)
# ax.legend()
# ax.set_title("Distribution of character length sentiment-wise [Top 5 sentiments]")
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8,6))
# for sentiment in df['sentiment'].value_counts().sort_values()[-5:].index.tolist():
#     #print(sentiment)
#     sns.kdeplot(df[df['sentiment']==sentiment]['token_length'],ax=ax, label=sentiment)
# ax.legend()
# ax.set_title("Distribution of token length sentiment-wise [Top 5 sentiments]")
# plt.show()




avg_df = df.groupby('sentiment').agg({'char_length':'mean', 'token_length':'mean'})

fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
ax1.bar(avg_df.index, avg_df['char_length'])
ax2.bar(avg_df.index, avg_df['token_length'], color='green')
ax1.set_title('Avg number of characters')
ax2.set_title('Avg number of token(words)')
ax1.set_xticklabels(avg_df.index, rotation = 45)
ax2.set_xticklabels(avg_df.index, rotation = 45)
plt.show()