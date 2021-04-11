import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = {'first_name': ['SimpleNN', 'KNearestNeighbours', 'RBFSVM', 'RandomForest'],
            'TruePositives': [2098, 1915, 2361, 1714],
            'FalsePositives': [950, 1103, 1220, 1008],
            'FalseNegatives': [620, 803, 357, 1008],
            'TrueNegatives': [956, 803, 686, 898]}
df = pd.DataFrame(raw_data, columns=['TruePositives', 'FalsePositives', 'FalseNegatives', 'TrueNegatives'])

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,5))
bar_width = 0.75
# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['TruePositives']))]
# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]
# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score Data
        df['TruePositives'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='True Positives',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F4561D')
# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the mid_score Data
        df['FalsePositives'],
        # set the width
        width=bar_width,
        # with pre_score on the bottom
        bottom=df['TruePositives'],
        # with the label mid score
        label='FalsePositives',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F1911E')
# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the post_score Data
        df['FalseNegatives'],
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[i+j for i,j in zip(df['TruePositives'],df['FalsePositives'])],
        # with the label post score
        label='False Negatives',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F1BD1A')
# set the x ticks with names
plt.xticks(tick_pos, df['first_name'])
# Set the label and legends
ax1.set_ylabel("Total Score")
ax1.set_xlabel("Test Subject")
plt.legend(loc='upper left')
# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

