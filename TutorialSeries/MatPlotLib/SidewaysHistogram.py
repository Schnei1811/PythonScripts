# import matplotlib.pyplot as plt
# import numpy as np
#
#
# plt.rcdefaults()
# fig, ax = plt.subplots()
#
# animals = ('White-Tailed Deer', 'Mouflon', 'Red Deer', 'Agouti', 'Wild Boar', 'Wood Mouse', 'Roe Deer', 'Great Tinamou',
#            'White-Nosed Coati', 'Red Brocket Deer', 'Paca', 'Common Opossum', 'Collared Peccary', 'Ocelot',
#            'European Hare', 'Spiny Rat', 'Red-Tailed Squirrel', 'Red Fox')
#
# y_pos = (1091, 896, 802, 518, 487, 455, 362, 350, 325, 297, 285, 264, 263, 184, 176, 175, 143, 120)
#
# performance = 3 + 10 * np.random.rand(len(animals))
#
# ax.barh(y_pos, performance, align='center',
#         color='green', ecolor='black')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(animals)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Performance')
# ax.set_title('How fast do you want to go today?')
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
animals = ('White-Tailed Deer', 'Mouflon', 'Red Deer', 'Agouti', 'Wild Boar', 'Wood Mouse', 'Roe Deer', 'Great Tinamou',
           'White-Nosed Coati', 'Red Brocket Deer', 'Paca', 'Common Opossum', 'Collared Peccary', 'Ocelot',
           'European Hare', 'Spiny Rat', 'Red-Tailed Squirrel', 'Red Fox')
y_pos = np.arange(len(animals))
performance = (1091, 896, 802, 518, 487, 455, 362, 350, 325, 297, 285, 264, 263, 184, 176, 175, 143, 120)

ax.barh(y_pos, performance, align='center',
        color='green', ecolor='black', linewidth=0.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(animals)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Images')
ax.set_title('Reconyx Camera Trap Data Set Distribution of Species Captured Within Camera Trap Images')

plt.show()