# libraries
import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25


#January 2015

#Day 1

#bars1 = [1, 22, 17, 10, 16, 11]
#bars2 = [0, 8, 6, 3, 1, 3]
#bars3 = [3, 0, 1, 1, 2, 1]

#avg_oct = [1, 5, 6, 3, 4, 6]

#Full Day 2
#bars1 = [23, 17, 10, 16, 11, 24, 19, 9, 15, 9, 8, 8, 8, 7, 13, 7, 20, 18, 12, 34, 28, 29, 27, 17, 8, 20, 23, 18]
#bars2 = [8, 6, 3, 1, 3, 10, 3, 0, 0, 0, 0, 1, 0, 0, 2, 5, 11, 7, 1, 4, 0, 11, 20, 6, 2, 7, 9, 8]
#bars3 = [0, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 3, 2, 0, 2, 1, 1, 1, 1, 0]

#avg_oct = [6, 7, 7, 7, 7, 9, 7, 7, 9, 8, 7, 7, 7, 8, 9, 8, 12, 12, 10, 11, 10, 10, 10, 9, 8, 9, 11, 9]

#bbars1 = [23, 17, 10, 16, 11]
#bars2 = [8, 6, 3, 1, 3]
#bars3 = [0, 1, 1, 2, 1]
#avg_oct = [6, 7, 7, 7, 7]

#cbars1 = [24, 19, 9, 15, 9, 8, 8, 8, 7, 13]
#bars2 = [10, 3, 0, 0, 0, 0, 1, 0, 0, 2]
#bars3 = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
#avg_oct = [9, 7, 7, 9, 8, 7, 7, 7, 8, 9]

#dbars1 = [7, 20, 18, 12]
#bars2 = [5, 11, 7, 1]
#bars3 = [0, 2, 1, 1]
#avg_oct = [8, 12, 12, 10]

#ebars1 = [34, 28, 29, 27, 17, 8, 20, 23, 18]
#bars2 = [4, 0, 11, 20, 6, 2, 7, 9, 8]
#bars3 = [3, 2, 0, 2, 1, 1, 1, 1, 0]
#avg_oct = [11, 10, 10, 10, 9, 8, 9, 11, 9]

#Day 3

#bars1 = [8, 20, 23, 18, 4]
#bars2 = [2, 7, 9, 6, 0]
#bars3 = [1, 1, 1, 0, 0]

#avg_oct = [4, 10, 4, 4, 1]





#December 2016 MasterSheet2016

#dive1
#bars1 = [7, 3, 3]
#bars2 = [0, 0, 0]
#bars3 = [1, 0, 0]
#avg_oct = [3, 2, 2]

#dive2
#bars1 = [3, 3, 2, 1, 1, 1, 5, 1, 2, 1, 9, 4, 5, 2, 1]
#bars2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#bars3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#avg_oct = [2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 1]

#dive3
#bars1 = [2, 8, 6, 10]
#bars2 = [0, 0, 0, 0]
#bars3 = [1, 0, 0, 1]
#avg_oct = [2, 3, 3, 3]

#dive4
#bars1 = [1, 1, 4, 3, 9, 2, 7]
#bars2 = [0, 0, 0, 0, 0, 0, 0]
#bars3 = [0, 0, 0, 0, 0, 0, 1]
#avg_oct = [1, 1, 2, 3, 4, 2, 2]




# December 2015

#Dive 1
# bars1 = [6, 15, 13, 3]
# bars2 = [0, 0, 0, 0]
# bars3 = [0, 0, 0, 0]
# avg_oct = [4, 7, 7, 3]

#Dive 2
# bars1 = [1, 1, 8, 9, 22, 4]
# bars2 = [0, 0, 0, 1, 0, 0]
# bars3 = [0, 1, 2, 1, 1, 0]
# avg_oct = [1, 3, 4, 3, 4, 3]

#Dive 3
# bars1 = [2, 9, 5, 1, 4, 1]
# bars2 = [0, 0, 0, 0, 0, 0]
# bars3 = [2, 1, 0, 1, 0, 0]
# avg_oct = [1, 7, 4, 1, 4, 1]

#Dive 4
bars1 = [6, 9, 2, 5, 2]
bars2 = [0, 2, 2, 0, 0]
bars3 = [0, 0, 0, 0, 0]
avg_oct = [3, 4, 3, 2, 2]


# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Excursions')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Intruders')
# plt.bar(r2, bars3, width=barWidth, edgecolor='white', label='Mating')
plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Mating')

for i, v in enumerate(avg_oct):
    plt.text(i, bars1[i] + .3, str(avg_oct[i]), fontweight='bold')

# Add xticks on the middle of the group bars
plt.title('Octopus Activity Summary')
plt.ylabel('Frequency')
plt.xlabel('Time (min)')
plt.xticks([r + barWidth for r in range(len(bars1))], [x for x in range(30, 870, 30)], rotation=90)

# Create legend & Show graphic
plt.legend()
plt.show()
