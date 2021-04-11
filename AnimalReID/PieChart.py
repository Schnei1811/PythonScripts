import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(aspect="equal"))

data = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13']
ingredients = [201,3,5,9,7,13,17,15,22,7,4,6,2]

wedges, texts, autotexts = ax.pie(ingredients, autopct='%1.1f%%', startangle=90)

ax.legend(wedges, data,
          title="Octopus",
          loc="center left",
          bbox_to_anchor=(1, 0, 1, 1))

plt.setp(autotexts)

ax.set_title("Number of Resident Octopus Excursions")

plt.show()




# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13'
sizes = [201,3,5,9,7,13,17,15,22,7,4,6,2]
explode = [0,0,0,0,0,0,0,0,0,0,0,0,0]
#explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(14, 12))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Number of Resident Octopus Excursions")
plt.show()


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13'
sizes = [201,3,5,9,7,13,17,15,22,7,4,6,2]
explode = [0.1,0,0,0,0,0,0,0,0,0,0,0,0]
#explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(14, 12))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("Number of Resident Octopus Excursions")
plt.show()




# Pie chart
labels = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13']
sizes = [201,3,5,9,7,13,17,15,22,7,4,6,2]
explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
# colors

fig2, ax1 = plt.subplots(figsize=(14, 12))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
# draw circle
centre_circle = plt.Circle((0, 0), 0.80, fc='white')
ax1.set_title("Number of Resident Octopus Excursions")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(aspect="equal"))

recipe = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13']

data = [201,3,5,9,7,13,17,15,22,7,4,6,2]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=0)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Number of Resident Octopus Excursions")

plt.show()





