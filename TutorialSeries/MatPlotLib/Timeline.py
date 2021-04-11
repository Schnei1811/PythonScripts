import matplotlib.pyplot as plt
import pandas as pd

dates = ["Tue  2 Jun 16:55:51 CEST 2015",
"Wed  3 Jun 14:51:49 CEST 2015",
"Fri  5 Jun 10:31:59 CEST 2015",
"Sat  6 Jun 20:47:31 CEST 2015",
"Sun  7 Jun 13:58:23 CEST 2015",
"Mon  8 Jun 14:56:49 CEST 2015",
"Tue  9 Jun 23:39:11 CEST 2015",
"Sat 13 Jun 16:55:26 CEST 2015",
"Sun 14 Jun 15:52:34 CEST 2015",
"Sun 14 Jun 16:17:24 CEST 2015",
"Mon 15 Jun 13:23:18 CEST 2015"]

dates = ["Apr 1990",
         "May 1990",
         "June 1990",
         "1998",
         "2001",
         "2003",
         "2004",
         "2005",
         "2007",
         "2009",
         "2010",
         "2011",
         "2013",
         "2013",
         "2013",
         "2014",
         "2016",
         "2016",
         "2017",
         "2017"]

values = [1,2,4,3,3,2,1,2,3,3,1,1,1,1,1,1,1,1,1,1]

X = pd.to_datetime(dates)
fig, ax = plt.subplots(figsize=(6,1))
ax.scatter(X, [1]*len(X), c=values,
           marker='s', s=100)
fig.autofmt_xdate()

# everything after this is turning off stuff that's plotted by default

ax.yaxis.set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

ax.get_yaxis().set_ticklabels([])
day = pd.to_timedelta("1", unit='D')
plt.xlim(X[0] - day, X[-1] + day)
plt.show()