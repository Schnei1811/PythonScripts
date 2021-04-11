from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

def parse_commit_dates(filename, dates=dict()):
    """ needs date file
     * one line per date
     * format: YYYY-mm-dd
     * produced with: git log --date=short --format=%cd > commitdates.txt
    """
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.strip()

            if line not in dates:
                dates[line] = 1
            else:
                dates[line] += 1

    return dates


def plot_commit_dates(dates):

    d = []
    v = []

    for key, value in sorted(dates.items(), key=lambda t: t[0]):
        print(key, value)

        date = datetime.strptime(key, '%Y-%m-%d')

        d.append(mdates.date2num(date))
        v.append(value)

    fig, ax = plt.subplots()
    ax.bar(d, v)
    ax.xaxis_date()

    # format the ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y'))
    ax.autoscale_view()

    fig.autofmt_xdate()
    plt.show()

if __name__ == '__main__':
    commit_dates = parse_commit_dates('commitdates-java.txt')
    commit_dates = parse_commit_dates('commitdates-webapp.txt', commit_dates)

    plot_commit_dates(commit_dates)

    plt.plot([1, 2, 3, 4])
    plt.ylabel("asd")
    plt.show()