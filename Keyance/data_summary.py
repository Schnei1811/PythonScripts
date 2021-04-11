import csv
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# csv_path = "G://Keyence//20-09-13-AMCAK.csv"

# csv_path = "G://Keyence//20-10-13-Stefan02BOLDdata.csv"

csv_path = "G://Keyence/ExcelSheets/bold-master-data.csv"

phylum_dct = {}
class_dct = {}
order_dct = {}
family_dct = {}
subfamily_dct = {}
tribe_dct = {}
genus_dct = {}
species_dct = {}


def dict_add(dct, item):
    if item == "":
        item = "Empty"
    if item not in dct:
        dct[item] = 1
    else:
        dct[item] += 1

def plot_bar(dct, title, empty=True):
    print(dct)
    if empty == False:
        if "Empty" in dct:
            del dct["Empty"]
    plt.clf()
    plt.bar(dct.keys(), dct.values())
    plt.xticks(rotation=90)
    plt.title(title)
    figure = plt.gcf()
    figure.set_size_inches(48, 28)
    plt.savefig(title, dpi=600)

def plot_pie_subplot(ax, title, dct, empty=True):
    if empty == False:
        if "Empty" in dct:
            del dct["Empty"]
    # ax.pie(dct.values(), labels=dct.keys(), autopct="%1.1f%%", shadow=True)
    ax.pie(dct.values(), labels=dct.keys(), shadow=True)
    ax.set_title(title)

def plot_bar_subplot(ax, title, dct, empty=True):
    if empty == False:
        if "Empty" in dct:
            del dct["Empty"]
    ax.bar(dct.keys(), dct.values())
    ax.tick_params(labelrotation=90)
    ax.set_title(title)

def create_table(title, dct):
    x = PrettyTable()
    num_images = sum(dct.values())
    x.field_names = [title, "Num. Images", "Percent"]
    for key in dct:
        x.add_row([key, dct[key], round((dct[key] / num_images) * 100, 2)])
    print(x)

with open(csv_path, newline="") as f:
    reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(reader):
        if i == 0:
            pass
        else:
            dict_add(phylum_dct, row[14])
            dict_add(class_dct, row[15])
            dict_add(order_dct, row[16])
            dict_add(family_dct, row[17])
            dict_add(subfamily_dct, row[18])
            dict_add(tribe_dct, row[19])
            dict_add(genus_dct, row[20])
            dict_add(species_dct, row[21])

phylum_dct = {k: v for k, v in sorted(phylum_dct.items(), key=lambda item: item[1], reverse=True)}
class_dct = {k: v for k, v in sorted(class_dct.items(), key=lambda item: item[1], reverse=True)}
order_dct = {k: v for k, v in sorted(order_dct.items(), key=lambda item: item[1], reverse=True)}
family_dct = {k: v for k, v in sorted(family_dct.items(), key=lambda item: item[1], reverse=True)}
subfamily_dct = {k: v for k, v in sorted(subfamily_dct.items(), key=lambda item: item[1], reverse=True)}
tribe_dct = {k: v for k, v in sorted(tribe_dct.items(), key=lambda item: item[1], reverse=True)}
genus_dct = {k: v for k, v in sorted(genus_dct.items(), key=lambda item: item[1], reverse=True)}
species_dct = {k: v for k, v in sorted(species_dct.items(), key=lambda item: item[1], reverse=True)}


plot_bar(phylum_dct, "Phylum Bar Graph")
plot_bar(class_dct, "Class Bar Graph")
plot_bar(order_dct, "Order Bar Graph")
plot_bar(family_dct, "Family Bar Graph")
plot_bar(tribe_dct, "Tribe Bar Graph")
plot_bar(genus_dct, "Genus Bar Graph")
plot_bar(species_dct, "Species Bar Graph")

plot_bar(phylum_dct, "Phylum Bar Graph Empty", empty=False)
plot_bar(class_dct, "Class Bar Graph Empty", empty=False)
plot_bar(order_dct, "Order Bar Graph Empty", empty=False)
plot_bar(family_dct, "Family Bar Graph Empty", empty=False)
plot_bar(tribe_dct, "Tribe Bar Graph Empty", empty=False)
plot_bar(genus_dct, "Genus Bar Graph Empty", empty=False)
plot_bar(species_dct, "Species Bar Graph Empty", empty=False)

create_table("Phylum", phylum_dct)
create_table("Class", class_dct)
create_table("Order", order_dct)
create_table("Family", family_dct)
create_table("SubFamily", subfamily_dct)
create_table("Tribe", tribe_dct)
create_table("Genus", genus_dct)
create_table("Species", species_dct)

fig, ax = plt.subplots(2, 4)
fig.suptitle('Pie Chart Keyance Data', fontsize=16)

plot_pie_subplot(ax[0, 0], "Phylum", phylum_dct)
plot_pie_subplot(ax[0, 1], "Class", class_dct)
plot_pie_subplot(ax[0, 2], "Order", order_dct)
plot_pie_subplot(ax[0, 3], "Family", family_dct)
plot_pie_subplot(ax[1, 0], "SubFamily", subfamily_dct)
plot_pie_subplot(ax[1, 1], "Tribe", tribe_dct)
plot_pie_subplot(ax[1, 2], "Genus", genus_dct)
plot_pie_subplot(ax[1, 3], "Species", species_dct)
figure = plt.gcf()
figure.set_size_inches(48, 28)
#plt.show()
plt.savefig("Pie Chart Keyance Data", dpi=600)


fig, ax = plt.subplots(2, 4)
fig.suptitle('Bar Chart Keyance Data', fontsize=16)

plot_bar_subplot(ax[0, 0], "Phylum", phylum_dct)
plot_bar_subplot(ax[0, 1], "Class", class_dct)
plot_bar_subplot(ax[0, 2], "Order", order_dct)
plot_bar_subplot(ax[0, 3], "Family", family_dct)
plot_bar_subplot(ax[1, 0], "SubFamily", subfamily_dct)
plot_bar_subplot(ax[1, 1], "Tribe", tribe_dct)
plot_bar_subplot(ax[1, 2], "Genus", genus_dct)
plot_bar_subplot(ax[1, 3], "Species", species_dct)
figure = plt.gcf()
figure.set_size_inches(48, 28)
#plt.show()
plt.savefig("Bar Chart Keyance Data", dpi=600)

fig, ax = plt.subplots(2, 4)
fig.suptitle('Pie Chart Keyance Data No Empty', fontsize=16)

plot_pie_subplot(ax[0, 0], "Phylum", phylum_dct, empty=False)
plot_pie_subplot(ax[0, 1], "Class", class_dct, empty=False)
plot_pie_subplot(ax[0, 2], "Order", order_dct, empty=False)
plot_pie_subplot(ax[0, 3], "Family", family_dct, empty=False)
plot_pie_subplot(ax[1, 0], "SubFamily", subfamily_dct, empty=False)
plot_pie_subplot(ax[1, 1], "Tribe", tribe_dct, empty=False)
plot_pie_subplot(ax[1, 2], "Genus", genus_dct, empty=False)
plot_pie_subplot(ax[1, 3], "Species", species_dct, empty=False)
figure = plt.gcf()
figure.set_size_inches(48, 28)
#plt.show()
plt.savefig("Pie Chart Keyance Data No Empties", dpi=600)

fig, ax = plt.subplots(2, 4)
fig.suptitle('Bar Chart Keyance Data No Empty', fontsize=16)

plot_bar_subplot(ax[0, 0], "Phylum", phylum_dct, empty=False)
plot_bar_subplot(ax[0, 1], "Class", class_dct, empty=False)
plot_bar_subplot(ax[0, 2], "Order", order_dct, empty=False)
plot_bar_subplot(ax[0, 3], "Family", family_dct, empty=False)
plot_bar_subplot(ax[1, 0], "SubFamily", subfamily_dct, empty=False)
plot_bar_subplot(ax[1, 1], "Tribe", tribe_dct, empty=False)
plot_bar_subplot(ax[1, 2], "Genus", genus_dct, empty=False)
plot_bar_subplot(ax[1, 3], "Species", species_dct, empty=False)
figure = plt.gcf()
figure.set_size_inches(48, 28)
#plt.show()
plt.savefig("Bar Chart Keyance Data No Empty", dpi=600)