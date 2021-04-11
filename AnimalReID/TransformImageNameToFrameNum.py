import csv

videoname = '003Razor Billed Curassow Mitu tuberoso11'

curassowlist, spinylist, tapirlist = [], [], []
animallist = []

with open('Peru/ObjectDetectorTrainingFile.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if 'sow' in row[3]:
            animallist.append([row[4], row[5], row[6], row[7], row[0].split('.')[1], row[3]])
with open('Peru/ObjectDetectorTrainingFile.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if 'piny' in row[3]:
            animallist.append([row[4], row[5], row[6], row[7], row[0].split('.')[1], row[3]])

with open('Peru/ObjectDetectorTrainingFile.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if 'apir' in row[3]:
            animallist.append([row[4], row[5], row[6], row[7], row[0].split('.')[1], row[3]])


with open('{}-Results.csv'.format(videoname), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerows(animallist)

# with open('{}-Results.csv'.format(videoname), 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerows(spinylist)
#
# with open('{}-Results.csv'.format(videoname), 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerows(tapirlist)


