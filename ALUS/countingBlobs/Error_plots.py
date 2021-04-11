import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

def build_error_plot():



    return



csv_dir = "G:\\PythonData\\ALUS\\20_12_06_Alus_Added_Aranea_NonMite\\adam\\performance_report_0.csv"

img_dir = "G:\\PythonData\\ALUS\\20_12_06_Alus_Added_Aranea_NonMite\\ALUS_Classifications"

performance_report = []

with open(csv_dir, newline="") as csvfile:
    test = csv.reader(csvfile, delimiter=",")
    for row in test:
        performance_report.append(row)

test_dict = {}

for row in performance_report[1:]:
    if row[1] not in test_dict:
        test_dict[row[1]] = [[row[0], row[2]]]
    else:
        test_dict[row[1]].append([row[0], row[2]])

print(performance_report)
print(test_dict)

for key in test_dict:
    for i, items in enumerate(test_dict[key]):
        image = cv2.imread(os.path.join(img_dir, items[0]))

        #plt.subplot((len(test_dict[key]) // 4) + 1, 4, ((i // 4) + 1, (i % 4) + 1))
        print("rows", 4)
        print("col", (len(test_dict[key]) // 4))

        print("addy", ((i % 4) + 1, (i // 4) + 1))
        plt.subplot(4, (len(test_dict[key]) // 4), i+1)
        #plt.subplot((len(test_dict[key]) // 4) + 1, 4, ((i % 4) + 1, (i // 4) + 1))
        plt.imshow(image)
    plt.show()