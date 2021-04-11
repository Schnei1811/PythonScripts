import csv



# with open("OutputEdit.txt", newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter='\t')
#     for row in reader:
#         print(row)


with open("OutputEdit.txt",'w') as f:
    f.encoding = 'latin1'
    read_data = f.read()


print(read_data)