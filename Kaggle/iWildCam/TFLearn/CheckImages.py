import csv

reader = csv.reader(open('iWildSubmissionOutput.csv', 'r'))
d = {}
for row in reader:
   k, v = row
   d[k] = v

TEST_DICT = d

print(TEST_DICT)

for file in TEST_DICT:
    print(TEST_DICT[file])
    if sum(TEST_DICT[file]) >= 3: animal = 1
    else: animal = 0
    id = file.split('.')[0]