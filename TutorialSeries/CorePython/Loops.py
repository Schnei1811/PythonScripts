import random
import sys
import os

for x in range(0, 10):
    print(x, '', end="")

print('\n')

grocery_list = ['Juice', 'Tomatoes', 'Potatoes', 'Bananas']

for y in grocery_list:
    print(y)

for x in [2, 4, 6, 8, 10]:
    print(x)

num_list = [[1, 2, 3], [10, 20, 30], [100, 200, 300]]

for x in range(0, 3):
    for y in range(0, 3):
        print(num_list[x][y])

'numbers 0-20'
random_num = random.randrange(0, 20)

while (random_num != 15):
    print(random_num)
    random_num = random.randrange(0, 100)

'iterator'
i = 0;

while(i<=20):
    if(i%2 == 0):
        print(i)
    elif(i==9):
        break
    else:
        i+=1
        continue
    i+=1


seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
list(enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

def enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1

print()