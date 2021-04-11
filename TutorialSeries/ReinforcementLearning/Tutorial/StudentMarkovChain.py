import numpy as np
import random

#Discount Factor
Discount = 0.9

#State Transition Matrix
STM = [[0, 0.5, 0, 0, 0, 0.5, 0],
       [0, 0, 0.8, 0, 0, 0, 0.2],
       [0, 0, 0, 0.6, 0.4, 0, 0],
       [0, 0, 0, 0, 0, 0, 0.99],
       [0.2, 0.4, 0.4, 0, 0, 0, 0],
       [0.1, 0, 0, 0, 0, 0.9, 0],
       [0, 0, 0, 0, 0, 0, 0.99]]

#State Return Value
SRV = np.array([[-2], [-2], [-2], [10], [1], [-1], [0]])

initial_value_state = np.zeros((9, 1))

#print((1 - np.multiply(Discount, STM))**-1 * SRV)

State = 0
Path = []

while True:
    Decision = random.random()
    for i in range(0, 6):
        Decision -= STM[State][i]
        if Decision < 0:
            State = i
            if State == 0: print('Class1')
            elif State == 1: print('Class2')
            elif State == 2: print('Class3')
            elif State == 3: print('Pass')
            elif State == 4: print('Pub')
            elif State == 5: print('FB')
            elif State == 6: print('Sleep')
            break
    Path.append(State)
    if State == 3:
        print('Passed!')
        break

G1 = 0
for i in range(0, len(Path)):
    G1 += SRV[Path[i]]*Discount**i
    print(G1, Path[i])

v = SRV / (1- np.multiply(Discount, STM))
print(v)
















