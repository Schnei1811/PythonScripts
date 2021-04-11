import random
import sys
import os

'''if else elif
==   equal
!=   not equal
> < >= <='''

age = 24

if age > 16 :
    print('You are old enough to drive')
else :
    print('You are not old enough to drive')

if age >= 21 :
    print('You are old enough to drive a tractor trailer')
elif age >= 16:
    print('You are old enough to drive a car')
else :
    print('You are not old enough to drive')

if ((age >=1) and (age <= 18)):
    print("You get a birthday")
elif (age == 21) or (age >= 65):
    print("You get a birthday")
elif not(age == 30):
    print("You don't get a birthday")
else:
    print("You get a birthday party yeah")



