import random
import sys
import os
import numpy as np

print("Hello World")

#Comment
'''
Multiline Comment
'''
name = "Stefan"
print(name)

' Numbers String Lists Tuples Dictionaries(maps)'
'''
+ - * /
% modualist. returns remainder of division
** exponential calculations
// perform a division and discard remainder all together and round down)
Order of operations matter    BEDMAS
, indicates separation of variables
'''

print("5+2 = ", 5+2)
print("5-2 = ", 5-2)
print("5*2 = ", 5*2)
print("5/2 = ", 5/2)
print("5%2 = ", 5%2)
print("5**2 = ", 5**2)
print("5//2 = ", 5//2)

quote = "\"Always Remember you are unique"
multi_line_quote = ''' just
like everyone else'''

new_string = quote + multi_line_quote
print("%s %s %s" % ('I like the quote', quote, multi_line_quote))

print('\n' * 5)
'"\ n" prints new lines'

print("I don't like ", end="")
print("newlines")

#mu = np.zeros((numfeatures,1))
#sigma = np.zeros((numfeatures,1))
#Xnorm = np.zeros((mtrain,numfeatures))