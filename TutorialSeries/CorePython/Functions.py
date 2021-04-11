import random
import sys
import os

def addNumber(fNum,lNum):           #Define Function
    sumNum = fNum + lNum
    return sumNum                   #Cannot Reference sumNum outside of function. Out of Scope

print(addNumber(1,4))

print('What is your name?')
name = sys.stdin.readline()
print('Hello',name)

long_string = "I'll catch you if you fall - The Floor"

print(long_string[:4])
print(long_string[-5:])
print(long_string[:-5])
print(long_string[:4] + " be there")        #concatenate
#output character %c     %s string shows up     %d signed interget     %.5f number with a decimal place with at least 5 places
print("%c is my %s letter and my number %d number is %.5f" %
      ('X','favourite',1, .14))

print(long_string.capitalize())                 #Capatilize first letter of string
print(long_string.find("Floor"))                #Character number
print(long_string.isalpha())                    #Return true all characters have been entered into the string are all letters
print(long_string.isalnum())                    #Make sure everything is a number
print(len(long_string))
print(long_string.replace("Floor", "Ground"))
print(long_string.strip())
quote_list=long_string.split(" ")
print(quote_list)

