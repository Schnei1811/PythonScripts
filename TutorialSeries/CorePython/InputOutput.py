import random
import sys
import os

#Use "ab+" to Read & Append to File (it also opens or creates the file)
test_file = open("ChargingBehaviour1Data.txt", "wb")
print(test_file.mode)
print(test_file.name)
test_file.write(bytes("Write me to the file\n", 'UTF-8'))
test_file.close()
#reading and writing is r+
test_file = open("ChargingBehaviour1Data.txt", "r+")
text_in_file = test_file.read()
print(text_in_file)
os.remove("ChargingBehaviour1Data.txt")

