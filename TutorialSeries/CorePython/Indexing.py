import numpy as np

x = np.arange(10)
print(x)
print(x[2])
print('\n', x[-2])

x.shape = (2,5)
print(x)
print(x[1,3])       #second row, 4th coloum
print(x[1,-1])      #second row, last number
print(x[0])         #first row

x = np.arange(10)
print('\n', x[2:5])
print(x[-7])
print(x[:-7])       #all of before -7
print(x[-7:])       #all of after -7
print(x[1:7:2])     #the second number between 1-7
print(x[0:9:4])     #the fourth number between 0-9

y = np.arange(35).reshape(5,7)  #count 35, 5 lines 7 coloums
print('\n', y)
print(y[1:3])       #Rows 1:2       Not including 0, 3 and 4
print(y[3:3])       #Empty
print(y[3:4])       #Rows 4:4       Row 4 but not 5

#Coloumn indicated by the second number
print('\n', y[:, 1:2])      #All Rows, 2nd Coloumn (starting from 1)
print(y[:3, :-3])           #All Rows minus the last 3 coloums

print('\n', y[0:4:1])       #All rows
print(y[0:4:2])             #Every 2 rows
print(y[0:4:3])             #Every 3 rows
print(y[3:4,::3])           #4th row. Every 3 coloumns

x = np.arange(10,1,-1)
print('\n',x)
print(x[np.array([3,3,1,8])],'\n')    #The 4th, 4th, 2nd, and 9th

a = np.zeros((3,3))
v = np.array([[1,2,3]])
print(v)
a[1,1] = np.dot(v,np.transpose(v))
print(a)

a = [3,3,3,3]
b = 3*np.ones(3).reshape(3,1)
print(a)
print(b)
c= a*b
print(c)


print(np.arange(25))































