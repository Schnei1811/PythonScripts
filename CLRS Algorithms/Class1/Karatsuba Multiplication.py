# #Recursion in computer science is a method where the solution to a problem is
# # based on solving smaller instances of the same problem.
# int1 = 3141592653589793238462643383279502884197169399375105820974944592
# int2 = 2718281828459045235360287471352662497757247093699959574966967627
#
# int1str = ''
# int2str = ''
#
# for character in str(int1):
#     int1str = int1str + character
# for character in str(int2):
#     int2str = int2str + character
#
# a = int(int1str[:int(len(int1str)/2)])
# b = int(int1str[int(len(int1str)/2):])
# c = int(int2str[:int(len(int2str)/2)])
# d = int(int2str[int(len(int2str)/2):])
#
# total = int(10**len(int1str)*a*c + 10**(len(int1str)/2)*(a*d+b*c) + b*d)
#
# print(total)
#

print('Karatsuba multiplication in Python')
x=input("first_number=")
y=input("second_number=")
print('------------------------')
x=int(x)
y=int(y)
import math
import time
def karatsuba(x,y):

  x=str(x)
  y=str(y)

  len_x=len(x)
  len_y=len(y)

  if(int(len_x)==1 or int(len_y)==1):
    return int(x)*int(y)
  else:

    B=10
    exp1=int(math.ceil(len_x/2.0))
    exp2=int(math.ceil(len_y/2.0))
    if(exp1<exp2):
      exp=exp1
    else:
      exp=exp2
    m1=len_x-exp
    m2=len_y-exp
    a=karatsuba(int(x[0:m1]),int(y[0:m2]))
    c=karatsuba(int(x[m1:len_x]),int(y[m2:len_y]))
    b=karatsuba(int(x[0:m1])+int(x[m1:len_x]),int(y[0:m2])+int(y[m2:len_y]))-a-c
    results=a*math.pow(10,2*exp) + b*math.pow(10,exp) + c
    return int(results)

start_time=time.time()
ctrl = x*y
tpt=time.time() - start_time
print (x,'*',y,'=',ctrl)
print("--- %s seconds ---" % tpt)

start_time=time.time()
output=karatsuba(x,y)
tpt=time.time() - start_time
print ('karatsuba(',x,',',y,')=',output)
print("--- %s seconds ---" % tpt)