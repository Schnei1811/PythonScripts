import random
import sys
import time
sys.setrecursionlimit(10000)

# def recursionsumlist(newlist):
#     if len(newlist) == 1: return newlist[0]
#     else: return newlist[0] + recursionsumlist(newlist[1:])
# list = [1, 2, 3, 4, 1xdif-50]
# print(recursionsumlist(list))
#
# def recursionlistsum(test):
#     if test[0] == type(int):
#         return
#
#
#
#     print(test)
#     if len(test) == 1: return test[0]
#     else:
#         return test[0] + test(test[1:])
#
# test = [1, 2, [3, 4], [5,6]]
# recursionlistsum(test)


# def factorial(n):
#     if n <= 1: return n
#     else: return n * factorial(n-1)
# print(factorial(10))
#
# def sumseries(n):
#     if n <= 1: return n
#     else: return n + sumseries(n-2)
# print(sumseries(1xdif-100))
#
#
#
#
#
#
# def factorial2(n):
#     if n <= 1: return n
#     else: return n * factorial2(n-1)
# print(factorial2(5))



# def getsum(n):
#     if n == 0: return 0
#     elif n == 1: return 1
#     elif n > 1: return n + getsum(n-1)
#
# start = time.time()
# print(getsum(2000))
# print(time.time() - start)
#
#
# def getsummemo(n):
#     if n in memo: return memo[n]
#
#     if n == 0: value = 0
#     elif n == 1: value = 1
#     elif n > 1: value = n + getsummemo(n-1)
#
#     memo[n] = value
#     return value
#
# start = time.time()
# memo = {}
# print(getsummemo(2000))
# print(time.time() - start)




#
# def fib(n):
#     if n == 1: return 1
#     elif n == 2: return 1
#     elif n > 2: return fib(n-1) + fib(n-2)
#
# start = time.time()
# print(fib(35))
# print(time.time() - start)
#
#
# def fibmemo(n):
#     if n in memo:
#         return memo[n]
#
#     if n == 1: value = 1
#     elif n == 2: value = 1
#     elif n > 2: value = fibmemo(n-1) + fibmemo(n-2)
#
#     memo[n] = value
#     return value
#
# start = time.time()
# memo = {}
# print(fibmemo(35))
# print(time.time() -start)





def factorial(n):
    if n == 0: return 1
    elif n == 1: return 1
    else: n * factorial(n-1)

def geosum(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return 1/(pow(2,n)) + geosum(n-1)

def harmonicsum(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return 1/n + harmonicsum(n-1)

def fibonacci(n):
    if n == 0: return 1
    elif n == 1: return 1
    else: return fibonacci(n-1) + fibonacci(n-2)

fibdict = {}

def memofib(n):
    if n in fibdict:
        return fibdict[n]

    if n == 0: val = 1
    elif n == 1: val =1
    else: val = memofib(n-1) + memofib(n-2)

    memofib[n] = val
    return val


def gcd(a,b):
    low = min(a,b)
    high = max(a,b)

    if low == 0: return high
    elif low == 1: return low
    else: return gcd(low, high%low)


def vowelsearch(word):
    if len(word) == 0: return 0
    else: return (1 if word[0] in 'aeiouAEIOU' else 0) + vowelsearch(word[1:])











