import random
import sys
sys.setrecursionlimit(10000)

list = [int(1000*random.random()) for i in range(1000)]

for i,j in enumerate(list):
    if i == 0: sum = list[i]
    else: sum = sum + list[i]
print(sum)

def recursionsumlist(list):
    if len(list) == 1: return list[0]
    else: return list[0] + recursionsumlist(list[1:])
print(recursionsumlist(list))

def to_string(n, base):
    conver_tString = "0123456789ABCDEF"
    if n < base: return conver_tString[n]
    else: return to_string(n // base, base) + conver_tString[n % base]
print(to_string(2835, 16))

def recursive_list_sum(data_list):
    total = 0
    for element in data_list:
        if type(element) == type([]): total = total + recursive_list_sum(element)
        else: total = total + element
    return total
print(recursive_list_sum([1, 2, [3, 4], [5, 6]]))

def factorial(n):
    if n <= 1: return 1
    else: return n * (factorial(n - 1))
print(factorial(5))

def sum_series(n):
    if n < 1: return 0
    else: return n + sum_series(n-2)
print(sum_series(1000))

def harmonic_sum(n):
    if n < 2: return 1
    else: return 1 / n + (harmonic_sum(n - 1))
print(harmonic_sum(7))

def geometric_sum(n):
    if n < 0: return 0
    else: return 1 / (pow(2, n)) + geometric_sum(n - 1)
print(geometric_sum(7))

def power(a, b):
    if b == 0: return 1
    elif a == 0: return 0
    elif b == 1: return a
    else: return a * power(a, b - 1)
print(power(2, 4))

# exp(2, 4)
# +-- 2 * exp(2, 3)
# |       +-- 2 * exp(2, 2)
# |       |       +-- 2 * exp(2, 1)
# |       |       |       +-- 2 * exp(2, 0)
# |       |       |       |       +-- 1
# |       |       |       +-- 2 * 1
# |       |       |       +-- 2
# |       |       +-- 2 * 2
# |       |       +-- 4
# |       +-- 2 * 4
# |       +-- 8
# +-- 2 * 8
# +-- 16

def GreatestCommonDemoninator(a, b):
    low = min(a, b)
    high = max(a, b)
    if low == 0: return high
    elif low == 1: return 1
    else: return GreatestCommonDemoninator(low, high % low)
print(GreatestCommonDemoninator(4, 14))