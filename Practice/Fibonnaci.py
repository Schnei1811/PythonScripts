



def fibonnaci(val):
    a = 0
    b = 1
    for i in range(val):
        a,b = b, a+b
    return a
print(fibonnaci(8))


def fib_rec(n):
    if n == 1: return 1
    elif n == 2: return 1
    elif n > 2:
        return fib_rec(n-1) + fib_rec(n-2)

numlist = []

for i in range(1, 25):
    numlist.append(i)

#fiblist = [fib_rec(n) for n in numlist]
#print(fiblist)

fib_cache = {}

def fib_rec_cache(n):
    if n in fib_cache:
        return fib_cache[n]

    if n == 1: value = 1
    elif n == 2: value = 1
    elif n > 2:
        value = fib_rec(n-1) + fib_rec(n-2)

    fib_cache[n] = value
    print(value, fib_cache)
    return value

#fiblist = [fib_rec_cache(n) for n in numlist]

#print(fiblist)

















def fib(n):
    if n == 1: return 1
    elif n == 2: return 1
    elif n > 2:
        return fib(n-1) + fib(n-2)


print(fib(10))

















def fib(n):
    if n == 1: return 1
    elif n == 2: return 1
    elif n > 2: return fib(n-1) + fib(n-2)


















fib_cache ={}

def fib_memo(n):
    if n in fib_cache:
        return fib_cache[n]

    if n == 1: value = 1
    elif n == 2: value = 2
    elif n > 2: value = fib(n-1) + fib(n-2)

    fib_cache[n] = value
    return value






fib_cache = {}

def fib_memo(n):
    if n in fib_cache:
        return fib_cache[n]

    if n == 1: value = 1
    elif n == 2: value = 1
    elif n > 2: value = fib_memo(n-1) + fib_memo(n-2)

    fib_cache[n] = value
    return value















fib_cache = {}

def fib_memo(n):
    if n in fib_cache:
        return fib_cache[n]

    if n == 1: value = 1
    elif n == 2: value = 1
    elif n > 2: value = fib_memo(n-1) + fib_memo(n-2)

    fib_cache[n] = value
    return value









































