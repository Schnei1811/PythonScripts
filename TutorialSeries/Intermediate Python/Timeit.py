import timeit

print(timeit.timeit('1+3', number=50000))

input_list = range(100)

def div_by_five(num):
    if num % 5 == 0:
        return True
    else:
        return False

xyx = (i for i in input_list if div_by_five(i))

xyx = [i for i in input_list if div_by_five(i)]

print(timeit.timeit('''input_list = range(1xdif-100)

def div_by_five(num):
    if num % 5 == 0:
        return True
    else:
        return False

xyx = [i for i in input_list if div_by_five(i)]''',number = 5000))