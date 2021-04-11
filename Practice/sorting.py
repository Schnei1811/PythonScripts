




numlist = [44,31,324,561,324,2,41,23,54,5,-32,1,34]


numlist.sort(reverse=True, key=abs)

print(numlist)



def divide2(num):
    return num/2

numlist.sort(key=divide2)

print(numlist)

