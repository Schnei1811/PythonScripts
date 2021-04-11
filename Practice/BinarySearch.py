def binarySearch(alist, item):
    first = 0
    last = len(alist)-1

    while first <= last:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            return True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return False

#charlist = ['a', 'q', 'v', 'c', 't', 't', 'e', 'd', 'f', 's', 'q', 'z']
#item = 'q'
#print(binarySearch(charlist, item))


intlist = [1, 2, 4, 5, 75, 32, 14, 10, 11]
item = 45
for item in intlist:
    print(binarySearch(intlist, item))

item = 32
print(binarySearch(intlist, item))


ints = '1234567890'

var = "GAV4562DE"


