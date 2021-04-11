


def binarysearch(nums, item):
    first = 0
    last = len(nums)-1

    while first <= last:
        midpoint = (first+last)//2
        if nums[midpoint] == item:
            return True
        else:
            if item < midpoint:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return False

def binary_search(nums, item):
    first = 0
    last = len(nums) - 1

    while first <= last:
        midpoint = (first + last) // 2
        if nums[midpoint] == item:
            return True
        elif nums[midpoint] > item:
            last = midpoint - 1
        else:
            first = midpoint + 1
    return False







nums = [1, 2, 4, 5, 75, 32, 14, 10, 11]
nums.sort()

print(nums)

for val in nums:
    print(binarysearch(nums, val))

for val in nums:
    print(binary_search(nums, val))





def binary_int_search(numlist, val):
    first = 0
    last = len(numlist)-1

    while first <= last:
        mid = (first + last) // 2
        if numlist[mid] == val:
            return True
        elif numlist[mid] > val:
            last = mid - 1
        else:
            first = mid + 1
    return False

intlist = [4,21,6523,3,1,34,5,23,42,1,2,5,3]

intlist.sort()

for val in intlist:
    print(binary_int_search(intlist, val))

print(binary_int_search(intlist, 5423))





def binary_int_search(intlist, val):
    first = 0
    last = len(intlist)-1

    while first <= last:
        mid = (first+last) // 2
        if intlist[mid] == val: return True
        elif intlist[mid] > val: last = mid - 1
        else: first = mid + 1
    return False

intlist = [14,51233,451234,2314,12,23,141,34,1243321,3]

intlist.sort()

for num in intlist:
    print(binary_int_search(intlist, num))
print(binary_int_search(intlist, 5))














def binary_char_search(charlist, val):
    first = 0
    last = len(charlist) - 1

    while first <= last:
        mid = (first + last)//2
        if charlist[mid] == val: return True
        elif charlist[mid] > val: last = mid -1
        else: first = mid +1
    return False



charlist = ['ape', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

for char in charlist:
    print(binary_char_search(charlist, char))

print(binary_char_search(charlist, 'ape'))
print(binary_char_search(charlist, 'ap'))







def binary_search(list, val):
    first = 0
    last = len(list) - 1

    while first <= last:
        mid = (first + last) // 2
        if list[mid] == val:
            return True
        elif list[mid] > val:
            last = mid - 1
        else:
            first = mid + 1
    return True























def binary_search(list, val):
    first = 0
    last = len(list) -1

    while first <= last:
        mid = (first + last) // 2
        if list[mid] == val: return True
        elif list[mid] > val: last = mid - 1
        else: first = mid + 1
    return False


































