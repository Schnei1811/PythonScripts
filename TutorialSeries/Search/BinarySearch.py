



def binarysearch(inputlist, val):
    first = 0
    last = len(inputlist) -1

    while first <= last:
        mid = (first + last) // 2
        if inputlist[mid] == val: return True
        elif inputlist[mid] > val: last = mid - 1
        else: first = mid + 1
    return False



searchlist = [53,41,23,56,32,13,5,7,8,4,2,3123,5,743,2,1,3]
searchlist.sort()

print(binarysearch(searchlist, 13))
print(binarysearch(searchlist, 1234))


vocablist = ['apple', 'orange', 'banana', 'kiwi', 'pineapple']
vocablist.sort()

print(binarysearch(vocablist, 'orange'))
print(binarysearch(vocablist, 'watermelon'))