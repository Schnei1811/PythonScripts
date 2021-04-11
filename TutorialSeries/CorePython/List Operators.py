#lists
list = [1, 2, 3]

print(list.append(4))                       #Add element to end of list
print(list.insert(0, 0))                    #Insert index and element
print(list.extend([4,5,6]))                 #Adds list to list. Similar to + +=
print(list.index(2))                        #Returns first index of element. Throws error if not present
for i in list:
    if list[i] == 2: print(i)               #Finds index with possibility of error
print(list.remove(2))                       #Searches for first instance and removes. Throws error if not present
print(list.sort())                          #Sorts list but does not return it. Sorted fnc preferred
print(list.reverse())                       #Reverses list in place
print(list.pop())                           #Removes and returns element of index. Default last element

print(sorted(list, reverse=True))
