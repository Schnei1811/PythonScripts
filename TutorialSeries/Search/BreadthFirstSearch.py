



def bfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            vertexlist.extend(graph[vertex] - visited)
    return False




graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}


print(bfs(graph, 'A', 'F'))







list1 = [1,245,214,11,32,135,5,132]
list2 = [3,42,1,5,3,2,13,145]
list1.sort()
list2.sort()
mergedlist = []

while len(list1) and len(list2):
    if list1[0] < list2[0]:
        mergedlist.append(list1.pop(0))
    else:
        mergedlist.append(list2.pop(0))

mergedlist.extend(list1)
mergedlist.extend(list2)

print(mergedlist)


#strings
string = 'Hello World!'

print(string.lower())                       #lower case
print(string.upper())
print(string.strip('H'))                    #Remove outer character. default space
print(string.isalpha())                     #True if all characters of string alphabetical
print(string.isdigit())
print(string.isspace())
print(string.startswith('H'))               #Returns predicate if first character matches
print(string.endswith('!'))                 #Returns predicate if last character matches
print(string.find('W'))                     #Returns location of character
print(string.replace('Hello', 'Goodbye'))   #Replaces str with str
print(string.split(' '))                    #Convert string to list separated by delimited
print(string.join(['Hi ', 'World']))        #Joins list with string as delimiter
print(''.join(['Hi ', 'World']))            #To join list into string


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







def vowelsearch(word):
    if word == '': return 0
    else: return (1 if word[0] in 'aeiouAEIOU' else 0) + vowelsearch(word[1:])

print(vowelsearch('wqcdasrreia'))




def binary_search(inputlist, val):
    first = 0
    last = len(inputlist) - 1

    while first <= last:
        mid = (first+last) // 2
        if inputlist[mid] == val: return True
        elif inputlist[mid] > val: last = mid-1
        else: first = mid + 1
    return False

















