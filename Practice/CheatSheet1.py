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
strlist = [char for char in string]
#String[:3]		    beginning to 3
#String[3:]		    3 to end
#String[-3:]		last 3
#String[:-3]		beginning to last three

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
print(''.join(['Hi ', 'World']))            #To join list into string
print(sorted(list, reverse=True))

def bfs_path(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def bfs(graph, start, val):
    visited = set()
    queue = [start]

    while queue:
        vertex = queue.pop(0)
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

def dfs(graph, start, val):
    visited = set()
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

