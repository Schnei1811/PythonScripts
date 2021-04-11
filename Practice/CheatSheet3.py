nums = [1,2,3,4]
mylist = [n*n for n in nums]
#list of tuples
mylist2 = [(letter, num) for letter in 'abcd' for num in range(4)]

names = ['Bruce', 'Clark', 'Peter', 'Logan', 'Wade']
heros = ['Batman', 'Superman', 'Spiderman', 'Wolverine', 'Deadpool']
my_dict = {name: hero for name, hero in zip(names, heros)}

if 'Bruce' in my_dict:pass
for key in my_dict:print(key)               #get key
for key in my_dict:print(my_dict[key])      #get values
listofkeys = my_dict.items()
listofvals = my_dict.values()

def secondtuple(tuples):
    return tuples[1]

def sort_last(tuples):
    return sorted(tuples, key=secondtuple)

import itertools
list(itertools.permutations([1, 2, 3]))

#string to list
a = [x for x in 'abcdefgh']

string = 'di'
if isinstance(string, str) == True:pass

def merge(list1, list2):
    list1.sort()
    list2.sort()
    mergelist = []
    while list1 and list2:
        if list1[0] < list2[0]:
            mergelist.append(list1.pop(0))
        else:
            mergelist.append(list2.pop(0))
    mergelist.extend(list1)
    mergelist.extend(list2)
    return mergelist

_end = '_end_'
def make_trie(*words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[_end] = _end
    return root

print(make_trie('foo','bar','baz','barz'))