



stringlist = ['a', 'b', 'c','d','a','e','f','a','e']

uniquestringdict = {}

for string in stringlist:
    if string not in uniquestringdict:
        uniquestringdict[string] = 1
    else:
        uniquestringdict[string] += 1

print(uniquestringdict)

sorted = sorted(uniquestringdict.items(), key=lambda x: x[1])

print(sorted)