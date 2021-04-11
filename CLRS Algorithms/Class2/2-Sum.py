from time import clock

start = clock()


def myhash(val):
    return val >> 15


f = open('prob-2sum.txt', 'r')
valnew = [True for x in range(6103503)]
tlist = [0 for x in range(-10000, 10000 + 1)]
tmp = f.read()
f.close()
print('read complete')
vallist = [int(val) for val in tmp.split()]
vallist = set(vallist)
print('convert to set@int complete')
minval = min(vallist)
for val in vallist:
    val_key = myhash(val - minval)
    if valnew[val_key] == True:
        valnew[val_key] = [val]
    else:
        valnew[val_key].append(val)
print('hash complete', len(valnew), len(vallist))

for val in vallist:
    firkey = myhash(-10000 - val - minval)
    seckey = myhash(10000 - val - minval)
    if firkey < len(valnew):
        if valnew[firkey] != True:
            for tmp in valnew[firkey]:
                if tmp + val in range(-10000, 10000 + 1):
                    tlist[tmp + val + 10000] = 1
    if firkey < len(valnew):
        if valnew[seckey] != True:
            for tmp in valnew[seckey]:
                if tmp + val in range(-10000, 10000 + 1):
                    tlist[tmp + val + 10000] = 1

print('output: ', sum(tlist))

finish = clock()
print(finish - start)