from time import clock
from heapq import heappush, heappop

start = clock()

f = open('median.txt', 'r')
tmp = f.read()
f.close()
data = [int(val) for val in tmp.split()]
out = [0 for x in range(len(data))]


# rudeway with high complexity
# 17s running time
def rudeway(data, out):
    for ind in range(len(data)):
        b = data[0:ind + 1]
        b.sort()
        out.append(b[(len(b) + 1) / 2 - 1])
    return sum(out) % 10000


# print(rudeway(Data,out))

# use heapq, minus(min heap)=max heap
# 0.231407100855s
def heapway(data, out):
    lheap = []
    rheap = []
    out[0] = data[0]
    tmp = sorted(data[0:2])
    out[1] = tmp[0]
    heappush(lheap, -tmp[0])
    heappush(rheap, tmp[1])
    for ind in range(2, len(data)):
        if data[ind] > rheap[0]:
            heappush(rheap, data[ind])
        else:
            heappush(lheap, -data[ind])

        if len(rheap) > len(lheap):
            heappush(lheap, -heappop(rheap))
        if len(lheap) > len(rheap) + 1:
            heappush(rheap, -heappop(lheap))
        out[ind] = -lheap[0]

    return sum(out) % 10000


print(heapway(data, out))

finish = clock()
print(finish - start)