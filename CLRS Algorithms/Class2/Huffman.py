from heapq import heappush, heappop, heapify
import numpy as np
np.set_printoptions(threshold=np.NaN)

def codecreate(symbol2weights, tutor= False):
    heap = [[float(wt), [sym, []]] for sym, wt in symbol2weights.items()]
    heapify(heap)
    if tutor: print("ENCODING:", sorted(symbol2weights.items()))
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        if tutor: print(" COMBINING:", lo, 'AND:', hi)
        for i in lo[1:]: i[1].insert(0, '0')
        for i in hi[1:]: i[1].insert(0, '1')
        lohi = [lo[0] + hi[0]] + lo[1:] + hi[1:]
        if tutor: print("  PRODUCING:", lohi, '\n')
        heappush(heap, lohi)
    codes = heappop(heap)[1:]
    for i in codes: i[1] = ''.join(i[1])
    return sorted(codes, key=lambda x: (len(x[-1]), x))

#Q1 19
#Q2 9

data = np.loadtxt('Huffman.txt').astype(str)
data = data[1:]

for i in range(0, 1000): data = np.insert(data, i*2, i)
cleaned = data.tolist()

symbol2weights = dict((symbol, wt) for symbol, wt in zip(cleaned[0::2], cleaned[1::2]) )

huff = codecreate(symbol2weights, True)
print("\nSYMBOL\tWEIGHT\tHUFFMAN CODE")
for h in huff: print("%s\t%s\t%s" % (h[0], symbol2weights[h[0]], h[1]))

print(huff[2,:])














