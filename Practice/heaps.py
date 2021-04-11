import heapq

heap = [19, 9, 4, 10, 11, 8, 2]
print(heap)

# Transform list into heap, in-place, in O(n) time
heapq.heapify(heap)
print(heap)

heapq.heappush(heap, 5)
print(heap)

print('min', heap[0])

#remove min
min = heapq.heappop(heap)
print(heap)

print('min', heapq.heappushpop(heap, 1))
print(heap)

