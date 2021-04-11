import heapq

array = [15, 12, 10, 1, 20, 16, 25]

greaterlist = []
lesslist =[]


for i in range(len(array)):
	if i == 0: 
        print(array[i])
        temp = array[i]
	elif i == 1:
		if temp > array[i]: 
            greaterlist.append(temp)
            lesslist.append(array[i])
        else:
	        greatestlist.append(array[i])
	        lesslist.append(temp)
    heapq.heapify(greaterlist)
    heapq.heapify_max(lesslist)
    print((greaterlist[0] + lesslist[0])/2)
	else:
		if i % 2 == 1: