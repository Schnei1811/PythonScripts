def q1():
    jobsFile = open('jobs.txt','r')
    lines = jobsFile.readlines()[1:]

    jobs = []
    length,weight = 0,0

    for line in lines:
        weight = int(line.split()[0])
        length = int(line.split()[1])
        jobs.append([weight,length,weight - length])

    jobs = sorted(jobs,key = lambda x:(x[2],x[0]))
    jobs = jobs[-1::-1]#inverse, decreasing order
    sumTime = 0
    sumLength = 0
    for job in jobs:
        sumLength += job[1]
        sumTime += job[0] * sumLength
    print(sumTime)

def q2():
    jobsFile = open('jobs.txt','r')
    lines = jobsFile.readlines()[1:]

    jobs = []
    length,weight = 0,0
    for line in lines:
        weight = int(line.split()[0])
        length = int(line.split()[1])
        jobs.append([weight,length,float(weight) / float(length)])

    jobs = sorted(jobs,key = lambda x:x[2])
    jobs = jobs[-1::-1]
    sumTime = 0
    sumLength = 0
    for job in jobs:
        sumLength += job[1]
        sumTime += job[0] * sumLength
    print(sumTime)

# def q3():
#     edges = [map(int, x.split(' ')) for x in open('edges.txt', 'r').read().split('\n')[1:-1]]
#     vertices = set()
#     for edge in edges:
#         vertices.add(edge[0])
#         vertices.add(edge[1])
#     spanned = set()
#     spanned.add(vertices.pop())
# 
#     total_cost = 0
#     while len(vertices) > 0:
#         best_cost = 9999999
#         for edge in edges:
#             if edge[0] in spanned and edge[1] in vertices and edge[2] < best_cost:
#                 best_cost = edge[2]
#                 best_vert = edge[1]
#             if edge[1] in spanned and edge[0] in vertices and edge[2] < best_cost:
#                 best_cost = edge[2]
#                 best_vert = edge[0]
#         spanned.add(best_vert)
#         vertices.remove(best_vert)
#         total_cost += best_cost
# 
#     # print vertices
#     #    print best_cost
#     #    print best_vert
#     #    print spanned
#     #    print total_cost
#     print total_cost

#q1()
#q2()
#2.7
q3()













