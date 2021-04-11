

def binary_search(inputlist, val):
    first = 0
    last = len(inputlist) - 1

    while first <= last:
        mid = (first + last) // 2
        if inputlist[mid] == val: return True
        elif inputlist[mid] > val: last = mid - 1
        else: first = mid + 1
    return False


def dfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        visited.add(vertex)
        vertexlist.extend(graph[vertex] - visited)
    return False


def bfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        if vertex in visited:
            visited.add(vertex)
            vertexlist.extend(graph[vertex] - visited)
    return visited














