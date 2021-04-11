

def bfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            vertexlist.extend(graph[vertex] - visited)
    return visited


def bfs_path(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))









graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}


print(bfs(graph, 'A', 'F'))


print(list(bfs_path(graph, 'A', 'F')))