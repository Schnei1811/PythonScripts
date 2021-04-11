

def dfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        visited.add(vertex)
        vertexlist.extend(graph[vertex] - visited)
    return visited


graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}


print(dfs(graph, 'A', 'F'))

