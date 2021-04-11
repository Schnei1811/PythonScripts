
def dfs(graph, start, val):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            if vertex == val: return True
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


# undirected because connected nodes have each value

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

print(dfs(graph, 'A', 'B'))



# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#     while stack:
#         vertex = stack.pop()
#         if vertex not in visited:
#             if vertex == val: return True
#             visited.add(vertex)
#             stack.extend(graph[vertex] - visited)
#     return visited

#
#
# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#     while stack:
#         vertex = stack.pop()
#         if vertex == val: return True
#         visited.add(vertex)
#         stack.extend(graph[vertex] - visited)
#     return visited
#
#
#
# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#
#     while stack:
#         vertex = stack.pop()
#         if vertex == val: return True
#         visited.add(vertex)
#         stack.extend(graph[vertex] - visited)
#     return visited
#
#
#
#
# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#
#     while stack:
#         vertex = stack.pop()
#         if vertex == val: return True
#         visited.add(vertex)
#         stack.extend(graph[vertex] - visited)
#     return visited
#
#
# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#
#     while stack:
#         vertex = stack.pop()
#         if vertex == val: return True
#         visited.add(vertex)
#         stack.extend(graph[stack] - visited)
#     return visited

#
# def dfs(graph, start, val):
#     visited = set()
#     stack = [start]
#
#     while stack:
#         vertex = stack.pop()
#         if vertex == val: return True
#         visited.add(vertex)
#         stack.extend(graph[vertex] - visited)
#     return visited



def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

print(list(dfs_paths(graph, 'A', 'F')))








def dfs(graph, start, val):
    visited = set()
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex == val: return True
        visited.add(vertex)
        stack.extend(graph[vertex] - visited)
    return visited




def bfs(graph, start, val):
    visited = set()
    queue = [start]

    while queue:
        vertex = queue.pop()
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited





def dfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        visited.add(vertex)
        vertexlist.extend(graph[vertex] - visited)
    return visited




def bfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex not in visited:
            visited.add(vertex)
            vertexlist.extend(graph[vertex] - visited)
    return visited

print(bfs(graph, 'A', 'B'))





def dfs(graph, start, val):
    visited = set()
    vertexlist = start

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        visited.add(vertex)
        vertexlist.extend(graph[vertex] - visited)
    return visited


def bfs(graph, start, val):
    visited = set()
    vertexlist = start

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        if vertex not in visited:
            visited.add(vertex)
            vertexlist(graph[vertex] - visited)
    return vertexlist




def dfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        visited.add(vertex)
        vertexlist.extend(graph[vertex] - visited)
    return visited






def bfs(graph, start, val):
    visited = set()
    vertexlist = [start]

    while vertexlist:
        vertex = vertexlist.pop()
        if vertex == val: return True
        if vertex in visited:
            visited.add(vertex)
            vertexlist.extend(graph[vertex] - vertexlist)
    return visited













































