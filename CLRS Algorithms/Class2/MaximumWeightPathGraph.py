import numpy as np

data = np.loadtxt('MWIS.txt').astype(str)
data = data[1:]

for i in range(0, 1000): data = np.insert(data, i*2, i+1)
cleaned = data.tolist()

data = np.loadtxt('MWIS.txt')

def max_weighted_independent_set_in_path_graph(weights):
    """ Computes the independent set with maximum total weight for a path graph.
    A path graph has all vertices are connected in a single path, without cycles.
    An independent set of vertices is a subset of the graph vertices such that
    no two vertices are adjacent in the path graph.
    Complexity: O(n); Space: O(n)
    Args:
        weights: list, of vertex weights in the order they are present in the
            graph.
    Returns:
        list, format [max_weight: int, vertices: list]
    """

    # 0. Initialization: A[i] - max total weight of the independent set for
    # the first i vertices in the graph.
    a = [0] * (len(weights)+1)
    a[0] = 0 # Max weight for empty graph.
    a[1] = weights[0] # Max weight for the graph with only the first weight.

    # 1. Compute the max total weight possible for any independent set.
    for i in range(2, len(weights)+1):
        a[i] = max(a[i-1], a[i-2]+weights[i-1])

    max_weight = a[len(weights)]

    # 2. Trace back from the solution through the subproblems to compute the
    # vertices in the independent set.
    vertices = []
    i = len(weights)
    while (i>0):
        if a[i-2] + weights[i-1] >= a[i-1]:
            vertices.insert(0, weights[i-1])
            i -= 2
        else:
            i -= 1

    return [max_weight, vertices]

#Q3 = 10100110

maxweights, vertices = max_weighted_independent_set_in_path_graph(data)

print(maxweights, vertices)
print(cleaned)