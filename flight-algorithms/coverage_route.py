import math, time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from planning import shortest_path
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='coverage-route')

@lru_cache(maxsize=30000)
def distance(a, b):
    if a == b:
        return 0.0
    try:
        path = [a] + shortest_path(a, b)
        return sum((math.dist(x, y) for x, y in zip(path, path[1:])))
    except ValueError:
        return math.inf

def optimize(cells, covers, remaining, start, home):
    began = time.monotonic()
    uncovered = remaining.copy()
    selected = []
    gain_matrix = np.zeros((len(cells), len(cells)), dtype=np.int16)
    for i, ids in enumerate(covers):
        gain_matrix[i, ids] = 1
    while uncovered.any():
        gains = gain_matrix @ uncovered.astype(np.int16)
        maximum = int(gains.max())
        if maximum == 0:
            break
        choices = np.flatnonzero(gains == maximum)
        i = min(choices, key=lambda i: math.dist(start, cells[i]))
        selected.append(int(i))
        uncovered[covers[i]] = False
    if uncovered.any():
        raise ValueError('Some cells have no valid camera viewpoint.')
    nodes = [tuple(start)] + [tuple(cells[i]) for i in selected] + [tuple(home)]
    n = len(nodes)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = distance(nodes[i], nodes[j])
    pending = set(range(1, n - 1))
    tour = [0]
    while pending:
        nxt = min(pending, key=lambda i: matrix[tour[-1], i])
        tour.append(nxt)
        pending.remove(nxt)
    tour.append(n - 1)
    for _ in range(12):
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b, c, d = (tour[i - 1], tour[i], tour[j], tour[j + 1])
                if matrix[a, c] + matrix[b, d] + 1e-07 < matrix[a, b] + matrix[c, d]:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
        if not improved:
            break
    if any((not math.isfinite(matrix[a, b]) for a, b in zip(tour, tour[1:]))):
        raise ValueError('No route to the coverage viewpoints.')
    return {'views': [selected[i - 1] for i in tour[1:-1]], 'nominal_distance': sum((matrix[a, b] for a, b in zip(tour, tour[1:]))), 'planning_seconds': time.monotonic() - began, 'selected_views': len(selected)}
