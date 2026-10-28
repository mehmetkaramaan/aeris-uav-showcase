import heapq, math
OBSTACLES = [(11, 6, 19, 12), (24.5, -13, 31.5, -5), (35.5, 5, 38.5, 11)]

def blocked(e, n, margin=1.1):
    return not (-8 <= e <= 48 and -28 <= n <= 28) or any((x - margin <= e <= xx + margin and y - margin <= n <= yy + margin for x, y, xx, yy in OBSTACLES))

def astar(start, goal, zones=()):

    def occupied(e, n):
        return blocked(e, n) or any((math.hypot(e - z[0], n - z[1]) < z[2] + 1.1 for z in zones))
    s = tuple((round(v) for v in start))
    g = tuple((round(v) for v in goal))
    if occupied(*g):
        raise ValueError('Goal is inside an obstacle or outside the training area.')
    q = [(0, s)]
    cost = {s: 0}
    parent = {}
    while q:
        _, u = heapq.heappop(q)
        if u == g:
            route = [g]
            while route[-1] != s:
                route.append(parent[route[-1]])
            return list(reversed(route))[1:] or [g]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            v = (u[0] + dx, u[1] + dy)
            if occupied(*v) or (dx and dy and (occupied(u[0] + dx, u[1]) or occupied(u[0], u[1] + dy))):
                continue
            c = cost[u] + math.hypot(dx, dy)
            if c < cost.get(v, 1000000000.0):
                cost[v] = c
                parent[v] = u
                heapq.heappush(q, (c + math.dist(v, g), v))
    raise ValueError('No safe route between obstacles.')

def visible(a, b, zones=(), margin=2.9):
    if blocked(*a) or blocked(*b):
        return False
    for x, y, xx, yy in OBSTACLES:
        lo, hi = (0.0, 1.0)
        for p, d, mn, mx in [(a[0], b[0] - a[0], x - margin, xx + margin), (a[1], b[1] - a[1], y - margin, yy + margin)]:
            if abs(d) < 1e-09:
                if not mn <= p <= mx:
                    lo = 2.0
                    break
            else:
                t1, t2 = sorted(((mn - p) / d, (mx - p) / d))
                lo = max(lo, t1)
                hi = min(hi, t2)
        if lo <= hi:
            return False
    for x, y, r in zones:
        dx, dy = (b[0] - a[0], b[1] - a[1])
        t = max(0, min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / max(1e-09, dx * dx + dy * dy)))
        if math.hypot(a[0] + t * dx - x, a[1] + t * dy - y) < r + margin:
            return False
    return True

def rectangle_clearance(p, box):
    x, y = p
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))

def clearance_ok(position, path, zones=(), margin=3.0):
    initial = [min(margin, max(1.1, rectangle_clearance(position, b) - 0.02)) for b in OBSTACLES]
    for q in path:
        if blocked(*q):
            return False
        if any((rectangle_clearance(q, b) < m for b, m in zip(OBSTACLES, initial))):
            return False
        if any((math.dist(q, z[:2]) < z[2] + min(margin, max(1.1, math.dist(position, z[:2]) - z[2] - 0.02)) for z in zones)):
            return False
    return True

def shortest_path(start, goal, zones=()):
    start, goal = (tuple(start), tuple(goal))
    if not visible(goal, goal, zones):
        raise ValueError('Goal is inside an obstacle or exclusion zone.')
    if visible(start, goal, zones):
        return [goal]
    if not visible(start, start, zones):
        exits = []
        for x, y, xx, yy in OBSTACLES:
            exits.extend([(x - 3.2, start[1]), (xx + 3.2, start[1]), (start[0], y - 3.2), (start[0], yy + 3.2)])
        for x, y, r in zones:
            angle = math.atan2(start[1] - y, start[0] - x)
            exits.append((x + (r + 3.2) * math.cos(angle), y + (r + 3.2) * math.sin(angle)))
        options = []
        for p in exits:
            if math.dist(start, p) > 8 or not visible(p, p, zones):
                continue
            path = [(start[0] + (p[0] - start[0]) * k / 30, start[1] + (p[1] - start[1]) * k / 30) for k in range(31)]
            if not clearance_ok(start, path, zones, 2.9):
                continue
            try:
                route = [p] + shortest_path(p, goal, zones)
                options.append((sum((math.dist(a, b) for a, b in zip([start] + route, route))), route))
            except ValueError:
                pass
        if options:
            return min(options, key=lambda a: a[0])[1]
        raise ValueError('No route out of the obstacle clearance region.')
    vertices = [start, goal]
    for x, y, xx, yy in OBSTACLES:
        vertices.extend([(x - 3.2, y - 3.2), (x - 3.2, yy + 3.2), (xx + 3.2, y - 3.2), (xx + 3.2, yy + 3.2)])
    for x, y, r in zones:
        rr = (r + 3.2) / math.cos(math.pi / 24)
        vertices.extend(((x + rr * math.cos(i * math.pi / 12), y + rr * math.sin(i * math.pi / 12)) for i in range(24)))
    vertices = vertices[:2] + [p for p in vertices[2:] if visible(p, p, zones)]
    q = [(math.dist(start, goal), 0)]
    cost = {0: 0}
    parent = {}
    edges = {}
    while q:
        _, i = heapq.heappop(q)
        if i == 1:
            route = [goal]
            while i != 0:
                i = parent[i]
                route.append(vertices[i])
            return list(reversed(route))[1:]
        for j, p in enumerate(vertices):
            if i == j:
                continue
            key = tuple(sorted((i, j)))
            if key not in edges:
                edges[key] = visible(vertices[i], p, zones)
            if not edges[key]:
                continue
            c = cost[i] + math.dist(vertices[i], p)
            if c < cost.get(j, math.inf):
                cost[j] = c
                parent[j] = i
                heapq.heappush(q, (c + math.dist(p, goal), j))
    raise ValueError('No safe visibility-graph route.')
