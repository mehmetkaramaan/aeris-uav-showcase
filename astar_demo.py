"""Independent educational A* sample from the AERIS portfolio.

This example does not connect to PX4 or include the private mission system.
Run: python astar_demo.py  (writes route.svg)
"""
import heapq
import math
from pathlib import Path


def astar(start, goal, obstacles, width, height):
    """Find a shortest 8-connected grid path; forbid diagonal corner cutting.

    Cells are integer (x, y) pairs. Obstacles must already include any desired
    vehicle clearance. The sample assumes a static, fully known grid.
    """
    blocked = set(obstacles)

    def free(p):
        return 0 <= p[0] < width and 0 <= p[1] < height and p not in blocked

    if not free(start) or not free(goal):
        raise ValueError('Start and goal must be free cells inside the grid.')
    queue = [(math.dist(start, goal), 0.0, start)]
    costs, parent = {start: 0.0}, {}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nxt = (current[0] + dx, current[1] + dy)
            if not free(nxt):
                continue
            if dx and dy and (not free((current[0] + dx, current[1]))
                              or not free((current[0], current[1] + dy))):
                continue
            candidate = cost + math.hypot(dx, dy)
            if candidate < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = candidate, current
                heapq.heappush(queue, (candidate + math.dist(nxt, goal), candidate, nxt))
    raise ValueError('No route exists on this grid.')


def demo(output='route.svg'):
    width, height = 28, 20
    obstacles = {(x, y) for x in range(8, 13) for y in range(4, 14)}
    obstacles |= {(x, y) for x in range(18, 22) for y in range(9, 18)}
    path = astar((2, 3), (25, 17), obstacles, width, height)
    scale = 22
    cells = ''.join(f'<rect x="{x*scale}" y="{(height-1-y)*scale}" '
                    f'width="{scale}" height="{scale}" fill="#454b53"/>'
                    for x, y in sorted(obstacles))
    points = ' '.join(f'{(x+.5)*scale},{(height-y-.5)*scale}' for x, y in path)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width*scale} {height*scale}">'
           f'<rect width="100%" height="100%" fill="#202226"/>{cells}'
           f'<polyline points="{points}" fill="none" stroke="#c5b58a" stroke-width="3"/>'
           '</svg>')
    Path(output).write_text(svg, encoding='utf-8')
    print(f'Path: {len(path)} cells; distance: '
          f'{sum(math.dist(a,b) for a,b in zip(path,path[1:])):.2f} grid units')
    return path


if __name__ == '__main__':
    demo()
