# Solves a raster occupancy grid with classic search algorithm

import numpy as np
import heapq
import time
from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy.spatial import distance

class PriorityQueue:
    # Source: https://www.redblobgames.com/pathfinding/a-star/implementation.html
    def __init__(self):
        self.elements = []
    def empty(self):
        return len(self.elements) == 0
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        return heapq.heappop(self.elements)[1]

def getNeighbors(i, m, n, env, ntype = 4):
    '''Get all valid cells in a cells neighborhood.

    Args:
        i (int, int): Row, column coordinates of cell in environment.
        m (int): Number of rows in environment.
        n (int): number of columns in environment.
        env (2D list of int): Occupancy grid.
        occupancyFlag (int): Flag that indicates occupancy.

    Returns:
        List of (int, int): Neighbors as (row, col).
    '''

    B = [] # Initialize list of neighbors

    # Diagonals may require checking muliple locations feasibility.
    # Use these booleans to avoid repeated checks.
    upAllowed        = False
    downAllowed      = False
    leftAllowed      = False
    rightAllowed     = False
    upleftAllowed    = False
    uprightAllowed   = False
    downleftAllowed  = False
    downrightAllowed = False

    # Check the neighbors and append to B if within bounds and feasible.

    # 4-way neighborhood

    # Up
    if(i[0] - 1 >= 0):
        if(env[i[0] - 1][i[1]] == 0):
            upAllowed = True
            B.append((i[0] - 1, i[1]))
    # Down
    if(i[0] + 1 < m):
        if(env[i[0] + 1][i[1]] == 0):
            downAllowed = True
            B.append((i[0] + 1, i[1]))
    # Left
    if(i[1] - 1 >= 0):
        if(env[i[0]][i[1] - 1] == 0):
            leftAllowed = True
            B.append((i[0], i[1] - 1))
    # Right
    if(i[1] + 1 < n):
        if(env[i[0]][i[1] + 1] == 0):
            rightAllowed = True
            B.append((i[0], i[1] + 1))

    # 8-way neighborhood
    if ntype >= 8:
        # Up-Left
        if(i[0] - 1 >= 0 and i[1] - 1 >= 0):
            if(env[i[0] - 1][i[1] - 1] == 0):
                upleftAllowed = True
                B.append((i[0] - 1, i[1] - 1))
        # Up-Right
        if(i[0] - 1 >= 0 and i[1] + 1 < n):
            if(env[i[0] - 1][i[1] + 1] == 0):
                uprightAllowed = True
                B.append((i[0] - 1, i[1] + 1))
        # Down-Left
        if(i[0] + 1 < m and i[1] - 1 >= 0):
            if(env[i[0] + 1][i[1] - 1] == 0):
                downleftAllowed = True
                B.append((i[0] + 1, i[1] - 1))
        # Down-Right
        if(i[0] + 1 < m and i[1] + 1 < n):
            if(env[i[0] + 1][i[1] + 1] == 0):
                downrightAllowed = True
                B.append((i[0] + 1, i[1] + 1))

    # 16-way neighborhood
    if ntype >= 16:
        # Up-Up-Left
        if(i[0] - 2 >= 0 and i[1] - 1 >= 0 and upAllowed \
           and upleftAllowed and leftAllowed):
            if(env[i[0] - 2][i[1] - 1] == 0):
                B.append((i[0] - 2, i[1] - 1))
        # Up-Up-Right
        if(i[0] - 2 >= 0 and i[1] + 1 < n and upAllowed \
           and uprightAllowed and rightAllowed):
            if(env[i[0] - 2][i[1] + 1] == 0):
                B.append((i[0] - 2, i[1] + 1))
        # Up-Left-Left
        if(i[0] - 1 >= 0 and i[1] - 2 >= 0 and upAllowed \
           and upleftAllowed and leftAllowed):
            if(env[i[0] - 1][i[1] - 2] == 0):
                B.append((i[0] - 1, i[1] - 2))
        # Up-Right-Right
        if(i[0] - 1 >= 0 and i[1] + 2 < n and upAllowed \
           and uprightAllowed and rightAllowed):
            if(env[i[0] - 1][i[1] + 2] == 0):
                B.append((i[0] - 1, i[1] + 2))
        # Down-Down-Left
        if(i[0] + 2 < m and i[1] - 1 >= 0 and downAllowed \
           and downleftAllowed and leftAllowed):
            if(env[i[0] + 2][i[1] - 1] == 0):
                B.append((i[0] + 2, i[1] - 1))
        # Down-Down-Right
        if(i[0] + 2 < m and i[1] + 1 < n and downAllowed \
           and downrightAllowed and rightAllowed):
            if(env[i[0] + 2][i[1] + 1] == 0):
                B.append((i[0] + 2, i[1] + 1))
        # Down-Left-Left
        if(i[0] + 1 < m and i[1] - 2 >= 0 and downAllowed \
           and downleftAllowed and leftAllowed):
            if(env[i[0] + 1][i[1] - 2] == 0):
                B.append((i[0] + 1, i[1] - 2))
        # Down-Right-Right
        if(i[0] + 1 < m and i[1] + 2 < n and downAllowed \
           and downrightAllowed and rightAllowed):
            if(env[i[0] + 1][i[1] + 2] == 0):
                B.append((i[0] + 1, i[1] + 2))
    return B


# Dijkstra
def dijkstra(grid, start, goal, ntype = 4, trace = False):
    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    if trace:
        tgrid = grid.copy()
    else:
        tgrid = None

    # Explore
    while not frontier.empty():
        current = frontier.get()
        if trace:
            tgrid[current] = 0.25

        # Check if goal
        if current == goal:
            break
        # Add suitable neighbors

        neighbors = getNeighbors(current, rows, cols, grid, ntype)
        for next in neighbors:
            if next[0] < 0 or next[1] < 0 or next[0] >= rows or next[1] >= cols:
                continue
            if grid[next] != 0:
                continue
            new_cost = cost_so_far[current] + distance.euclidean(next, current)
            update = False
            if next not in cost_so_far:
                update = True
            else:
                if new_cost < cost_so_far[next]:
                    update = True
            if update:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        if trace:
            tgrid[current] = 0.5
        path.append(current)
        current = came_from[current]
    if trace:
        tgrid[start] = 0.5
    path.append(start)
    path.reverse()

    return path, tgrid

# A*
def astar(grid, start, goal, ntype = 4, trace = False):
    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    if trace:
        tgrid = grid.copy()
    else:
        tgrid = None

    # Explore
    while not frontier.empty():
        current = frontier.get()
        if trace:
            tgrid[current] = 0.25
        # Check if goal
        if current == goal:
            break
        # Add suitable neighbors
        neighbors = getNeighbors(current, rows, cols, grid, ntype)

        for next in neighbors:
            if next[0] < 0 or next[1] < 0 or next[0] >= rows or next[1] >= cols:
                continue
            if grid[next] != 0:
                continue
            new_cost = cost_so_far[current] + distance.euclidean(current, next)
            update = False
            if next not in cost_so_far:
                update = True
            else:
                if new_cost < cost_so_far[next]:
                    update = True
            if update:
                cost_so_far[next] = new_cost
                priority = new_cost + distance.euclidean(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        if trace:
            tgrid[current] = 0.5
        path.append(current)
        current = came_from[current]
    if trace:
        tgrid[start] = 0.5
    path.append(start)
    path.reverse()

    return path, tgrid


def selectNeighborRules(ntype = "4"):
    if ntype == "4":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main():

    parser = OptionParser()
    parser.add_option("-g", "--grid",
                      help = "Path to ascii binary occupancy grid (nonzero = obstacle).",
                      default = "test/env_4.txt")
    parser.add_option(      "--sx",
                      help = "Start location column.",
                      default = 0)
    parser.add_option(      "--sy",
                      help = "Start location row.",
                      default = 0)
    parser.add_option(      "--dx",
                      help = "Goal location column.",
                      default = 29)
    parser.add_option(      "--dy",
                      help = "Goal location row.",
                      default = 29)
    parser.add_option("-n", "--nhood_type",
                      help = "Neighborhood type (4, 8, or 16).",
                      default = 4)
    parser.add_option(      "--solver",
                      help = "Path finding algorithm (A*, dijkstra).",
                      default = "dijkstra")
    parser.add_option(      "--trace",
                      help = "Path to save map of solver's history. Shows which cells were evaluated.",
                      default = None)

    (options, args) = parser.parse_args()

    gridFile = options.grid
    start = (int(options.sy), int(options.sx))
    goal = (int(options.dy), int(options.dx))
    solver = options.solver.lower()
    ntype = int(options.nhood_type)
    if ntype != 4 and ntype != 8 and ntype != 16:
        print("Invalid neighborhood type {}".format(ntype))
        exit(-1)
    trace = False
    traceOutFile = options.trace
    if options.trace is not None:
        trace = True

    # Load occupancy grid from comma-delimited ASCII file
    grid = np.loadtxt(gridFile, delimiter = ",")
    # Binarize grid
    grid[grid > 0] = 1
    rows, cols = grid.shape

    print("Using {r}x{c} grid {g}".format(r = rows, c = cols, g = gridFile))
    print("Using {n}-way neighborhood".format(n = ntype))
    print("  Start:", start)
    print("  End:", goal)

    # Solve
    t0 = time.time()

    if solver == "a*":
        path, traceGrid = astar(grid, start, goal, ntype = ntype, trace = trace)
    else:  # Default to dijkstra
        solver = "dijkstra"
        path, traceGrid = dijkstra(grid, start, goal, ntype = ntype, trace = trace)
    t1 = time.time()
    print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))


    # Visualize
    # Trace solver
    if trace:
        plt.imshow(traceGrid, cmap = "Greys")
        plt.show()

if __name__ == '__main__':
    main()


