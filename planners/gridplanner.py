# Solves a raster occupancy grid with classic search algorithm

import numpy as np
import queue
import time
from optparse import OptionParser

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
def dijkstra(grid, start, goal, ntype = 4):
    rows, cols = grid.shape
    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # Explore
    while not frontier.empty():
        current = frontier.get()
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
            new_cost = cost_so_far[current] + pow((pow(next[0] - current[0], 2) + pow(next[1] - current[1], 2)), 0.5)
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
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# A*
def astar(grid, start, goal, ntype = 4):
    rows, cols = grid.shape
    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # Explore
    while not frontier.empty():
        current = frontier.get()
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
            new_cost = cost_so_far[current] + pow((pow(next[0] - current[0], 2) + pow(next[1] - current[1], 2)), 0.5)
            update = False
            if next not in cost_so_far:
                update = True
            else:
                if new_cost < cost_so_far[next]:
                    update = True
            if update:
                cost_so_far[next] = new_cost
                priority = new_cost + pow((pow(next[0] - goal[0], 2) + pow(next[1] - goal[1], 2)), 0.5)
                frontier.put(next, priority)
                came_from[next] = current
    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def selectNeighborRules(nhoodType = "4"):
    if nhoodType == "4":
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
    parser.add_option(      "--nhood_type",
                      help = "Neighborhood type (4, 8, or 16).",
                      default = 4)
    parser.add_option(      "--solver",
                      help = "Path finding algorithm (A*, dijkstra).",
                      default = "dijkstra")

    (options, args) = parser.parse_args()

    gridFile = options.grid
    start = (int(options.sy), int(options.sx))
    goal = (int(options.dy), int(options.dx))
    solver = options.solver.lower()
    nhoodType = int(options.nhood_type)
    if nhoodType != 4 and nhoodType != 8 and nhoodType != 16:
        print("Invalid neighborhood type {}".format(nhoodType))
        exit(-1)

    # Load occupancy grid from comma-delimited ASCII file
    grid = np.loadtxt(gridFile, delimiter = ",")
    rows, cols = grid.shape

    print("Using {r}x{c} grid {g}".format(r = rows, c = cols, g = gridFile))
    print("Using {n}-way neighborhood".format(n = nhoodType))
    print("  Start:", start)
    print("  End:", goal)

    # Solve
    t0 = time.time()

    if solver == "a*":
        path = astar(grid, start, goal, nhoodType)
    else:  # Default to dijkstra
        solver = "dijkstra"
        path = dijkstra(grid, start, goal, nhoodType)

    t1 = time.time()
    print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))

if __name__ == '__main__':
    main()


