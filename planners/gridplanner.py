# Solves a raster occupancy grid with classic search algorithm

import numpy as np
import heapq
import time
from optparse import OptionParser
import matplotlib.pyplot as plt
from math import acos, cos, sin, ceil, floor
from bresenham import bresenham
from haversine import haversine

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

def calcWork(v, w, currentsGrid_u, currentsGrid_v, targetSpeed_mps, timeIn = 0, interval = 3600, pixelsize_m = 1):
    '''
    This function calculates the work applied by a vehicle
    between two points, given rasters with u, v force components

    v: start point (row, col) in force rasters
    w: end point (row, col) in force rasters
    currentsGrid_u: 3D Raster of forces u components.
        [time index, row, column] = u
    currentsGrid_v: 3D Raster of forces v components.
        [time index, row, column] = v
    '''

    elapsed = float(timeIn)
    (index, rem) = divmod(elapsed, interval)
    index = floor(index)

    if v != w:
        # Heading
        vecs = (w[1] - v[1], w[0] - v[0])
        dotprod = vecs[0] * 1 + vecs[1] * 0
        maga = pow(vecs[0] * vecs[0] + vecs[1] * vecs[1], 0.5)
        heading_rad = 0.0
        if (maga != 0):
            costheta = dotprod / maga
            heading_rad = acos(costheta)

        # Work
        work = 0
        b = list(bresenham(v[0], v[1], w[0], w[1]))

        for p in b[1:]: # Skip first pixel -> already there!
            xA = targetSpeed_mps * cos(heading_rad)
            yA = targetSpeed_mps * sin(heading_rad)

            #xB = currentsGrid_u[0, p[0], p[1]]
            #yB = currentsGrid_v[0, p[0], p[1]]

            # Interpolate between nearest time's currents
            if currentsGrid_u.shape[0] > 1:
                cmag = currentsGrid_u[index, p[0] - 1, p[1] - 1] * (rem / interval) + \
                    currentsGrid_u[index + 1, p[0] - 1, p[1] - 1] * (1 - (rem / interval))
                cdir = currentsGrid_v[index, p[0] - 1, p[1] - 1] * (rem / interval) + \
                    currentsGrid_v[index + 1, p[0] - 1, p[1] - 1] * (1 - (rem / interval))
            else:
                # Static currents -> can't interpolate in time
                cmag = currentsGrid_u[0, p[0] - 1, p[1] - 1]
                cdir = currentsGrid_v[0, p[0] - 1, p[1] - 1]

            xB = cmag * cos(cdir)
            yB = cmag * sin(cdir)

            dV = (xB - xA, yB - yA)

            magaDV = pow(dV[0] * dV[0] + dV[1] * dV[1], 0.5)
            dirDV = 0.0
            dotprod = dV[0] * 1 + dV[1] * 0
            if magaDV != 0:
                costheta = dotprod / magaDV
                dirDV = acos(costheta)

            work += magaDV * pixelsize_m

            # Update time
            elapsed += pixelsize_m / targetSpeed_mps
            (index, rem) = divmod(elapsed, interval)
            index = floor(index)

    else:
        work = 0

    return work


def solve(grid, start, goal, solver = 0, ntype = 4, trace = False,
             currentsGrid_u = None, currentsGrid_v = None, targetSpeed_mps = 100, timeOffset = 0, pixelsize_m = 1):

    solvers = ["dijkstra", "astar"]
    if solver >= len(solvers):
        print("Invalid solver ID {}".format(solver_id))
        exit(-1)

    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    time_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    time_so_far[start] = timeOffset

    tgrid = None
    if trace:
        tgrid = grid.copy()

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

            dist = pow((pow(current[0] - next[0], 2) + pow(current[1] - next[1], 2)), 0.5) * pixelsize_m
            # Cost
            if currentsGrid_u is None:
                # Distance cost
                new_cost = cost_so_far[current] + dist
            else:
                # Energy cost
                new_cost = cost_so_far[current] + calcWork(current, next, currentsGrid_u, currentsGrid_v,
                                    targetSpeed_mps, timeIn = time_so_far[current], pixelsize_m = pixelsize_m)

            update = False
            if next not in cost_so_far:
                update = True
            else:
                if new_cost < cost_so_far[next]:
                    update = True
            if update:
                cost_so_far[next] = new_cost
                if solver == 1: # A*
                    priority = new_cost + pow((pow(next[0] - goal[0], 2) + pow(next[1] - goal[1], 2)), 0.5)
                else:           # Default: dijkstra
                    priority = new_cost

                frontier.put(next, priority)
                came_from[next] = current
                time_so_far[next] = time_so_far[current] +  (dist / targetSpeed_mps)

    # Reconstruct path
    current = goal
    path = []

    while current != start:
        if trace:
            tgrid[current] = 1.5
        path.append(current)
        current = came_from[current]
    if trace:
        tgrid[start] = 0.5
    path.append(start)
    path.reverse()

    return path, tgrid, cost_so_far[goal], time_so_far[goal]


def selectNeighborRules(ntype = "4"):
    if ntype == "4":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main():

    parser = OptionParser()
    parser.add_option("-g", "--grid",
                      help = "Path to ascii binary occupancy grid (nonzero = obstacle).",
                      default = "test/env_4.txt")
    parser.add_option("-u", "--currents_mag",
                      help = "Path to grid with magnitude of water velocity.",
                      default = None)
    parser.add_option("-v", "--currents_dir",
                      help = "Path to grid with direction of water velocity.",
                      default = None)
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
    currentsGridFile_u = options.currents_mag
    currentsGridFile_v = options.currents_dir
    start = (int(options.sy), int(options.sx))
    goal = (int(options.dy), int(options.dx))
    solver = options.solver.lower()
    ntype = int(options.nhood_type)
    if ntype not in [4, 8, 16]:
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

    usingCurrents = False
    # Load water currents grids from
    if currentsGridFile_u is not None:
        currentsGrid_u = np.loadtxt(currentsGridFile_u, delimiter = ",")
        currentsGrid_v = np.loadtxt(currentsGridFile_v, delimiter = ",")
        usingCurrents = True

        if len(currentsGrid_u.shape) == 2:
            cu = np.zeros((1, currentsGrid_u.shape[0], currentsGrid_u.shape[1]))
            cu[0] = currentsGrid_u.copy()
            currentsGrid_u = cu.copy()
            cv = np.zeros((1, currentsGrid_v.shape[0], currentsGrid_v.shape[1]))
            cv[0] = currentsGrid_v.copy()
            currentsGrid_v = cv

        print("Incorporating forces into energy cost:")
        print("  magnitude: {}".format(currentsGridFile_u))
        print("  direction: {}".format(currentsGridFile_v))


    if solver == "a*" or solver == "astar":
        solver_id = 1
    else:
        # Default solver to dijkstra
        solver = "dijkstra"
        solver_id = 0

    # Solve
    t0 = time.time()
    if usingCurrents:
        if solver == "a*":
            path, traceGrid, T, C  = solve(grid, start, goal, solver = solver_id, ntype = ntype, trace = trace,
                                    currentsGrid_u = currentsGrid_u, currentsGrid_v = currentsGrid_v)
    else:
        path, traceGrid, T, C = solve(grid, start, goal, ntype = ntype, trace = trace)
    t1 = time.time()
    print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))
    print("Cost: {c}".format(c = C))

    # Visualize
    # Trace solver
    if trace:
        plt.imshow(traceGrid, cmap = "Greys")
        plt.show()

if __name__ == '__main__':
    main()


