#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
import dill as pickle
import time

def getGridExtent (data):
    # Source: https://gis.stackexchange.com/a/104367
    #
    # data: gdal object
    cols = data.RasterXSize
    rows = data.RasterYSize
    transform = data.GetGeoTransform()
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols  }

def getNeighbors(i, m, n, env, ntype = 4):
    '''
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


###########
# Options #
###########
freecell = 0
occupied = 1

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "test/full.tif")
parser.add_option("-g", "--graph",
        help = "Path to save graph",
        default = "test/uni.pickle")
parser.add_option("-m", "--map",
        help = "Path to save graph map",
        default = "test/uni.png")
parser.add_option("-n", "--nhood_type",
        help = "Neighborhood type (4, 8, or 16).",
        default = 4)

(options, args) = parser.parse_args()

regionRasterFile = options.region
graphOutFile = options.graph
mapOutFile = options.map
ntype = int(options.nhood_type)
if ntype not in [4, 8, 16]:
    print("Invalid neighborhood type {}.".format(ntype))
    exit(-1)

print("Using input region raster: {}".format(regionRasterFile))
print("Using {n}-way neighborhood".format(n = ntype))

###################
# Region -> Graph #
###################

# Read region raster
regionData = gdal.Open(regionRasterFile)
regionExtent = getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

# Convert to graph
print("Begin building uniform graph")
t0 = time.time()
graph = {}
for row in range(regionExtent["rows"]):
    for col in range(regionExtent["cols"]):
        # If free space
        if grid[row][col] == freecell:
            # Add edges to unoccupied neighbors
            edges_ = getNeighbors((row, col), regionExtent["rows"], regionExtent["cols"], grid, ntype = ntype)
            graph[(row, col)] = edges_
t1 = time.time()
print("Done building uniform graph, {} seconds".format(t1 - t0))

# Graph stats
edgecount = 0
for p in list(graph.keys()):
    edgecount += len(graph[p])
print("Graph of {n} nodes, {e} edges".format(n = len(graph.keys()), e = edgecount))


# Save graph
pickle.dump(graph, open(graphOutFile, "wb"))
print("Saved uniform graph to file: {}".format(graphOutFile))

# Early exit because an attempt to plot may crash your computer; proceed with caution.
exit(0)

# Plot graph
print("Begin plotting uniform graph: {} nodes".format(len(graph)))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_facecolor('xkcd:lightblue')
plt.scatter(list(zip(*graph.keys()))[1], list(zip(*graph.keys()))[0], color = 'green', s = 0.025)
ax.set_ylim(ax.get_ylim()[::-1])
print("Done plotting uniform graph")
plt.savefig(mapOutFile)
print("Saved uniform graph map to file: {}".format(mapOutFile))

