#!/usr/bin/python3
# Generates N shortest paths

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
from osgeo import gdal
import heapq
from math import acos, cos, sin, ceil, floor, atan2
import time

def world2grid (lat, lon, transform, nrow):
    row = int ((lat - transform[3]) / transform[5])
    col = int ((lon - transform[0]) / transform[1])
    return (row, col)

def grid2world (row, col, transform, nrow):
    lon = transform[1] * col + transform[2] * row + transform[0]
    lat = transform[4] * col + transform[5] * row + transform[3]
    return (lat, lon)

def getGridExtent (data):
    # Source: https://gis.stackexchange.com/a/104367
    # data: gdal object
    cols = data.RasterXSize
    rows = data.RasterYSize
    transform = data.GetGeoTransform()
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols }

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

def solve(graph, start, goal, grid, regionTransform):
    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    cameFrom = {}
    costSoFar = {}
    cameFrom[start] = None
    costSoFar[start] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        try:
            edges = graph[current]
        except:
            continue
        for next in edges:
            dist = pow((pow(current[0] - next[0], 2) + pow(current[1] - next[1], 2)), 0.5)
            new_cost = costSoFar[current] + dist
            update = False
            if next not in costSoFar:
                update = True
            else:
                if new_cost < costSoFar[next]:
                    update = True
            if update:
                costSoFar[next] = new_cost
                priority = new_cost + pow((pow(next[0] - goal[0], 2) + pow(next[1] - goal[1], 2)), 0.5)

                frontier.put(next, priority)
                cameFrom[next] = current

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = cameFrom[current]
    path.append(start)
    path.reverse()

    return path, costSoFar[goal]


def main():
    parser = OptionParser()
    parser.add_option("-n", "--n_paths",
                      help="Number of paths to generate",
                      default=10)
    parser.add_option("-r", "--region",
                      help="Path to raster containing binary occupancy region (0 -> free space)",
                      default="test/inputs/full.tif")
    parser.add_option("-g", "--graph",
                      help="Path to graph",
                      default="test/outputs/graphplanner_visgraph/visgraph_P1.pickle")
    parser.add_option("-p", "--paths",
                      help="Path to save solution paths",
                      default="getNpaths_test.txt")
    parser.add_option("-m", "--map",
                      help="Path to save solution map",
                      default="getNpaths_test.png")
    parser.add_option("--sx",
                      help="Start longitude.",
                      default=-70.99428)
    parser.add_option("--sy",
                      help="Start latitude.",
                      default=42.32343)
    parser.add_option("--dx",
                      help="Destination longitude.",
                      default=-70.88737)
    parser.add_option("--dy",
                      help="Destination latitude.",
                      default=42.33600)
    (options, args) = parser.parse_args()

    npaths = int(options.n_paths)
    regionRasterFile = options.region
    graphFile = options.graph
    mapOutFile = options.map
    pathsOutFile = options.paths
    startPoint = (float(options.sy), float(options.sx))
    endPoint = (float(options.dy), float(options.dx))

    print(" Generating N = {} paths".format(npaths))
    print(" Using input region raster: {}".format(regionRasterFile))
    print("       input graph file: {}".format(graphFile))
    print(" Start:", startPoint)
    print(" End:", endPoint)
    print("")

    # Load raster
    regionData = gdal.Open(regionRasterFile)
    regionExtent = getGridExtent(regionData)
    regionTransform = regionData.GetGeoTransform()
    grid = regionData.GetRasterBand(1).ReadAsArray()

    # Load graph
    graph = pickle.load(open(graphFile, 'rb'))

    # Convert latlon -> rowcol
    start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
    end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

    # Init plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_ylim(0, regionExtent["rows"])
    ax.set_xlim(0, regionExtent["cols"])
    ax.set_facecolor('xkcd:lightblue')
    plt.imshow(grid, cmap=plt.get_cmap('gray'))

    paths = []
    t0 = time.time()
    for n in range(npaths):
        # Solve
        path, cost = solve(graph, start, end, grid, regionTransform)
        # Plot path
        numWaypoints = len(path)
        lats = np.zeros(numWaypoints)
        lons = np.zeros(numWaypoints)
        for i in range(numWaypoints):
            lats[i] = path[i][0]
            lons[i] = path[i][1]
        plt.plot(lons, lats)
        # Delete nodes from graph
        del graph[path[int(numWaypoints / 2)]]
        # Add path to paths
        paths.append(path)
    t1 = time.time()
    print("Solved {n} paths in {t} seconds".format(n = npaths, t = t1 - t0))

    # Finish plot
    ax.set_ylim(ax.get_ylim()[::-1])
    # plt.show()

    # Save paths
    lines = []
    for path in paths:
        lines.append("".join([str(p) + ";" for p in path]).replace(" ","")[:-1] + "\n")
    outFile = open(pathsOutFile, "w")
    outFile.writelines(lines)
    outFile.close()

    # Save plot
    plt.savefig(mapOutFile)



if __name__ == "__main__":
    main()
