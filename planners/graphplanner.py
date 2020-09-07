#!/usr/bin/python3
# Solves a graph with classic search algorithms

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
import heapq
from osgeo import gdal
from math import acos, cos, sin, ceil, floor, atan2
from bresenham import bresenham
import osgeo.gdalnumeric as gdn
from haversine import haversine
import pyvisgraph as vg
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
    #
    # data: gdal object
    cols = data.RasterXSize
    rows = data.RasterYSize
    transform = data.GetGeoTransform()
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols }

try:
    dict.iteritems
except AttributeError:
    def iteritems(d):
        return iter(d.items())

def raster2array(raster, dim_ordering="channels_last", dtype='float32'):
    '''
    Modified from: https://gis.stackexchange.com/a/283207
    '''
    bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
    return arr

def calcWork(v, w, currentsGrid_u, currentsGrid_v, targetSpeed_mps, geotransform,
             timeIn = 0, interval = 3600, pixelsize_m = 1, distMeas = "haversine"):
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
    index = min(floor(index), currentsGrid_u.shape[0] - 2)

    if v != w:
        # Heading
        vecs = (w[1] - v[1], w[0] - v[0])
        dotprod = vecs[0] * 1 + vecs[1] * 0
        maga = pow(vecs[0] * vecs[0] + vecs[1] * vecs[1], 0.5)
        heading_rad = 0.0
        if (maga != 0):
            costheta = dotprod / maga
            heading_rad = acos(costheta)

        # Distance
        rows = currentsGrid_u.shape[0]
        if distMeas == "haversine":
            v_latlon = grid2world(v[0], v[1], geotransform, rows)
            w_latlon = grid2world(w[0], w[1], geotransform, rows)
            hdist = haversine(v_latlon, w_latlon) * 1000
        elif distMeas == "euclidean-scaled":
            hdist = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5) * pixelsize_m
        elif distMeas == "euclidean":
            hdist = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)

        # Work
        work = 0
        b = list(bresenham(v[0], v[1], w[0], w[1]))
        hdist_ = hdist / len(b)

        for p in b[:]:
            xA = targetSpeed_mps * cos(heading_rad)
            yA = targetSpeed_mps * sin(heading_rad)

            # Interpolate between nearest time's currents
            if currentsGrid_u.shape[0] > 1:
                ua_ = currentsGrid_u[index, p[0], p[1]] * cos(currentsGrid_v[index, p[0], p[1]])
                va_ = currentsGrid_u[index, p[0], p[1]] * sin(currentsGrid_v[index, p[0], p[1]])
                ub_ = currentsGrid_u[index + 1, p[0], p[1]] * cos(currentsGrid_v[index + 1, p[0], p[1]])
                vb_ = currentsGrid_u[index + 1, p[0], p[1]] * sin(currentsGrid_v[index + 1, p[0], p[1]])

                u_ = ua_ * (1 - (rem / interval)) + ub_ * ((rem / interval))
                v_ = va_ * (1 - (rem / interval)) + vb_ * ((rem / interval))
                uv_ = np.array([u_, v_])

                cmag = np.sqrt(uv_.dot(uv_))
                cdir =  atan2(v_, u_)

                # Convert to knots for sanity check
                #knots = cmag * 1.94384
                #print(knots)

            else:
                # Static currents -> can't interpolate in time
                cmag = currentsGrid_u[0, p[0], p[1]]
                cdir = currentsGrid_v[0, p[0], p[1]]

            xB = cmag * cos(cdir)
            yB = cmag * sin(cdir)

            # Calculate applied force,
            # given desired and environment forces
            dV = (xB - xA, yB - yA)
            magaDV = pow(dV[0] * dV[0] + dV[1] * dV[1], 0.5)
            dirDV = 0.0
            dotprod = dV[0] * 1 + dV[1] * 0
            if magaDV != 0:
                costheta = dotprod / magaDV
                dirDV = acos(costheta)

            work += magaDV * hdist_

            # Update time
            elapsed += hdist_ / targetSpeed_mps
            (index, rem) = divmod(elapsed, interval)
            index = min(floor(index), currentsGrid_u.shape[0] - 2)
    else:
        work = 0
    return work, elapsed

def solve(graph, start, goal, grid, geotransform, solver = 0,
        currentsGrid_u = None, currentsGrid_v = None, currentsExtent = None,
        targetSpeed_mps = 1, timeOffset = 0, pixelsize_m = 1, distMeas = "haversine", trace = False):

    solvers = ["dijkstra", "astar"]
    if solver >= len(solvers):
        print("Invalid solver ID {}".format(solver_id))
        exit(-1)

    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    cameFrom = {}
    costSoFar = {}
    timeSoFar = {}
    cameFrom[start] = None
    costSoFar[start] = 0
    timeSoFar[start] = timeOffset

    tgrid = None
    if trace:
        tgrid = grid.copy()

    if distMeas == "haversine":
        # Lat, lon of goal
        g_latlon = grid2world(goal[0], goal[1], geotransform, rows)

    while not frontier.empty():
        current = frontier.get()

        if distMeas == "haversine":
            c_latlon = grid2world(current[0], current[1], geotransform, rows)

        if trace:
            tgrid[current] = 0.25

        # Check if goal
        if current == goal:
            break

        try:
            edges = graph[current]
        except:
            continue

        for next in edges:
            if distMeas == "haversine":
                n_latlon = grid2world(next[0], next[1], geotransform, rows)
                dist = haversine(c_latlon, n_latlon) * 1000
            elif distMeas == "euclidean-scaled":
                dist = pow((pow(current[0] - next[0], 2) + pow(current[1] - next[1], 2)), 0.5) * pixelsize_m
            elif distMeas == "euclidean":
                dist = pow((pow(current[0] - next[0], 2) + pow(current[1] - next[1], 2)), 0.5)


            # Cost
            if currentsGrid_u is None:
                et = timeSoFar[current] + (dist / targetSpeed_mps)
                # Distance cost
                new_cost = costSoFar[current] + dist
            else:
                # Energy cost
                work, et = calcWork(current, next, currentsGrid_u, currentsGrid_v, targetSpeed_mps,
                                    geotransform = geotransform, timeIn = timeSoFar[current], pixelsize_m = pixelsize_m, distMeas = distMeas)
                new_cost = costSoFar[current] + work

            update = False
            if next not in costSoFar:
                update = True
            else:
                if new_cost < costSoFar[next]:
                    update = True

            if update:
                costSoFar[next] = new_cost
                if solver == 1: # A*
                    if distMeas == "haversine":
                        priority = new_cost + (haversine(n_latlon, g_latlon) * 1000)
                    elif distMeas == "euclidean-scaled":
                        priority = new_cost + pow((pow(next[0] - goal[0], 2) + pow(next[1] - goal[1], 2)), 0.5) * pixelsize_m
                    elif distMeas == "euclidean":
                        priority = new_cost + pow((pow(next[0] - goal[0], 2) + pow(next[1] - goal[1], 2)), 0.5)
                else:           # Default: dijkstra
                    priority = new_cost

                frontier.put(next, priority)
                cameFrom[next] = current
                timeSoFar[next] = et

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        if trace:
            tgrid[current] = 1.5
        path.append(current)
        current = cameFrom[current]
    if trace:
        tgrid[start] = 0.5
    path.append(start)
    path.reverse()

    return path, tgrid, costSoFar[goal], timeSoFar[goal]

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

def main():
    parser = OptionParser()
    parser.add_option("-r", "--region",
            help = "Path to raster containing binary occupancy region (0 -> free space),",
            default = "test/inputs/full.tif")
    parser.add_option("-g", "--graph",
            help = "Path to uniform graph.",
            default = "test/outputs/uni-4.pickle")
    parser.add_option("-m", "--map",
            help = "Path to save solution path map.",
            default = "graphplanner_test.png")
    parser.add_option("-p", "--path",
            help = "Path to save solution path.",
            default = "graphplanner_test.txt")
    parser.add_option("-u", "--currents_mag",
            help = "Path to raster with magnitude of water velocity.",
            default = None)
    parser.add_option("-v", "--currents_dir",
            help = "Path to raster with direction of water velocity.",
            default = None)
    parser.add_option("--sx",
            help = "Start longitude.",
            default = -70.99428)
    parser.add_option("--sy",
            help = "Start latitude.",
            default = 42.32343)
    parser.add_option("--dx",
            help = "Destination longitude.",
            default = -70.88737)
    parser.add_option("--dy",
            help = "Destination latitude.",
            default = 42.33600)
    parser.add_option("--solver",
            help = "Path finding algorithm (A*, dijkstra).",
            default = "dijkstra")
    parser.add_option("--speed",
            help = "Target boat speed (meters/second).",
            default = 0.5)
    parser.add_option(      "--trace",
                      help = "Path to save map of solver's history. Shows which cells were evaluated.",
                      default = None)

    (options, args) = parser.parse_args()

    regionRasterFile = options.region
    graphFile = options.graph
    mapOutFile = options.map
    pathOutFile = options.path
    currentsRasterFile_u = options.currents_mag
    currentsRasterFile_v = options.currents_dir
    startPoint = (float(options.sy), float(options.sx))
    endPoint = (float(options.dy), float(options.dx))
    solver = options.solver.lower()
    targetSpeed_mps = float(options.speed)
    trace = False
    traceOutFile = options.trace
    if options.trace is not None:
        trace = True

    print("Using input region raster: {}".format(regionRasterFile))
    print("      input graph file: {}".format(graphFile))
    print("  Start:", startPoint)
    print("  End:", endPoint)
    print("  Speed: {} m/s".format(targetSpeed_mps))


    # Load raster
    regionData = gdal.Open(regionRasterFile)
    regionExtent = getGridExtent(regionData)
    regionTransform = regionData.GetGeoTransform()
    grid = regionData.GetRasterBand(1).ReadAsArray()

    # Read currents rasters
    elapsedTime = 0
    bandIndex = 0
    usingCurrents = False
    if currentsRasterFile_u is not None and currentsRasterFile_v is not None:
        bandIndex = 1

        # Load u components
        currentsData_u = gdal.Open(currentsRasterFile_u)
        currentsExtent_u = getGridExtent(currentsData_u)
        currentsTransform_u = currentsData_u.GetGeoTransform()
        currentsGrid_u = raster2array(currentsData_u)

        # Load v components
        currentsData_v = gdal.Open(currentsRasterFile_v)
        currentsExtent_v = getGridExtent(currentsData_v)
        currentsTransform_v = currentsData_v.GetGeoTransform()
        currentsGrid_v = raster2array(currentsData_v)

        # Sanity check that the current u, v rasters match
        if currentsExtent_u["rows"] != currentsExtent_v["rows"] or \
                currentsExtent_u["cols"] != currentsExtent_v["cols"]:
            print("[-] Spatial extent mismatch between currents u, v")
            exit(-1)
        if currentsData_u.RasterCount != currentsData_v.RasterCount:
            print("[-] Band count mismatch between currents u, v")
            exit(-1)

        usingCurrents = True
        print("Incorporating forces into energy cost:")
        print("  u: {}".format(currentsRasterFile_u))
        print("  v: {}".format(currentsRasterFile_v))

    # Default to normal dictionary-formatted graph
    graph = pickle.load(open(graphFile, 'rb'))

    # Convert latlon -> rowcol
    start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
    end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

    # Calculate pixel distance
    s = grid2world(0, 0, regionTransform, regionExtent["rows"])
    e = grid2world(0, regionExtent["cols"] - 1, regionTransform, regionExtent["rows"])
    dist_m = haversine(s, e) * 100
    dist = pow((pow(0, 2) + pow(regionExtent["cols"], 2)), 0.5)
    pixelsize_m = dist_m / dist

    if solver == "a*" or solver == "astar":
        solver_id = 1
    else:
        # Default solver to dijkstra
        solver = "dijkstra"
        solver_id = 0

    # Solve
    t0 = time.time()
    if usingCurrents:   # Solve with current forces --> energy cost
            path, traceGrid, C, T = solve(graph, start, end, grid, regionTransform, solver_id,
                    currentsGrid_u, currentsGrid_v, currentsExtent_u,
                    targetSpeed_mps, pixelsize_m = pixelsize_m, trace = trace)
    else:   # Default to distance-based
            path, traceGrid, C, T = solve(graph, start, end, grid, regionTransform, solver_id, pixelsize_m = pixelsize_m, trace = trace)
    t1 = time.time()
    print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))

    # Stat path
    numWaypoints = len(path)
    path_distance = 0
    prev_point = path[0]
    for point in path[1:]:
        prev_latlon = grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
        point_latlon = grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
        path_distance += haversine(prev_latlon, point_latlon)
        prev_point = point
    path_duration = (path_distance * 1000 / targetSpeed_mps) / 60

    print(T / 60)
    print('    Distance: {:.4f} km'.format(path_distance))
    print('    Duration: {:.4f} min'.format(path_duration))
    if usingCurrents:
        print('    Cost: {:.4f}'.format(C))

    # Visualize
    # Plot trace of solver
    if trace:
        plt.imshow(traceGrid, cmap = "Greys")
        plt.savefig(traceOutFile)

    # Plot solution path
    lats = np.zeros(numWaypoints)
    lons = np.zeros(numWaypoints)
    for i in range(numWaypoints):
        lats[i] = path[i][0]
        lons[i] = path[i][1]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_ylim(0, regionExtent["rows"])
    ax.set_xlim(0, regionExtent["cols"])
    ax.set_facecolor('xkcd:lightblue')
    # Plot raster
    plt.imshow(grid, cmap=plt.get_cmap('gray'))
    plt.plot(lons, lats)

    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig(mapOutFile)

    np.savetxt(pathOutFile, path, delimiter=',')


if __name__ == '__main__':
    main()
