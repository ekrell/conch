#!/usr/bin/python4

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
from heapq import heapify, heappush, heappop
from osgeo import gdal
from math import acos, cos, sin, ceil, floor
from bresenham import bresenham
import osgeo.gdalnumeric as gdn
from haversine import haversine
import pyvisgraph as vg
# Conch modules
import rasterSetInterface as rsi
import gridUtils

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

def calcWork(v, w, currentsGrid_u, currentsGrid_v, targetSpeed_mps, timeIn = 0, interval = 3600):
    '''
    This function calculates the work applied by a vehicle
    between two points, given rasters with u, v force components
    The rasters MUST match: same extent, resolution, pixels are same latlon
    
    v: start point (row, col) in force rasters
    w: end point (row, col) in force rasters
    currentsGrid_u: 3D Raster of forces u components.
        [time index, row, column] = u
    currentsGrid_v: 3D Raster of forces v components.
        [time index, row, column] = v
    '''

    (index, rem) = divmod(timeIn, interval)
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

       # Distance
       distance = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)

       # Work
       work = 0
       b = list(bresenham(v[0], v[1], w[0], w[1]))
       pixeldist = distance / (len(b) -1)

       for p in b[1:]: # Skip first pixel -> already there!
            xA = targetSpeed_mps * cos(heading_rad)
            yA = targetSpeed_mps * sin(heading_rad)

           #xB = currentsGrid_u[0, p[0], p[1]]
           #yB = currentsGrid_v[0, p[0], p[1]]
            cmag = currentsGrid_u[index, p[0], p[1]] * (rem / interval) + \
                 currentsGrid_u[index + 1, p[0], p[1]] * (1 - (rem / interval))
            cdir = currentsGrid_v[index, p[0], p[1]] * (rem / interval) + \
                 currentsGrid_v[index + 1, p[0], p[1]] * (1 - (rem / interval))

            xB = cmag * cos(cdir)
            yB = cmag * sin(cdir)
                      
            dV = (xB - xA, yB - yA)

            magaDV = pow(dV[0] * dV[0] + dV[1] * dV[1], 0.5)
            dirDV = 0.0
            dotprod = dV[0] * 1 + dV[1] * 0
            if magaDV != 0:
                costheta = dotprod / magaDV
                dirDV = acos(costheta)

            work += magaDV * pixeldist
    else:
        work = 0

    return work


def astar(graph, origin, destination, regionTransform = None, 
        currentsGrid_u = None, currentsGrid_v = None, currentsTransform = None, currentsExtent = None, 
        targetSpeed_mps = 100, timeOffset = 0):
    """
    A* search algorithm, using Euclidean distance heuristic
    Note that this is a modified version of an
    A* implementation by Amit Patel.
    https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """

    frontier = priority_dict()
    frontier[origin] = 0
    cameFrom = {}
    costSoFar = {}
    timeSoFar = {}
    cameFrom[origin] = None
    costSoFar[origin] = 0
    timeSoFar[origin] = timeOffset

    while len(frontier) > 0:
        v = frontier.pop_smallest()
        if v == destination:
            break

        # Need latlon IF considering water currents
        v_latlon = gridUtils.grid2world(v[0], v[1], regionTransform, regionExtent["rows"])
        dest_latlon = gridUtils.grid2world(destination[0], destination[1], 
                regionTransform, regionExtent["rows"])

        if currentsTransform is not None:
            v_currents = gridUtils.world2grid(v_latlon[0], v_latlon[1], 
                    currentsTransform, currentsExtent["rows"])

        edges = graph[v]
        for w in edges:
            w_latlon = gridUtils.grid2world(w[0], w[1], regionTransform, regionExtent["rows"])
            # No currents --> cost = distance
            if currentsGrid_u is None:
                cost = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)
            # Currents --> cost = energy expended
            else:
                # Find corresponding coordinates in currents rasters
                w_currents = gridUtils.world2grid(w_latlon[0], w_latlon[1], 
                        currentsTransform, currentsExtent_u["rows"])
                # Calculate work applied by vehicle along edge
                cost = calcWork(v_currents, w_currents, currentsGrid_u, currentsGrid_v,
                        targetSpeed_mps, timeIn = timeSoFar[v])

            new_cost = costSoFar[v] + cost
            if w not in costSoFar or new_cost < costSoFar[w]:
                # Apply heuristic
                heuristic = pow((pow(w[0] - destination[0], 2) + pow(w[1] - destination[1], 2)), 0.5)
                heuristic = haversine(w_latlon, dest_latlon)
                costSoFar[w] = new_cost

                # Estimated duration to reach this point
                dist = haversine(v_latlon, w_latlon)
                timeSoFar[w] = timeSoFar[v] + (dist / targetSpeed_mps)
                
                priority = new_cost + dist
                frontier[w] = priority
                cameFrom[w] = v
                
    return (frontier, cameFrom, costSoFar[v], timeSoFar[v])


def dijkstra(graph, origin, destination,  regionTransform, 
        currentsGrid_u = None, currentsGrid_v = None, currentsTransform = None, currentsExtent = None, 
        targetSpeed_mps = 100, timeOffset = 0):

    D = {}
    P = {}
    timeSoFar = {}
    Q = priority_dict()
    Q[origin] = 0
    timeSoFar[origin] = timeOffset

    dest_latlon = gridUtils.grid2world(destination[0], destination[1],
            regionTransform, regionExtent["rows"])

    for v in Q:
        D[v] = Q[v]
        if v == destination: break

        v_latlon = gridUtils.grid2world(v[0], v[1], regionTransform, regionExtent["rows"])
                
        # Need latlon IF considering water currents
        if currentsGrid_u is not None:
            v_currents = gridUtils.world2grid(v_latlon[0], v_latlon[1], 
                    currentsTransform, currentsExtent_u["rows"])

        edges = graph[v]
        for w in edges:
            w_latlon = gridUtils.grid2world(w[0], w[1], regionTransform, regionExtent["rows"])

            # No currents --> cost = distance
            if currentsGrid_u is None:
                dcost = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)
            # Currents --> cost = energy expended
            else:
                # Find corresponding coordinates in currents rasters
                w_currents = gridUtils.world2grid(w_latlon[0], w_latlon[1], 
                        currentsTransform, currentsExtent_u["rows"])
                dcost = calcWork(v_currents, w_currents, currentsGrid_u, currentsGrid_v, 
                        targetSpeed_mps, timeIn = timeSoFar[v])
            # Add up cost
            cost = D[v] + dcost

            if w in D:
                if cost < D[w]:
                    raise ValueError
            elif w not in Q or cost < Q[w]:
                Q[w] = cost
                P[w] = v

                timeSoFar[w] = timeSoFar[v] + haversine(v_latlon, w_latlon)


    return (D, P, Q[v], timeSoFar[v])

class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is that priorities
    of items can be efficiently updated (amortized O(1)) using code as
    'thedict[item] = new_priority.'

    Note that this is a modified version of
    https://gist.github.com/matteodellamico/4451520 where sorted_iter() has
    been replaced with the destructive sorted iterator __iter__ from
    https://gist.github.com/anonymous/4435950
    """
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in iteritems(self)]
        heapify(self._heap)

    def smallest(self):
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def __iter__(self):
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()


###########
# Options #
###########

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif")
parser.add_option("-g", "--graph",
        help = "Path to uniform graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_uniform.pickle")
parser.add_option("-m", "--map",
        help = "Path to save solution path map",        
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_graph_mini_uniform_path.png")
parser.add_option("-u", "--currents_u",
        help = "Path to raster containing u components of water velocity",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_currents_u_2.tif")
parser.add_option("-v", "--currents_v",
        help = "Path to raster containing v components of water velocity.",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_currents_v_2.tif")
parser.add_option("--sx", 
        help = "Start longitude.",
        default = -70.92)
parser.add_option("--sy", 
        help = "Start latitude.",
        default = 42.29)
parser.add_option("--dx", 
        help = "Destination longitude.",
        default = -70.96)
parser.add_option("--dy", 
        help = "Destination latitude.",
        default = 42.30)
parser.add_option("--solver",
        help = "Path finding algorithm (A*, dijkstra).",
        default = "dijkstra")

(options, args) = parser.parse_args()

regionRasterFile = options.region
graphFile = options.graph
mapOutFile = options.map
currentsRasterFile_u = options.currents_u
currentsRasterFile_v = options.currents_v
startPoint = (options.sy, options.sx)
endPoint = (options.dy, options.dx)
solver = options.solver.lower()
targetSpeed_mps = 0.5

# Load raster
regionData = gdal.Open(regionRasterFile)
regionExtent = rsi.getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

print("Using input region raster: {}".format(regionRasterFile))

# Read currents rasters
elapsedTime = 0
bandIndex = 0
usingCurrents = False
if currentsRasterFile_u is not None and currentsRasterFile_v is not None:
    bandIndex = 1

    # Load u components
    currentsData_u = gdal.Open(currentsRasterFile_u)
    currentsExtent_u = rsi.getGridExtent(currentsData_u)
    currentsTransform_u = currentsData_u.GetGeoTransform()
    currentsGrid_u = raster2array(currentsData_u)

    # Load v components
    currentsData_v = gdal.Open(currentsRasterFile_v)
    currentsExtent_v = rsi.getGridExtent(currentsData_v)
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
start = gridUtils.world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
end = gridUtils.world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

# solve
if usingCurrents:   # Solve with current forces --> energy cost
    if solver == "a*":
        D, P, C, T = astar(graph, start, end, regionTransform, 
                currentsGrid_u, currentsGrid_v, currentsTransform_u, currentsExtent_u,
                targetSpeed_mps)
    else:  # Default to dijkstra
        D, P, C, T = dijkstra(graph, start, end, 
                regionTransform, currentsGrid_u, currentsGrid_v, currentsTransform_u, currentsExtent_u,
                    targetSpeed_mps)
else:   # Default to distance-based    
    if solver == "a*":
        D, P, C, T = astar(graph, start, end, regionTransform)
    else:  # Default to dijkstra
        D, P, C, T = dijkstra(graph, start, end, regionTransform)

path = []
while 1:
    path.append(end)
    if end == start: break
    end = P[end]
path.reverse()
numWaypoints = len(path)

path_distance = 0
prev_point = path[0]
for point in path[1:]:
    prev_latlon = gridUtils.grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
    point_latlon = gridUtils.grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
    path_distance += haversine(prev_latlon, point_latlon)
    prev_point = point

if usingCurrents:
    print("Planning results (with currents):")
else:
    print("Planning results (no currents):")
print('    Distance: {:.4f} km'.format(path_distance))
print('    Duration: {:.4f} min'.format( (path_distance / (targetSpeed_mps / 100)) / 60))
print('    Cost: {:.4f}'.format(C))

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
plt.scatter(list(zip(*graph.keys()))[1], list(zip(*graph.keys()))[0], color = 'green', s = 0.025, alpha = 0.1)
plt.plot(lons, lats)
ax.set_ylim(ax.get_ylim()[::-1])
plt.savefig(mapOutFile)
