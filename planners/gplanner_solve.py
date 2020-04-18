#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
from heapq import heapify, heappush, heappop
from osgeo import gdal
from math import acos, cos, sin, ceil
from bresenham import bresenham
# Conch modules
import rasterSetInterface as rsi
import gridUtils

try:
    dict.iteritems
except AttributeError:
    def iteritems(d):
        return iter(d.items())

def astar(graph, origin, destination,
        regionTransform = None, currents_u = None, currents_v = None, currentsTransform = None):
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
    cameFrom[origin] = None
    costSoFar[origin] = 0

    while len(frontier) > 0:
        current = frontier.pop_smallest()
        if current == destination:
            break
        edges = graph[current]
        for w in edges:
            distance = pow((pow(current[0] - w[0], 2) + pow(current[1] - w[1], 2)), 0.5)
            new_cost = costSoFar[current] + distance
            if w not in costSoFar or new_cost < costSoFar[w]:
                costSoFar[w] = new_cost
                distance = pow((pow(w[0] - destination[0], 2) + pow(w[1] - destination[1], 2)), 0.5)
                priority = new_cost + distance
                frontier[w] = priority
                cameFrom[w] = current
    return (frontier, cameFrom)


def dijkstra(graph, origin, destination, 
        regionTransform = None, currents_u = None, currents_v = None, currentsTransform = None):
    D = {}
    P = {}
    Q = priority_dict()
    Q[origin] = 0

    for v in Q:
        D[v] = Q[v]
        if v == destination: break
                
        # Need latlon IF considering water currents
        if currents_u is not None:
            v_latlon = gridUtils.grid2world(v[0], v[1], regionTransform, regionExtent["rows"])
            v_currents = gridUtils.world2grid(v_latlon[0], v_latlon[1], 
                    currentsTransform, currentsExtent_u["rows"])

        edges = graph[v]
        #if add_to_visgraph != None and len(add_to_visgraph[v]) > 0:
        #    edges = add_to_visgraph[v] | graph[v]
        for w in edges:
            
            # No currents --> cost = distance
            if currents_u is None:
                cost = D[v] + pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)
            
            # Currents --> cost = energy expended
            else:
                # Find corresponding coordinates in currents rasters
                w_latlon = gridUtils.grid2world(w[0], w[1], regionTransform, regionExtent["rows"])
                w_currents = gridUtils.world2grid(w_latlon[0], w_latlon[1], 
                        currentsTransform, currentsExtent_u["rows"])

                # Skip if latlons match... appropriate?
                if v_currents != w_currents:
                    # Heading
                    vecs = (w[1] - v[1], w[0] - v[0])
                    dotprod = vecs[0] * 1 + vecs[1] * 0
                    maga = pow(vecs[0] * vecs[0] + vecs[1] * vecs[1], 0.5)
                    heading_rad = 0.0
                    if (maga != 0):
                        costheta = dotprod / maga
                        heading_rad = acos(costheta)

                    #print(v)
                    #print(w)
                    #print(vecs)
                    #print (heading_rad)

                    # Distance
                    distance = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)

                    # Work
                    targetSpeed = 100 # units?

                    work = 0
                    b = list(bresenham(v_currents[0], v_currents[1], w_currents[0], w_currents[1]))
                    pixeldist = distance / (len(b) -1)

                    for p in b[1:]: # Skip first pixel -> already there!
                        xA = targetSpeed * cos(heading_rad)
                        yA = targetSpeed * sin(heading_rad)
                        xB = currents_u.GetRasterBand(1).ReadAsArray()[p[0], p[1]]
                        yB = currents_v.GetRasterBand(1).ReadAsArray()[p[0], p[1]]
                        
                        dV = (xB - xA, yB - yA)

                        #print(v, w, heading_rad)
                        #print(xA, yA, xB, yB)
                        #print(dV)

                        magaDV = pow(dV[0] * dV[0] + dV[1] * dV[1], 0.5)
                        dirDV = 0.0
                        dotprod = dV[0] * 1 + dV[1] * 0
                        if magaDV != 0:
                            costheta = dotprod / magaDV
                            dirDV = acos(costheta)

                        work += magaDV * pixeldist

                        #print(magaDV, dirDV)
                else:
                    work = 0

                cost = D[v] + work

                #print(v, w, v_currents, w_currents)
                #print(heading_rad, distance)

            if w in D:
                if cost < D[w]:
                    raise ValueError
            elif w not in Q or cost < Q[w]:
                Q[w] = cost
                P[w] = v
    return (D, P)

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
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_currents_u.tif")
parser.add_option("-v", "--currents_v",
        help = "Path to raster containing v components of water velocity.",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_currents_v.tif")
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
solver = options.solver

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
    currentsTransform_u = currentsData_u.GetGeoTransform()
    # Load v components
    currentsData_v = gdal.Open(currentsRasterFile_v)
    currentsExtent_v = rsi.getGridExtent(currentsData_v)
    currentsTransform_v = currentsData_v.GetGeoTransform()

    # Sanity check that the current u, v rasters match
    if currentsExtent_u["rows"] != currentsExtent_v["rows"] or \
            currentsExtent_u["cols"] != currentsExtent_v["cols"]:
        print("[-] Spatial extent mismatch between currents u, v")
        exit(-1)
    if currentsData_u.RasterCount != currentsData_v.RasterCount:
        print("[-] Band count mismatch between currents u, v")
        exit(-1)

    #usingCurrents = True
    print("Incorporating forces into energy cost:")
    print("  u: {}".format(currentsRasterFile_u))
    print("  v: {}".format(currentsRasterFile_v))

# Load graph
graph = pickle.load(open(graphFile, 'rb'))

# Convert latlon -> rowcol
start = gridUtils.world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
end = gridUtils.world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

# solve
if usingCurrents:   # Solve with current forces --> energy cost
    if solver == "A*":
        D, P = astar(graph, start, end, 
                regionTransform, currentsData_u, currentsData_v, currentsTransform_u)
    else:  # Default to dijkstra
        D, P = dijkstra(graph, start, end, 
                regionTransform, currentsData_u, currentsData_v, currentsTransform_u)
else:   # Default to distance-based    
    if solver == "A*":
        D, P = astar(graph, start, end)
    else:  # Default to dijkstra
        D, P = dijkstra(graph, start, end)

path = []
while 1:
    path.append(end)
    if end == start: break
    end = P[end]
path.reverse()
numWaypoints = len(path)

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
