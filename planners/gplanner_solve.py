#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
from heapq import heapify, heappush, heappop
from osgeo import gdal
# Conch modules
import rasterSetInterface as rsi

try:
    dict.iteritems
except AttributeError:
    def iteritems(d):
        return iter(d.items())

def dijkstra(graph, origin, destination):
    D = {}
    P = {}
    Q = priority_dict()
    Q[origin] = 0

    for v in Q:
        D[v] = Q[v]
        if v == destination: break

        edges = graph[v]
        #if add_to_visgraph != None and len(add_to_visgraph[v]) > 0:
        #    edges = add_to_visgraph[v] | graph[v]
        for w in edges:
            elength = D[v] + pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)
            if w in D:
                if elength < D[w]:
                    raise ValueError
            elif w not in Q or elength < Q[w]:
                Q[w] = elength
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
startPoint = (-70.92, 42.29)
endPoint = (-70.97, 42.30)
# Should use raster to transform to row, col
start = (100, 200)
end = (50, 50)

regionRasterFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif"
graphFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_uniform.pickle"


# Load raster
regionData = gdal.Open(regionRasterFile)
regionExtent = rsi.getGridExtent(regionData)
grid = regionData.GetRasterBand(1).ReadAsArray()

# Load graph
graph = pickle.load(open(graphFile, 'rb'))

# solve
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
plt.show()
