#!/usr/bin/python3

import pyvisgraph as vg
import fiona
import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape
import matplotlib.pyplot as plt
import geopandas as gp
from optparse import OptionParser
from haversine import haversine

def costPath(path):
    pathDistance = 0.0
    prevPoint = path[0]
    for point in path[1:]:
        pathDistance += haversine((prevPoint.y, prevPoint.x), (point.y, point.x))
        prevPoint = point
    return pathDistance

###########
# Options #
###########
##startPoint = (-70.908544, 42.311243)
##endPoint = (-70.910486, 42.297507)
startPoint = (-70.92, 42.29) 
endPoint = (-70.97, 42.30)

graphFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.graph"
##shapeFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.shp"
shapeFile = "/home/ekrell/Downloads/ADVGEO_DL/tiny2.shp"
mapOutFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_path.png"

print("Using input graph: {}".format(graphFile))
print("Solving ({} N, {} W) --> ({} N, {} W)".format(startPoint[1], startPoint[0], endPoint[1], endPoint[0]))

#########
# Setup #
#########

# Load the shapefile
shapes = gp.read_file(shapeFile)

# Load the visibility graph
print("Begin loading visibility graph")
graph = vg.VisGraph()
graph.load(graphFile)
print("Done loading visibility graph")

#########
# Solve #
#########

start = vg.Point(startPoint[0], startPoint[1])
end = vg.Point(endPoint[0], endPoint[1])

# Solve with Dijkstra
print("Begin solving [dijkstra]")
solutionDijkstra = graph.shortest_path(start, end)
print("End solving")

# Print solution
shortestDistance = costPath(solutionDijkstra)
print('[Dijkstra] shortest path distance: {} km'.format(shortestDistance))
print('           number of waypoints: {}'.format(len(solutionDijkstra)))

#############
# Visualize #
#############
numWaypoints = len(solutionDijkstra)
lats = np.zeros(numWaypoints)
lons = np.zeros(numWaypoints)
for i in range(numWaypoints):
    lats[i] = solutionDijkstra[i].y
    lons[i] = solutionDijkstra[i].x

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
for index, row in shapes.iterrows():
    plt.fill(*row["geometry"].exterior.xy, facecolor = 'khaki', edgecolor = 'black', linewidth = 0.5)
ax.set_facecolor('xkcd:lightblue')
plt.scatter(lons, lats, s = 10, c = 'red')
plt.plot(lons, lats)
plt.scatter(lons[0], lats[0], s = 12, c = 'purple', marker = 'o')
plt.scatter(lons[numWaypoints - 1], lats[numWaypoints - 1], s = 12, c = 'white', marker = '*')

#edges = graph.graph.get_edges()
edges = graph.visgraph
for e in list(edges):
    plt.scatter([e.p1.x, e.p2.x], [e.p1.y, e.p2.y], s = 9, color = 'green')
    plt.plot([e.p1.x, e.p2.x], [e.p1.y, e.p2.y], color = 'green', linestyle = 'dashed')

########
# Save #
########

# Save map with solution path
plt.savefig(mapOutFile)

