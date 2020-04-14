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
import shapefile
import time

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
##endPoint = (-70.910486, 43.297507)
startPoint = (-70.92, 42.29) 
endPoint = (-70.97, 42.30)

graphFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.graph"
shapeFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.shp"
mapOutFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_path_mini.png"

print("Using input graph: {}".format(graphFile))
print("Solving ({} N, {} W) --> ({} N, {} W)".format(startPoint[1], startPoint[0], endPoint[1], endPoint[0]))

#########
# Setup #
#########

# Load the shapefile
input_shapefile = shapefile.Reader(shapeFile)
shapes = input_shapefile.shapes()

# Find extent
minx = shapes[0].points[0][0]
maxx = minx
miny = shapes[0].points[0][1]
maxy = miny
for shape in shapes:
    for point in shape.points:
        if point[0] < minx:
            minx = point[0]
        elif point[0] > maxx:
            maxx = point[0]
        if point[1] < miny:
            miny = point[1]
        elif point[1] > maxy:
            maxy = point[1] 

# Load the visibility graph
print("Begin loading visibility graph")
graph = vg.VisGraph()
graph.load(graphFile)
print("Done loading visibility graph")

#########
# Solve #
#########
start = vg.Point(round((round(startPoint[0], 10) - minx) / (maxx - minx) * 100, 6), 
                 round((round(startPoint[1], 10) - miny) / (maxy - miny) * 100, 6))
end = vg.Point(round((round(endPoint[0], 10) - minx) / (maxx - minx) * 100, 6), 
               round((round(endPoint[1], 10) - miny) / (maxy - miny) * 100, 6))

# Solve
print("Begin solving")
startTime = time.time()
solutionDijkstra = graph.shortest_path(start, end, solver = "astar")
print("End solving in {} seconds".format(time.time() - startTime))

for i in range(len(solutionDijkstra)):
    solutionDijkstra[i].x = \
            (solutionDijkstra[i].x / 100) * (maxx - minx) + minx
    solutionDijkstra[i].y = \
            (solutionDijkstra[i].y / 100) * (maxy - miny) + miny

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

# Load the shapefile
shapes = gp.read_file(shapeFile)
# Plot solution path
fig, ax = plt.subplots(nrows=1, ncols=1)
for index, row in shapes.iterrows():
    plt.fill(*row["geometry"].exterior.xy, facecolor = 'khaki', edgecolor = 'black', linewidth = 0.5)
ax.set_facecolor('xkcd:lightblue')
plt.scatter(lons, lats, s = 10, c = 'red')
plt.plot(lons, lats)
plt.scatter(lons[0], lats[0], s = 12, c = 'purple', marker = 'o')
plt.scatter(lons[numWaypoints - 1], lats[numWaypoints - 1], s = 12, c = 'white', marker = '*')

########
# Save #
########

# Save map with solution path
plt.savefig(mapOutFile)
print("Saved solution path map to file: {}".format(mapOutFile))

