#!/usr/bin/python3

import pyvisgraph as vg
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import dill as pickle
from osgeo import gdal
from pyvisgraph.visible_vertices import visible_vertices, point_in_polygon
from pyvisgraph.visible_vertices import closest_point
from pyvisgraph.graph import Graph, Edge
# Conch modules
import rasterSetInterface as rsi
import gridUtils

###########
# Options #
###########
rangeWidth = 100

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif")
parser.add_option("-v", "--visgraph",
        help = "Path to visibility graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.graph")
parser.add_option("-e", "--evisgraph",
        help = "Path to save extended visibility graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_evg.graph")
parser.add_option("-m", "--map",
        help = "Path to save solution path map",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_graph_mini_evg.png")
parser.add_option("--ydiff", 
        help = "Skip distance between initial rows, in percent of raster height.",
        default = 5)
parser.add_option("--xdiff", 
        help = "Skip distance between initial columns, in percent of raster height.",
        default = 5)
parser.add_option("--radius", 
        help = "Radius for circle-check.",
        default = 16)
parser.add_option("--threshold", 
        help = "Threshold for circle-check.",
        default = 2)

(options, args) = parser.parse_args()

rasterFileIn = options.region
graphFileIn = options.visgraph
graphFileOut = options.evisgraph
mapFileOut = options.map
ydiff = float(options.ydiff)
xdiff = float(options.xdiff)
rad = float(options.radius)
threshold = int(options.threshold)

# Load raster
regionData = gdal.Open(rasterFileIn)
# Find extent of raster (search domain)
regionExtent = rsi.getGridExtent(regionData)

# Load visibility graph
print("Begin loading visibility graph")
graph = vg.VisGraph()
graph.load(graphFileIn)
print("Done loading visibility graph")

# Generate uniform test points
print("Begin extending visibility graph")
print("  (Radius = {:2}, Threshold = {}, Y offset = {:2}%, X offset = {:2}%)".format(rad, threshold, ydiff, xdiff))
xy = np.mgrid[0:rangeWidth:xdiff, 0:rangeWidth:ydiff].reshape(2, -1).T
npoints = len(xy)

tpoints = graph.visgraph.get_points()
gx = [p.x for p in tpoints]
gy = [p.y for p in tpoints]
tpoints = [(t.x, t.y) for t in tpoints]

# Generate final points to add to extended visibility graph
centers = [(p[0], p[1]) for p in xy]
generated = []

radsq = rad * rad
for c in centers:
    pts = [c]
    for t in tpoints:
        # Check points in within radius
        if pow((t[0] - c[0]), 2) + pow((t[1] - c[1]), 2) <= radsq:
            pts.append(t)

    # If enough points within radius, add centroid'
    nWithin = len(pts) - 1
    if nWithin >= threshold:
        # Centroid of all points & circle center
        g = np.mean(np.array(pts), axis=0)
        # Above often lands in polygons, 
        # so now take the centroid between g and circle center
        g = np.mean(np.array([c, g]), axis = 0)
        generated.append(g)

# Switch order (x, y) --> (y, x) to match visibility graph
evg = [vg.Point(g[1], g[0]) for g in generated]
# Add points to visgraph
added = []
for g in evg:
    try:
        for v in visible_vertices(g, graph.graph):
            if point_in_polygon(g, graph.graph) < 0 and point_in_polygon(v, graph.graph) < 0:
                graph.visgraph.add_edge(Edge(g, v))
    except:
        continue
print("End extending visibility graph")

# Save evg
graph.save(graphFileOut)

# Plot points
ax = plt.subplot(111)
plt.scatter([c[0] for c in generated], [c[1] for c in generated], c = 'blue', alpha = 1.0)
plt.scatter(gx, gy, c = 'red')
ax.set_xlim(0, rangeWidth)
ax.set_ylim(0, rangeWidth)
plt.savefig(mapFileOut)
