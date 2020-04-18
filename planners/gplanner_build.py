#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
import dill as pickle
# Conch modules
import rasterSetInterface as rsi

###########
# Options #
###########
freecell = 0
occupied = 1

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif")
parser.add_option("-g", "--graph",
        help = "Path to save graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_uniform.pickle")
parser.add_option("-m", "--map",
        help = "Path to save graph map",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_graph_mini_uniform.png")

(options, args) = parser.parse_args()

regionRasterFile = options.region
graphOutFile = options.graph
mapOutFile = options.map

print("Using input region raster: {}".format(regionRasterFile))

###################
# Region -> Graph #
###################

# Read region raster
regionData = gdal.Open(regionRasterFile)
regionExtent = rsi.getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

# Convert to graph
print("Begin building uniform graph")
graph = {}
for row in range(regionExtent["rows"]):
    for col in range(regionExtent["cols"]):
        # If free space
        if grid[row][col] == freecell:
            # Add edges to unoccupied neighbors
            edges_ = []
            if (row - 1 >= 0) and (grid[row - 1][col] == freecell):
                edges_.append((row - 1, col))
            if (row + 1 < regionExtent["rows"]) and (grid[row + 1][col] == freecell):
                edges_.append((row + 1, col))
            if (col - 1 >= 0) and (grid[row][col - 1] == freecell):
                edges_.append((row, col - 1))
            if (col + 1 < regionExtent["cols"]) and (grid[row][col + 1] == freecell):
                edges_.append((row, col + 1))
            if (row - 1 >= 0) and (col - 1 >= 0) and (grid[row - 1][col - 1] == freecell):
                edges_.append((row - 1, col - 1))
            if (row + 1 < regionExtent["rows"]) and (col - 1 >= 0) and (grid[row + 1][col - 1] == freecell):
                edges_.append((row + 1, col - 1))
            if (row - 1 >= 0) and (col + 1 < regionExtent["cols"]) and (grid[row - 1][col + 1] == freecell):
                edges_.append((row - 1, col + 1))
            if (row + 1 < regionExtent["rows"]) and (col + 1 < regionExtent["cols"]) and \
                (grid[row + 1][col + 1] == freecell):
                    edges_.append((row + 1, col + 1))
            graph[(row, col)] = edges_
print("Done building uniform graph")

# Plot graph
print("Begin plotting uniform graph: {} nodes".format(len(graph)))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_facecolor('xkcd:lightblue')
plt.scatter(list(zip(*graph.keys()))[1], list(zip(*graph.keys()))[0], color = 'green', s = 0.025)
ax.set_ylim(ax.get_ylim()[::-1])
print("Done plotting uniform graph")
plt.savefig(mapOutFile)
print("Saved uniform graph map to file: {}".format(mapOutFile))

# Save graph
pickle.dump(graph, open(graphOutFile, "wb"))
print("Saved uniform graph to file: {}".format(graphOutFile))

