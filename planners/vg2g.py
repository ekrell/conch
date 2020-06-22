#!/usr/bin/python3

import pyvisgraph as vg
import fiona
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gp
from optparse import OptionParser
import dill as pickle
from osgeo import gdal
from pyvisgraph.visible_vertices import visible_vertices, point_in_polygon
from pyvisgraph.visible_vertices import closest_point
from pyvisgraph.graph import Graph, Edge

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
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols   }

def world2grid (lat, lon, transform, nrow):
    row = int ((lat - transform[3]) / transform[5])
    col = int ((lon - transform[0]) / transform[1])
    return (row, col)

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
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_evg.graph")
parser.add_option("-o", "--outgraph",
        help = "Path to output pickle graph.",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_vg.pickle")
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

(options, args) = parser.parse_args()

rasterFileIn = options.region
graphFileIn = options.visgraph
graphFileOut = options.outgraph
startPoint = (float(options.sy), float(options.sx))
endPoint = (float(options.dy), float(options.dx))

print(options)

print(rasterFileIn, graphFileIn)
print(startPoint)

# Load visibility graph
print("Begin loading visibility graph")
graph = vg.VisGraph()
graph.load(graphFileIn)
print("Done loading visibility graph")

# Scale based on raster dimensions
regionData = gdal.Open(rasterFileIn)
regionExtent = getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

minx = regionExtent["minx"]
maxx = regionExtent["maxx"]
miny = regionExtent["miny"]
maxy = regionExtent["maxy"]

# Add the start and stop point to grid
origin = vg.Point(round((round(startPoint[1], 10) - minx) / (maxx - minx) * rangeWidth, 6),
        round((round(startPoint[0], 10) - miny) / (maxy - miny) * rangeWidth, 6))
destination = vg.Point(round((round(endPoint[1], 10) - minx) / (maxx - minx) * rangeWidth, 6),
        round((round(endPoint[0], 10) - miny) / (maxy - miny) * rangeWidth, 6))

print("start:", startPoint, "scaled:", origin)
print("destination", endPoint, "scaled:", destination)

origin_exists = origin in graph.visgraph
dest_exists = destination in graph.visgraph
orgn = None if origin_exists else origin
dest = None if dest_exists else destination
if not origin_exists:
    for v in visible_vertices(origin, graph.graph, destination=dest):
        graph.visgraph.add_edge(Edge(origin, v))
if not dest_exists:
    for v in visible_vertices(destination, graph.graph, origin=orgn):
        graph.visgraph.add_edge(Edge(destination, v))

# Convert to simple dictionary graph
dgraph = {}

for v in graph.visgraph.get_points():
    v_latlon = ( (v.y  / rangeWidth) * (maxy - miny) + miny, (v.x  / rangeWidth) * (maxx - minx) + minx )
    v_rowcol = world2grid(v_latlon[0], v_latlon[1], regionTransform, regionExtent["rows"])
    v_edges = []
    for edge in graph.visgraph[v]:
        w = edge.get_adjacent(v)
        w_latlon = ( (w.y  / rangeWidth) * (maxy - miny) + miny, (w.x  / rangeWidth) * (maxx - minx) + minx )
        w_rowcol = world2grid(w_latlon[0], w_latlon[1], regionTransform, regionExtent["rows"])

        v_edges.append(w_rowcol)

    dgraph[v_rowcol] = v_edges

pickle.dump(dgraph, open(graphFileOut, "wb"))

