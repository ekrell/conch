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
import shapefile
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

rasterFileIn = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif"
graphFileIn = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_evg.graph"
shapeFileIn = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.shp"
graphFileOut = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini_vg.pickle"

startPoint = (-70.92, 42.29) 
endPoint = (-70.96, 42.30)

# Load shapefile
input_shapefile = shapefile.Reader(shapeFileIn)
shapes = input_shapefile.shapes()

# Load visibility graph
print("Begin loading visibility graph")
graph = vg.VisGraph()
graph.load(graphFileIn)
print("Done loading visibility graph")

# Find extent
#minx = shapes[0].points[0][0]
#maxx = minx
#miny = shapes[0].points[0][1]
#maxy = miny
#for shape in shapes:
#    for point in shape.points:
#        if point[0] < minx:
#            minx = point[0]
#        elif point[0] > maxx:
#            maxx = point[0]
#        if point[1] < miny:
#            miny = point[1]
#        elif point[1] > maxy:
#            maxy = point[1]

# Scale based on raster dimensions
regionData = gdal.Open(rasterFileIn)
regionExtent = rsi.getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

minx = regionExtent["minx"]
maxx = regionExtent["maxx"]
miny = regionExtent["miny"]
maxy = regionExtent["maxy"]

# Add the start and stop point to grid
origin = vg.Point(round((round(startPoint[0], 10) - minx) / (maxx - minx) * rangeWidth, 6), 
        round((round(startPoint[1], 10) - miny) / (maxy - miny) * rangeWidth, 6))
destination = vg.Point(round((round(endPoint[0], 10) - minx) / (maxx - minx) * rangeWidth, 6),
        round((round(endPoint[1], 10) - miny) / (maxy - miny) * rangeWidth, 6))

origin_exists = origin in graph.visgraph
dest_exists = destination in graph.visgraph
orgn = None if origin_exists else origin
dest = None if dest_exists else destination
if not origin_exists: 
    for v in visible_vertices(origin, graph.graph, destination=dest):
        graph.graph.add_edge(Edge(origin, v))
if not dest_exists:
    for v in visible_vertices(destination, graph.graph, origin=orgn):
        graph.graph.add_edge(Edge(destination, v))

# Convert to simple dictionary graph
dgraph = {}
for v in graph.graph.get_points():
    v_latlon = ( (v.y  / rangeWidth) * (maxy - miny) + miny, (v.x  / rangeWidth) * (maxx - minx) + minx )
    v_rowcol = gridUtils.world2grid(v_latlon[0], v_latlon[1], regionTransform, regionExtent["rows"])
    v_edges = []
    for edge in graph.graph[v]:
        w = edge.get_adjacent(v)
        w_latlon = ( (w.y  / rangeWidth) * (maxy - miny) + miny, (w.x  / rangeWidth) * (maxx - minx) + minx )
        w_rowcol = gridUtils.world2grid(w_latlon[0], w_latlon[1], regionTransform, regionExtent["rows"])

        v_edges.append(w_rowcol)

    dgraph[v_rowcol] = v_edges

pickle.dump(dgraph, open(graphFileOut, "wb"))

