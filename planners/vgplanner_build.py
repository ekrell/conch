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
import random
import time
from osgeo import gdal
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
parser.add_option("-g", "--graph", 
        help = "Path to save  visibility graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.graph")
parser.add_option("-s", "--shape", 
        help = "Path to save shapefile",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.shp")
parser.add_option("-m", "--map", 
        help = "Path to save polygon map",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.png")
parser.add_option("-v", "--vgmap",
        help = "Path to save visibility graph map",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_graph_mini.png")
parser.add_option("-n", "--num_workers", type = "int",
        help = "Number of parallel workers",
        default = 4)
parser.add_option("-b", "--build",
        help = "Build the visibility graph",
        action = "store_true",
        default = False)

(options, args) = parser.parse_args()

regionRasterFile = options.region
graphOutFile = options.graph
shapeOutFile = options.shape
mapOutFile = options.map
vgmapOutFile = options.vgmap
numWorkers = options.num_workers
build = options.build

print("Using input region raster: {}".format(regionRasterFile))
print("Computing with {} workers".format(numWorkers))

######################
# Region -> Polygons #
######################
print("Begin extracting polygons from raster")
mask = None
with rasterio.open(regionRasterFile) as src:
    # Read first band of image (should be binary occupancy grid)
    image = src.read(1).astype('uint8')
    # Only want polygons of obstacles, where 0 -> free cell
    mask = image != 0
    # Store each polygon
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=src.transform)))
geoms = list(results)

# Convert polygons to shapely format, to pyvisgraph format, to geopandas
shapes = []   # For manipulating polygons
polygons = [] # For pyvisgraph storage
gdf = gp.GeoDataFrame() # For writing the polygon shapefile
gdf['geometry'] = None
count = 0 # Track number of polygons
for geom in geoms:
    # Convert to shapely
    shape_ = shape(geom['geometry'])
    # Simplify shape boundary (since raster artificially blocky)
    #shape_ = shape_.buffer(0.0008, join_style=1).buffer(-0.0006, join_style=1)
    shape_ = shape_.simplify(0.0005, preserve_topology=False)
    # If smoothing too aggressive, may loose obstacles! 
    if shape_.is_empty: 
        print ("  Warning! shape lost in simplifying.")
        continue
    polygon_ = []
    # Convert shapely to pyvisgraph format
    for point in shape_.exterior.coords:
        polygon_.append(vg.Point(round(point[0], 10), round(point[1], 10)))
    shapes.append(shape_)
    polygons.append(polygon_)
    # Add as geopandas row
    gdf.loc[count, 'geometry'] = shape_

    count += 1
print("Done extracting {} polygons from raster".format(count))

# save the GeoDataFrame
gdf.to_file(driver = 'ESRI Shapefile', filename = shapeOutFile)
print("Saved shapefile to: {}".format(shapeOutFile))

# Visualize
fig, ax = plt.subplots(nrows=1, ncols=1)
for shape in shapes:
    plt.fill(*shape.exterior.xy, facecolor = 'khaki', edgecolor = 'black', linewidth = 0.5)
ax.set_facecolor('xkcd:lightblue')
plt.savefig(mapOutFile)
print("Saved map to file: {}".format(mapOutFile))

if build == False:
    print("Exiting. Check the saved map to verify polygons")
    print("Re-run with '--build' option to generate the visibility graph.")
    exit(0)

##############################
# Shapes -> Visibility graph #
##############################

## Read shapefile
input_shapefile = shapefile.Reader(shapeOutFile)
shapes = input_shapefile.shapes()

# Load raster
regionData = gdal.Open(regionRasterFile)
regionExtent = rsi.getGridExtent(regionData)
minx = regionExtent["minx"]
maxx = regionExtent["maxx"]
miny = regionExtent["miny"]
maxy = regionExtent["maxy"]

polygons = []
for shape in shapes:
    polygon = []
    for point in shape.points:
        x = ((round(point[0], 10) - minx) / (maxx - minx) * rangeWidth)
        y = ((round(point[1], 10) - miny) / (maxy - miny) * rangeWidth)
        polygon.append(vg.Point(x, y))
        #print (x, y)
        #polygon.append(vg.Point(round(point[0], 8), round(point[1], 8)))
    polygons.append(polygon)

# Start building the visibility graph 
graph = vg.VisGraph()
print('Begin building visibility graph')
t0 = time.time()
graph.build(polygons, workers = numWorkers)
t1 = time.time()
print('Done building visibility graph, {} seconds'.format(t1 - t0))

# Save graph
graph.save(graphOutFile)
print("Saved visibility graph to file: {}".format(graphOutFile))

# Plot visibility graph
edges = graph.visgraph.get_edges()
# Print only P proportion, since very time consuming to plot 
plotProp = 0.1
numSamples = int(len(edges) * plotProp)
#edges = random.sample(edges, numSamples)

print("Begin plotting visibility graph: {} edges".format(len(edges)))
for e in list(edges):
    plt.plot([ \
          (e.p1.x  / rangeWidth) * (maxx - minx) + minx, 
          (e.p2.x  / rangeWidth) * (maxx - minx) + minx], 
        [ (e.p1.y / rangeWidth) * (maxy - miny) + miny, 
          (e.p2.y / rangeWidth) * (maxy - miny) + miny], 
        color = 'green', linestyle = 'dashed', alpha = 0.5)
    plt.scatter([ \
         (e.p1.x  / rangeWidth) * (maxx - minx) + minx, 
         (e.p2.x  / rangeWidth) * (maxx - minx) + minx], 
        [(e.p1.y  / rangeWidth) * (maxy - miny) + miny,
         (e.p2.y  / rangeWidth) * (maxy - miny) + miny], 
        s = 1, color = 'green')
print("Done plotting visibiity graph")
# Save VG map
plt.savefig(vgmapOutFile)
print("Saved visibility graph map to file: {}".format(vgmapOutFile))
