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

###########
# Options #
###########

# Raster file
regionRasterFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.tif"
graphOutFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.graph"
shapeOutFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.shp"
mapOutFile = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.png"
numWorkers = 4
build = False

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.tif")
parser.add_option("-g", "--graph", 
        help = "Path to save  visibility graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.graph")
parser.add_option("-s", "--shape", 
        help = "Path to save shapefile",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.shp")
parser.add_option("-m", "--map", 
        help = "Path to save polygon map",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region.png")
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
    image = src.read(1).astype('uint8') # first band
    mask = image != 0
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=src.transform)))
geoms = list(results)

# Convert to shapely 
polygons = []
shapes = []
for geom in geoms:
    shape_ = shape(geom['geometry'])
    polygon_ = []
    for point in shape_.exterior.coords:
        polygon_.append(vg.Point(round(point[0], 8), round(point[1], 8)))
    shapes.append(shape_)
    polygons.append(polygon_)
print("Done extracting polygons from raster")

# Save shapefile
gdf  = gp.GeoDataFrame.from_features(geoms)
# save the GeoDataFrame
gdf.to_file(driver = 'ESRI Shapefile', filename= shapeOutFile)
print("Saved polygon shapefile to file: {}".format(shapeOutFile))

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

# Test shapefile
input_shapefile = shapefile.Reader('/home/ekrell/Downloads/ADVGEO_DL/tiny2.shp')
shapes = input_shapefile.shapes()

polygons = []
for shape in shapes:
    polygon = []
    for point in shape.points:
        polygon.append(vg.Point(point[0], point[1]))
    polygons.append(polygon)

# Start building the visibility graph 
graph = vg.VisGraph()
print('Begin building visibility graph')
graph.build(polygons, workers = numWorkers)
print('Done building visibility graph')

# Save graph
graph.save(graphOutFile)
print("Saved visibility graph to file: {}".format(graphOutFile))
