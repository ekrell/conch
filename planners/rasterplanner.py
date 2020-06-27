import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
from haversine import haversine
import time

import gridplanner

def getGridExtent(data):
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
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols }

def world2grid (lat, lon, transform, nrow):
    row = int ((lat - transform[3]) / transform[5])
    col = int ((lon - transform[0]) / transform[1])
    return (row, col)

def grid2world (row, col, transform, nrow):
    lon = transform[1] * col + transform[2] * row + transform[0]
    lat = transform[4] * col + transform[5] * row + transform[3]
    return (lat, lon)

parser = OptionParser()
parser.add_option("-r", "--region",
                  help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
                  default = "test/full.tif")
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
parser.add_option(      "--nhood_type",
                   help = "Neighborhood type (4, 8, or 16).",
                   default = 4)

(options, args) = parser.parse_args()

print(options)

regionRasterFile = options.region
startPoint = (float(options.sy), float(options.sx))
endPoint = (float(options.dy), float(options.dx))
solver = options.solver.lower()
nhoodType = int(options.nhood_type)
if nhoodType != 4 and nhoodType != 8 and nhoodType != 16:
    print("Invalid neighborhood type {}".format(nhoodType))
    exit(-1)

# Read region raster
regionData = gdal.Open(regionRasterFile)
regionExtent = getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()
rows, cols = grid.shape

print("Using {r}x{c} grid {g}".format(r = rows, c = cols, g = regionRasterFile))
print("Using {n}-way neighborhood".format(n = nhoodType))
print("  Start:", startPoint)
print("  End:", endPoint)

# Convert latlon -> rowcol
start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

# Solve
t0 = time.time()

if solver == "a*":
    path = gridplanner.astar(grid, start, end, ntype = nhoodType)
else:  # Default to dijkstra
    solver = "dijkstra"
    path = gridplanner.dijkstra(grid, start, end, ntype = nhoodType)
t1 = time.time()
print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))

# Stat path
path_distance = 0
prev_point = path[0]
for point in path[1:]:
    prev_latlon = grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
    point_latlon = grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
    path_distance += haversine(prev_latlon, point_latlon)
    prev_point = point
#path_duration = (path_distance / (targetSpeed_mps / 100)) / 60

print('    Distance: {:.4f} km'.format(path_distance))
#print('    Duration: {:.4f} min'.format(path_duration))
#print('    Cost: {:.4f}'.format(C))

print(path)









