import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
import haversine
import osgeo.gdalnumeric as gdn
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

def raster2array(raster, dim_ordering="channels_last", dtype='float32'):
    '''
    Modified from: https://gis.stackexchange.com/a/283207
    '''
    bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
    return arr

parser = OptionParser()
parser.add_option("-r", "--region",
                  help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free).",
                  default = "test/inputs/full.tif")
parser.add_option("-u", "--currents_mag",
                  help = "Path to grid with magnitude of water velocity.",
                  default = None)
parser.add_option("-v", "--currents_dir",
                  help = "Path to grid with direction of water velocity.",
                  default = None)
parser.add_option("-m", "--map",
                  help = "Path to save solution path map.",
                  default = None)
parser.add_option("-p", "--path",
                  help = "Path to save solution path.",
                  default = "test/rasterplan.txt")
parser.add_option("--sx",
                   help = "Start longitude.",
                   default = -70.99428)
parser.add_option("--sy",
                   help = "Start latitude.",
                   default = 42.32343)
parser.add_option("--dx",
                   help = "Destination longitude.",
                   default = -70.88737)
parser.add_option("--dy",
                   help = "Destination latitude.",
                   default = 42.33600)
parser.add_option("--solver",
                   help = "Path finding algorithm (A*, dijkstra).",
                   default = "dijkstra")
parser.add_option("-n", "--nhood_type",
                   help = "Neighborhood type (4, 8, or 16).",
                   default = 4)
parser.add_option(      "--trace",
                   help = "Path to save map of solver's history. Shows which cells were evaluated.",
                   default = None)
parser.add_option(      "--speed",
                  help = "Target boat speed (meters/second).",
                  default = 0.5)
parser.add_option(      "--statpath",
                  help = "Path to list of waypoints to print path information. Will not solve.",
                  default = None)
parser.add_option(     "--dist_measure",
                  help = "Which distance measurement to use (haversine, euclidean, or euclidean-scaled).",
                  default = "haversine")
(options, args) = parser.parse_args()

regionRasterFile = options.region
currentsRasterFile_u = options.currents_mag
currentsRasterFile_v = options.currents_dir
mapOutFile = options.map
pathOutFile = options.path
startPoint = (float(options.sy), float(options.sx))
endPoint = (float(options.dy), float(options.dx))
solver = options.solver.lower()
nhoodType = int(options.nhood_type)
if nhoodType != 4 and nhoodType != 8 and nhoodType != 16:
    print("Invalid neighborhood type {}".format(nhoodType))
    exit(-1)
distMeasure = options.dist_measure
if distMeasure not in ["haversine", "euclidean", "euclidean-scaled"]:
    distMeasure = "haversine"
trace = False
traceOutFile = options.trace
if options.trace is not None:
    trace = True
targetSpeed_mps = float(options.speed)
statPathFile = options.statpath

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
print("  Speed: {} m/s".format(targetSpeed_mps))
print("  Distance measurement: {d}".format(d = distMeasure))

# Read currents rasters
elapsedTime = 0
bandIndex = 0
usingCurrents = False
if currentsRasterFile_u is not None and currentsRasterFile_v is not None:
    usingCurrents = True
    bandIndex = 1

    # Load u components
    currentsData_u = gdal.Open(currentsRasterFile_u)
    currentsExtent_u = getGridExtent(currentsData_u)
    currentsTransform_u = currentsData_u.GetGeoTransform()
    currentsGrid_u = raster2array(currentsData_u)

    # Load v components
    currentsData_v = gdal.Open(currentsRasterFile_v)
    currentsExtent_v = getGridExtent(currentsData_v)
    currentsTransform_v = currentsData_v.GetGeoTransform()
    currentsGrid_v = raster2array(currentsData_v)

    # Sanity check that the current u, v rasters match
    if currentsExtent_u["rows"] != currentsExtent_v["rows"] or \
            currentsExtent_u["cols"] != currentsExtent_v["cols"]:
        print("[-] Spatial extent mismatch between currents u, v")
        exit(-1)
    if currentsData_u.RasterCount != currentsData_v.RasterCount:
        print("[-] Band count mismatch between currents u, v")
        exit(-1)

    # If single-band, make multi-band (with 1)
    if len(currentsGrid_u.shape) == 2:
        cu = np.zeros((1, currentsGrid_u.shape[0], currentsGrid_u.shape[1]))
        cu[0] = currentsGrid_u.copy()
        currentsGrid_u = cu.copy()
        cv = np.zeros((1, currentsGrid_v.shape[0], currentsGrid_v.shape[1]))
        cv[0] = currentsGrid_v.copy()
        currentsGrid_v = cv

    print("Incorporating forces into energy cost:")
    print("  magnitude: {}".format(currentsRasterFile_u))
    print("  direction: {}".format(currentsRasterFile_v))

# Convert latlon -> rowcol
start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

# Calculate pixel distance
s = grid2world(0, 0, regionTransform, regionExtent["rows"])
e = grid2world(0, 1, regionTransform, regionExtent["rows"])
dist_m = haversine.haversine(s, e) * 1000
pixelsize_m = dist_m
print(pixelsize_m)

if solver == "a*" or solver == "astar":
    solver_id = 1
else:
    # Default solver to dijkstra
    solver = "dijkstra"
    solver_id = 0

if statPathFile is None:
    # Solve
    t0 = time.time()
    if usingCurrents:
        path, traceGrid, C, T = gridplanner.solve(grid, start, end, solver = solver_id, ntype = nhoodType, trace = trace,
                currentsGrid_u = currentsGrid_u, currentsGrid_v = currentsGrid_v, geotransform = regionTransform,
                targetSpeed_mps = targetSpeed_mps, pixelsize_m = pixelsize_m, distMeas = distMeasure)
    else:
        path, traceGrid, C, T = gridplanner.solve(grid, start, end, solver = solver_id, ntype = nhoodType, trace = trace,
                targetSpeed_mps = targetSpeed_mps, pixelsize_m = pixelsize_m,
                geotransform = regionTransform, distMeas = distMeasure)
    t1 = time.time()
    print("Done solving with {s}, {t} seconds".format(s = solver, t = t1 - t0))
else:
    print("Stat of path file {f}".format(f = statPathFile))
    path = [(int(p[0]), int(p[1])) for p in np.loadtxt(statPathFile, delimiter = ",")]
    C, T = gridplanner.statPath(path, currentsGrid_u, currentsGrid_v, regionTransform, targetSpeed_mps, pixelsize_m = pixelsize_m)

# Stat path
numWaypoints = len(path)
path_distance = 0
prev_point = path[0]
for point in path[1:]:
    prev_latlon = grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
    point_latlon = grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
    path_distance += haversine.haversine(prev_latlon, point_latlon)
    prev_point = point
path_duration = (path_distance * 1000 / targetSpeed_mps) / 60

print(T / 60)
print('    Distance: {:.4f} km'.format(path_distance))
print('    Duration: {:.4f} min'.format(path_duration))
if usingCurrents:
    print('    Cost: {:.4f}'.format(C))


# Visualize
# Plot trace of solver
if trace:
    plt.imshow(traceGrid, cmap = "Greys")
    plt.savefig(traceOutFile)

# Plot solution path
if mapOutFile is not None:
    lats = np.zeros(numWaypoints)
    lons = np.zeros(numWaypoints)
    for i in range(numWaypoints):
        lats[i] = path[i][0]
        lons[i] = path[i][1]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_ylim(0, regionExtent["rows"])
    ax.set_xlim(0, regionExtent["cols"])
    ax.set_facecolor('xkcd:lightblue')
    # Plot raster
    plt.imshow(grid, cmap=plt.get_cmap('gray'))
    plt.plot(lons, lats)

    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig(mapOutFile)


# Write path
if pathOutFile is not None:
    np.savetxt(pathOutFile, path, delimiter = ",", fmt = "%d")
