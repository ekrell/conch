#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from optparse import OptionParser
import dill as pickle
from math import acos, cos, sin, ceil, floor
from osgeo import gdal
from bresenham import bresenham
import osgeo.gdalnumeric as gdn
from haversine import haversine
import time
import pygmo as pg

class solvePath:
    def __init__(self, start, end, grid, targetSpeed_mps, bounds = None,
                 currentsGrid_u = None, currentsGrid_v = None, currentsTransform = None, regionTransform = None,
                 waypoints = 5, timeIn = 0, interval = 3600, weights = (1, 1, 1)):
        self.start = np.array(start).astype(int)
        self.end = np.array(end).astype(int)
        self.waypoints = waypoints
        self.dim = waypoints * 2
        self.pathlen = waypoints + 2
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
        self.targetSpeed_mps = targetSpeed_mps
        self.currentsGrid_u = currentsGrid_u
        self.currentsGrid_v = currentsGrid_v
        self.currentsTransform = currentsTransform
        self.regionTransform = regionTransform
        self.timeIn = timeIn
        self.interval = interval
        self.emptyPath = np.zeros((waypoints + 2, 2)).astype(int)
        self.emptyPath[0, :] = self.start
        self.emptyPath[waypoints + 1, :] = self.end
        if bounds is None:
            bounds = [0, self.rows, 0, self.cols]
        self.b = bounds
        self.weights = weights

    def fitness(self, x):
        path = self.emptyPath
        path[1:self.waypoints + 1, 0] = x[::2].astype(int)
        path[1:self.waypoints + 1, 1] = x[1::2].astype(int)
        work, dist, obs = calcWork(path, self.pathlen, self.grid, self.targetSpeed_mps,
                self.currentsGrid_u, self.currentsGrid_v, self.currentsTransform, self.regionTransform, self.timeIn, self.interval)
        return [(work * self.weights[1]) + (0 * self.weights[2]) + (obs * obs * 100)]

    def get_bounds(self):
        lowerBounds = np.zeros(self.dim)
        upperBounds = np.zeros(self.dim)
        for i in range(self.dim):
            if i % 2 == 0:
                lowerBounds[i] = self.b[0]
                upperBounds[i] = self.b[1]
            else:
                lowerBounds[i] = self.b[2]
                upperBounds[i] = self.b[3]
        return (lowerBounds, upperBounds)

    def get_name(self):
        return "Solve path"

def raster2array(raster, dim_ordering="channels_last", dtype='float32'):
    # Modified from: https://gis.stackexchange.com/a/283207
    bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount + 1)]
    return np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)

def calcWork(path, n, regionGrid, targetSpeed_mps, currentsGrid_u = None, currentsGrid_v = None, currentsTransform = None, regionTransform = None, timeIn = 0, interval = 3600):
    '''
    This function calculates cost to move vehicle along path
    '''
    if n <= 1:
        return 0 # Vehicle does nothing

    v = path[0]

    pDist = np.zeros(n)
    pObs = np.zeros(n)
    pWork = np.zeros(n)

    # Estimate elapsed time, temporal raster band
    elapsed = float(timeIn)
    (index, rem) = divmod(elapsed, interval)
    index = floor(index)

    for i in range(1, n):

        w = path[i]

        # Heading
        vecs = (w[1] - v[1], w[0] - v[0])
        dotprod = vecs[0] * 1 + vecs[1] * 0
        maga = pow(vecs[0] * vecs[0] + vecs[1] * vecs[1], 0.5)
        heading_rad = 0.0
        if (maga != 0):
            costheta = dotprod / maga
            heading_rad = acos(costheta)

        # Distance
        distance = pow((pow(v[0] - w[0], 2) + pow(v[1] - w[1], 2)), 0.5)

        # Obstacles
        b = list(bresenham(v[0], v[1], w[0], w[1]))
        for p in b[1:]: # Skip first pixel -> already there!
            # Check obstacles
            if grid[p[0] - 1, p[1] - 1] != 0:
                pObs[i] += 1

        work = 0
        # Calc work to oppose forces
        if currentsGrid_u is not None and currentsGrid_v is not None:

            # Convert to latlon to get correct pixel in the water currents raster
            # This is becuase the size/resolution of regions and currents might not be equal
            v_latlon = grid2world(v[0], v[1], regionTransform, grid.shape[0])
            v_currents = world2grid(v_latlon[0], v_latlon[1], currentsTransform, currentsGrid_u.shape[1])
            w_latlon = grid2world(w[0], w[1], regionTransform, grid.shape[0])
            w_currents = world2grid(w_latlon[0], w_latlon[1], currentsTransform, currentsGrid_u.shape[1])

            b = list(bresenham(v_currents[0], v_currents[1], w_currents[0], w_currents[1]))
            pixeldist = distance / (len(b) -1)
            for p in b[1:]: # Skip first pixel -> already there!
                xA = targetSpeed_mps * cos(heading_rad)
                yA = targetSpeed_mps * sin(heading_rad)
                # Interpolate (in time) between nearest discrete force values
                cmag = currentsGrid_u[index, p[0] - 1, p[1] - 1] * (rem / interval) + \
                    currentsGrid_u[index + 1, p[0] - 1, p[1] - 1] * (1 - (rem / interval))
                cdir = currentsGrid_v[index, p[0] - 1, p[1] - 1] * (rem / interval) + \
                    currentsGrid_v[index + 1, p[0] - 1, p[1] - 1] * (1 - (rem / interval))
                xB = cmag * cos(cdir)
                yB = cmag * sin(cdir)
                dV = (xB - xA, yB - yA)
                magaDV = pow(dV[0] * dV[0] + dV[1] * dV[1], 0.5)
                dirDV = 0.0
                dotprod = dV[0] * 1 + dV[1] * 0
                if magaDV != 0:
                    costheta = dotprod / magaDV
                    dirDV = acos(costheta)
                work += magaDV * pixeldist

                # Update time
                elapsed += pixeldist / targetSpeed_mps
                (index, rem) = divmod(elapsed, interval)
                index = min(floor(index), currentsGrid_u.shape[0] - 2)

        pDist[i] = distance
        pWork[i] = work
        v = w

    return sum(pWork), sum(pDist), sum(pObs)

def world2grid(lat, lon, transform, nrow):
    row = int ((lat - transform[3]) / transform[5])
    col = int ((lon - transform[0]) / transform[1])
    return (row, col)

def grid2world(row, col, transform, nrow):
    lon = transform[1] * col + transform[2] * row + transform[0]
    lat = transform[4] * col + transform[5] * row + transform[3]
    return (lat, lon)

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
    return { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols  }

###########
# Options #
###########

parser = OptionParser()
# Environment
parser.add_option("-r", "--region",
                  help  = "Path to raster containing occupancy grid (0 -> free space).",
                  default = "test/acc2020/full.tif")
parser.add_option("-m", "--map",
                  help = "Path to save solution path map.",
                  default = "test/acc2020/test.png")
parser.add_option("-p", "--path",
                  help = "Path to save solution path.",
                  default = "test/acc2020/test.txt")
parser.add_option("-u", "--currents_mag",
                  help = "Path to raster with magnitude of water velocity.",
                  default = "test/acc2020/waterMag.tif")
parser.add_option("-v", "--currents_dir",
                  help = "Path to raster with direction of water velocity.",
                  default = "test/acc2020/waterDir.tif")
parser.add_option(      "--sx",
                  help = "Start longitude.",
                  default = -70.99428)
parser.add_option(      "--sy",
                  help = "Start latitude.",
                  default = 42.32343)
parser.add_option(      "--dx",
                  help = "Destination longitude.",
                  default = -70.88737)
parser.add_option(      "--dy",
                  help = "Destination latitude.",
                  default = 42.33600)
parser.add_option(      "--speed",
                  help = "Target boat speed (meters/second).",
                  default = 0.5)
parser.add_option(      "--time_offset",
                  help = "If needed, specify time offset (seconds) from start of water currents raster.",
                  default = 0, type = "int")
parser.add_option(      "--time_interval",
                  help = "Time (seconds) between bands in water currents raster.",
                  default = 3600, type = "int")
parser.add_option(      "--bounds",
                  help = "Comma-delimited bounds for region raster i.e. 'ymin,ymax,xmin,xmax'",
                  default = None)
# Optimization parameters
parser.add_option("-n", "--num_waypoints",   type = "int", default = 5,
                  help = "Number of solution waypoints to generate.")
parser.add_option(      "--generations",     type = "int", default = 1000,
                  help = "Number of optimization generations.")
parser.add_option(      "--pool_size",       type = "int", default = 100,
                  help = "Number of individuals in optimization pool")
parser.add_option(      "--distance_weight",     type = "int", default = 1,
                  help = "Weight of distance attribute in fitness.")
parser.add_option(      "--force_weight",    type = "int", default = 1,
                  help = "Weight of force attribute in fitness.")
parser.add_option(     "--reward_weight",    type = "int", default = 1,
                  help = "Weight of reward attribute in fitness")
parser.add_option(     "--hyperparams",
                  help = "Comma-delimited selection for solver and its options",
                  default = "pso,0.7298,2.05,2.05")

(options, args) = parser.parse_args()

# Environment
regionRasterFile = options.region
mapOutFile = options.map
pathOutFile = options.path
currentsRasterFile_u = options.currents_mag
currentsRasterFile_v = options.currents_dir
startPoint = (float(options.sy), float(options.sx))
endPoint = (float(options.dy), float(options.dx))
targetSpeed_mps = float(options.speed)
timeOffset_s = options.time_offset
timeInterval_s = options.time_interval

# Optimization
numWaypoints = options.num_waypoints
generations = options.generations
poolSize = options.pool_size
distanceWeight = options.distance_weight
forceWeight = options.force_weight
rewardWeight = options.reward_weight
weights = (distanceWeight, forceWeight, rewardWeight)
hyperparams = options.hyperparams.split(",")
for i in range(1, len(hyperparams)):
    hyperparams[i] = float(hyperparams[i])

print("Using input region raster: {}".format(regionRasterFile))
print("  Start:", startPoint)
print("    End:", endPoint)

# Load raster
regionData = gdal.Open(regionRasterFile)
regionExtent = getGridExtent(regionData)
regionTransform = regionData.GetGeoTransform()
grid = regionData.GetRasterBand(1).ReadAsArray()

# Bounds
if options.bounds is None:
    bounds = np.array([0, regionExtent["rows"], 0, regionExtent["cols"]])
else:
    bounds = np.array(options.bounds.split(",")).astype(int)

# Read currents rasters
elapsedTime = 0
bandIndex = 0
usingCurrents = False
if currentsRasterFile_u is not None and currentsRasterFile_v is not None:
    bandIndex = 1

    # Load force magnitudes
    currentsData_u = gdal.Open(currentsRasterFile_u)
    currentsExtent_u = getGridExtent(currentsData_u)
    currentsTransform_u = currentsData_u.GetGeoTransform()
    currentsGrid_u = raster2array(currentsData_u)

    # Load force directions
    currentsData_v = gdal.Open(currentsRasterFile_v)
    currentsExtent_v = getGridExtent(currentsData_v)
    currentsTransform_v = currentsData_v.GetGeoTransform()
    currentsGrid_v = raster2array(currentsData_v)

    # Sanity check that the current mag, dir rasters match
    if currentsExtent_u["rows"] != currentsExtent_v["rows"] or \
            currentsExtent_u["cols"] != currentsExtent_v["cols"]:
        print("[-] Spatial extent mismatch between currents u, v")
        exit(-1)

    usingCurrents = True
    print("Incorporating forces into energy cost:")
    print("  u: {}".format(currentsRasterFile_u))
    print("  v: {}".format(currentsRasterFile_v))

# Convert latlon -> rowcol
start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

# Solve
print("Begin solving")
prob = pg.problem(solvePath(start, end, grid, targetSpeed_mps = targetSpeed_mps,
        waypoints = numWaypoints, bounds = bounds, weights = weights,
        currentsGrid_u = currentsGrid_u, currentsGrid_v = currentsGrid_v,
        currentsTransform = currentsTransform_u, regionTransform = regionTransform,
        timeIn = timeOffset_s, interval = timeInterval_s))
algo = pg.algorithm(pg.pso(gen = generations, omega = hyperparams[1], eta1 = hyperparams[2], eta2 = hyperparams[3]))
#algo = pg.algorithm(pg.bee_colony(gen = generations, limit = 20))
algo.set_verbosity(10)
pop = pg.population(prob, poolSize)
t0 = time.time()
pop = algo.evolve(pop)
t1 = time.time()
# Extract best solution
x = pop.champion_x
fitness = pop.champion_f

# Solution information
work = 0
path = np.zeros((numWaypoints + 2, 2)).astype(int)
path[0, :] = np.array(start).astype(int)
path[numWaypoints + 1, :] = np.array(end).astype(int)
path[1:numWaypoints + 1, 0] = x[::2].astype(int)
path[1:numWaypoints + 1, 1] = x[1::2].astype(int)
work, dist, obs = calcWork(path, numWaypoints + 2, grid, targetSpeed_mps,
                           currentsGrid_u, currentsGrid_v, currentsTransform_u, regionTransform,
                           timeOffset_s, timeInterval_s)

# Haversine distance
path_distance = 0
prev_point = path[0]
for point in path[1:]:
    prev_latlon = grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
    point_latlon = grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
    path_distance += haversine(prev_latlon, point_latlon)
    prev_point = point
path_duration = ((path_distance / (targetSpeed_mps / 100)) / 60)

print("Done solving with {alg}, {s} seconds".format(alg = hyperparams[0], s = t1 - t0))
if usingCurrents:
    print("Planning results (with currents):")
else:
    print("Planning results (no currents):")
print('    Distance: {:.4f} km'.format(path_distance))
print('    Duration: {:.4f} min'.format(path_duration))
print('    Cost: {:.4f}'.format(work))
print('    Reward: COMING SOON')
print('    Fitness: ', fitness[0])

# Plot solution path
lats = np.zeros(numWaypoints + 2)
lons = np.zeros(numWaypoints + 2)
for i in range(numWaypoints + 2):
    lats[i] = path[i][0]
    lons[i] = path[i][1]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_ylim(0, regionExtent["rows"])
ax.set_xlim(0, regionExtent["cols"])
ax.set_facecolor('xkcd:lightblue')
plt.imshow(grid, cmap=plt.get_cmap('gray'))
plt.plot(lons, lats, '--')
plt.plot(lons[0], lats[0], 'ro')
plt.plot(lons[-1], lats[-1], 'rv')
ax.set_ylim(ax.get_ylim()[::-1])
plt.savefig(mapOutFile)
np.savetxt(pathOutFile, path, delimiter=',')
