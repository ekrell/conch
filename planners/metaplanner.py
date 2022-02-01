#!/usr/bin/python3
"""Surface vessel path planning using metaheuristic search algorithms.

The path can be optimized based on distance, energy consumption, and reward.
These objectives are combined with a weighted sum fitness function.

Planning is performed to navigate from a given start to goal coordinates.
Optimization is based on the environment, i.e., coastlines, water current
forecasts, and sampling reward. These are provided to the software in the
form of rasters of equal dimensions (rows, cols, grid spacing).
At minimum, the region (i.e. coast, islands) are required for obstacle avoidance.
More planning considerations are enabled by providing the other rasters.

The metaheuristic search algorithms are implemented by the PaGMO optimization
library, The pygmo python bindings are used to allow the user to choose from
the following: Particle Swarm Optimization, Genetic Algorithm, Artificial Bee
Colony, Differential Evolution.

Typically, metaheuristic search algorithms begin with a random initial
population of candidate solutions. In our research, we have found that
we achieve more reliable convergence using an initial population of diverse,
but feasible paths in the search space. Thus, we allow the user to supply
a file of paths in the form of waypoints.

For detailed information, see <https://github.com/ekrell/conch>.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Evan Krell"
__contact__ = "evan.krell@tamucc.edu"
__email__ = "evan.krell@tamucc.edu"
__license__ = "GPLv3"

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from optparse import OptionParser
import dill as pickle
from math import acos, cos, sin, ceil, floor, atan2
from osgeo import gdal
import bresenham_ as bresenham
import osgeo.gdalnumeric as gdn
import time
import pygmo as pg
from itertools import repeat
from haversine import haversine

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Source: stackoverflow user, derricw
    Link: https://stackoverflow.com/a/29546836
    License: CC BY-SA 4.0
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


class solvePath:
    """Defines a path planning problem to be solved with PaGMO library.

    The class is used to define a problem as

        prob = pg.problem(
            solvePath(
                start,
                end,
                grid,
                targetSpeed_mps = targetSpeed_mps,
                waypoints = numWaypoints,
                bounds = bounds,
                weights = weights,
                currentsGrid_u = currentsGrid_u,
                currentsGrid_v = currentsGrid_v,
                currentsTransform = currentsTransform_u,
                regionTransform = regionTransform,
                rewardGrid = rewardGrid,
                timeIn = timeOffset_s,
                interval = timeInterval_s,
                pixelsize_m = pixelsize_m))

    which uses the `__init__` function to initialize the problem based on the
    provided arguments. See the documenation to `__init__` for their definitions.

    """

    def __init__(self, start, end, grid, targetSpeed_mps = 1, bounds = None,
                 currentsGrid_u = None, currentsGrid_v = None, currentsTransform = None, regionTransform = None,
                 rewardGrid = None, waypoints = 5, timeIn = 0, interval = 3600, pixelsize_m = 1, weights = (0, 1, 1)):
        """Initializes a path planning task to solve.

         Args:
            start : (float, float)
                Tuple containing path start coordinates (row, column)
            end : (float, float):
                Tuple containing path goal coordinates (row, column)
            grid : ndarray(dtype=int, ndim=2)
                Array (2D) containing region occupancy grid where 0 indicates a free cell,
                and 1 indicates an obstacle.
            targetSpeed_mps : (float)
                Target speed of the vessel in meters per second.
            bounds (float, float, float, float):
                Tuple containing the coordinates of a bounding rectangle on the
                input 'grid' array. Planning will occur within the bounds.
                Of form (first row, last row, first column, last column).
                These values are direct array indices; the first row is '0'.
                If 'None', will use entire extent of the 'grid'.
            currentsGrid_u : ndarray(dtype=float, ndim=3)
                Array (3D) containing the magnitudes of the water currents vector field.
                The dimensions represent the (rows, columns, time steps) of a forecast
                of water currents over the planning region.
                The dimensions, rows, columns, grid spacing, should match the 'grid',
                so that at each (row, column) cell, the planner can look up the
                water currents.
                If 'None', will not incorporate water currents in fitness calculations.
            currentsGrid_v : ndarray(dtype=float, ndim=3)
                Array (3D) containing the directions of the water currents vector field.
                Must match the extent, dimensions of the magnitudes in `currentsGrid_u`.
                They form a pair to define the water currents forecast.
                If 'None', will not incorporate water currents in fitness calculations.
            currentsTransform : (float, float, float, float, float, float)
                Tuple containing the affine tranformation of the water currents
                grid coordinate space (row, column) to georeferences (lat, lon).
                Defined by GDAL, see <https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut>.
            regionTransform : (float, float, float, float, float, float)
                Tuple containing the affine tranformation of the region `grid`
                coordinate space (row, column) to georeferences (lat, lon).
                Defined by GDAL, see <https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut>.
            rewardGrid : ndarray(dtype=float, ndim=2)
                Array (2D) of reward values assigned to each cell of the region.
                Thus, must be the same dimensions as the region 'grid',
                so that each (row, column) can be used to look up the reward value.
                If 'None', will not incorporate `reward` in fitness calculations.
            waypounts : (int)
                Number of waypoints to include in the solution path.
            timeIn : (float)
                Elapsed time in seconds to begin planning, w.r.t. the water forecasts.
                Probable that a real planning task does not begin at exaclty
                the first hour of the water currents forecast. So, this value
                specifies the elapsed seconds since the start of the forecasts.
            interval : (float)
                Interval in seconds between the forecast time steps.
            pixelsize_m : (float)
                Size of pixels in meters.
            weights : (float, float, float)
                The user-specified weights of the planning objectives.
                These are (distance, energy, reward), respectively.
                By default, distance is set to 0 for energy-based planning.
                The reward will only be meaningful if a reward grid is provided.
        """

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
        self.rewardGrid = rewardGrid
        self.timeIn = timeIn
        self.interval = interval
        self.pixelsize_m = pixelsize_m
        self.emptyPath = np.zeros((waypoints + 2, 2)).astype(int)
        self.emptyPath[0, :] = self.start
        self.emptyPath[waypoints + 1, :] = self.end
        if bounds is None:
            # Default bounds to entire grid extent
            bounds = [0, self.rows - 1, 0, self.cols - 1]
        self.b = bounds
        self.weights = weights

    def fitness(self, x):
        """Calculate path fitness.

        Args:
            x : ndarray(dtype=float, ndim=1)
                Array (1D) containing path waypoints as a sequence of row, column values.
                The pattern is [row_0, col_0, row_1, col_1, ..., row_N, col_N],
                where N is the number of waypoints.

        Returns:
            float: The weighted sum of fitness criteria (distance, work, reward, obstacles)

        """

        # Convert path `x` to a 2D ndarray that contains the start and goal coordinates
        path = self.emptyPath
        path[1:self.waypoints + 1, 0] = x[::2].astype(int)
        path[1:self.waypoints + 1, 1] = x[1::2].astype(int)
        # Calculate the work, distance, obstacles, and reward along path
        work, dist, obs, reward = calcWork(path, self.pathlen, self.grid, self.targetSpeed_mps,
                self.currentsGrid_u, self.currentsGrid_v, self.currentsTransform, self.regionTransform,
                self.rewardGrid, self.timeIn, self.interval, self.pixelsize_m)
        # Calculate the fitness as a weighted sum of planning criteria
        return [(dist * self.weights[0])
                    + (work * self.weights[1])
                    + (-1 * reward * self.weights[2])
                    + (obs * obs * 100)]

    def get_name(self):
        return "Metaplanner: USV path planning using metaheuristics"

def raster2array(raster, dtype='float32'):
    """Converts geospatial raster to numpy ndarray.

    Given a GDAL geospatial array object,
    extract the array values to construct a numpy array.

    Modified from: https://gis.stackexchange.com/a/283207

    Args:
        raster : (GDALDataset)
            GDAL object containing geospatial raster data.
            See <https://gdal.org/user/raster_data_model.html#raster-data-model>.
        dtype : (string)
            String of format (numpy dtype) of constructed array.

    Returns:
        ndarray: Array version of raster values.
    """

    bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount + 1)]
    return np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)

def calcWork(path, n, regionGrid, targetSpeed_mps = 1, currentsGrid_u = None,
             currentsGrid_v = None, currentsTransform = None,
             regionTransform = None, rewardGrid = None, timeIn = 0,
             interval = 3600, pixelsize_m = 1):
    '''Calculates path fitness criteria.

    Given a sequence of waypoints, calculates the values of the fitness criteria
    if the vessel were to navigate along it. This is based on the input rasters
    (occupancy grid region, water current forecasts, reward grid). Each of the
    fitness criteria are returned to enable multi-objective planning.

    Args:
        path : ndarray(ndim=2, dtype=float)
            Array containing the path to evaluate as a sequence of waypoints.
            Format: [
                     [start_row,       start_col],
                     [waypoint_row_1,  waypoint_col_1],
                     ...
                     [waypoint_row_N,  waypoint_col_N]
                     [goal_row,        goal_col]
                    ]
        n : int
            Number of waypoints.
        regionGrid : ndarray(dtype=int, ndim=2)
            Array (2D) containing region occupancy grid where 0 indicates a free cell,
            and 1 indicates an obstacle.
        targetSpeed_mps : (float)
                Target speed of the vessel in meters per second.
        currentsGrid_u : ndarray(dtype=float, ndim=3)
            Array (3D) containing the magnitudes of the water currents vector field.
            The dimensions represent the (rows, columns, time steps) of a forecast
            of water currents over the planning region.
            The dimensions, rows, columns, grid spacing, should match the 'grid',
            so that at each (row, column) cell, the planner can look up the
            water currents.
            If 'None', will not incorporate water currents in fitness calculations.
        currentsGrid_v : ndarray(dtype=float, ndim=3)
            Array (3D) containing the directions of the water currents vector field.
            Must match the extent, dimensions of the magnitudes in `currentsGrid_u`.
            They form a pair to define the water currents forecast.
            If 'None', will not incorporate water currents in fitness calculations.
        currentsTransform : (float, float, float, float, float, float)
            Tuple containing the affine tranformation of the water currents
            grid coordinate space (row, column) to georeferences (lat, lon).
            Defined by GDAL, see <https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut>.
        regionTransform : (float, float, float, float, float, float)
            Tuple containing the affine tranformation of the region `grid`
            coordinate space (row, column) to georeferences (lat, lon).
            Defined by GDAL, see <https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut>.
        rewardGrid : ndarray(dtype=float, ndim=2)
            Array (2D) of reward values assigned to each cell of the region.
            Thus, must be the same dimensions as the region 'grid',
            so that each (row, column) can be used to look up the reward value.
            If 'None', will not incorporate `reward` in fitness calculations.
        timeIn : (float)
            Elapsed time in seconds to begin planning, w.r.t. the water forecasts.
            Probable that a real planning task does not begin at exaclty
            the first hour of the water currents forecast. So, this value
            specifies the elapsed seconds since the start of the forecasts.
        interval : (float)
            Interval in seconds between the forecast time steps.
        pixelsize_m : (float)
            Size of pixels in meters.
    '''

    # Initialize costs: distance, obstacles, work, reward
    costs = np.zeros(4)

    # Get all cells along path
    cells_ = np.array([bresenham.bresenhamline(np.array([path[i]]),
                                               np.array([path[i + 1]])) for i, p in enumerate(path[:-1])])
    # Number of cells in each path segment
    lens = np.array([len(c) for c in cells_])
    cells = np.vstack(cells_)

    # Obstacle penalty
    costs[1] = np.sum((grid[cells[:, 0], cells[:, 1]]))

    # Reward
    if rewardGrid is not None:
        costs[3] = np.sum((rewardGrid[cells[:, 0], cells[:, 1]]))

    # Points in (lon, lat)
    lonlat = np.vstack((regionTransform[1] * path[:, 1] + regionTransform[2] * path[:, 0] + regionTransform[0],
                        regionTransform[4] * path[:, 1] + regionTransform[5] * path[:, 0] + regionTransform[3])).T

    # Haversine distances, converted from km to meters
    hdists = haversine_np(lonlat[:-1][:, 0], lonlat[:-1][:, 1], lonlat[1:][:, 0], lonlat[1:][:, 1]) * 1000
    # Distance penalty
    costs[0] = sum(hdists)

    # If no water currents, return now
    if currentsGrid_u is None:
        costs[2] = costs[0]
        return costs[2], costs[0], costs[1], costs[3]

    # Calculate headings
    vecs = np.flip(path[1:] - path[:-1], axis = 1)
    dotprod = vecs[:, 0]
    maga = np.sqrt(vecs[:, 0] * vecs[:, 0] + vecs[:, 1] * vecs[:, 1])
    costheta = vecs[:, 0] / maga # Should check if div by 0?
    heading_rad = np.arccos(costheta)

    # Approx distance of each cell
    hdists_ = hdists / lens

    # Expand path-length variables to cell-length
    hdists_e = np.hstack([np.zeros(c.shape[0]) + hdists_[i] for i, c in enumerate(cells_)])

    # Estimate elapsed time to reach each cell
    ela = [(hdists_e[i] / targetSpeed_mps) * i for i in range(cells.shape[0])]

    # Vehicle target velocities
    xA = targetSpeed_mps * np.cos(heading_rad)
    yA = targetSpeed_mps * np.sin(heading_rad)
    xA = np.hstack([np.zeros(c.shape[0]) + xA[i] for i, c in enumerate(cells_)])
    yA = np.hstack([np.zeros(c.shape[0]) + yA[i] for i, c in enumerate(cells_)])

    # Calculate indices for accessing water current grids
    rem_idx = np.divmod(ela, interval)
    p = np.array(cells[:]).astype("int")
    idx_1 = np.minimum(rem_idx[0], currentsGrid_u.shape[0] - 2).astype("int")
    idx_2 = idx_1 + 1

    # Interpolate all needed water currents
    uv_ = np.array(
        (
            (currentsGrid_u[idx_1, p[:, 0], p[:, 1]] * np.cos(currentsGrid_v[idx_1, p[:, 0], p[:, 1]])) \
                * (1 - (rem_idx[1] / interval)) + \
                    (currentsGrid_u[idx_2, p[:, 0], p[:, 1]]) * np.cos((currentsGrid_v[idx_2, p[:, 0], p[:, 1]])) \
                        * (rem_idx[1] / interval),
            (currentsGrid_u[idx_1, p[:, 0], p[:, 1]]) * np.sin((currentsGrid_v[idx_1, p[:, 0], p[:, 1]])) \
                * (1 - (rem_idx[1] / interval)) + \
                    (currentsGrid_u[idx_2, p[:, 0], p[:, 1]]) * np.sin((currentsGrid_v[idx_2, p[:, 0], p[:, 1]])) \
                        * (rem_idx[1] / interval)
        )
    ).T

    # Calculate applied force required by vehicle to maintain target velocity
    cmag = np.sqrt(np.sum(uv_ * uv_, axis = 1))
    cdir = np.arctan2(uv_[:, 1], uv_[:, 0])
    xB = cmag * np.cos(cdir)
    yB = cmag * np.sin(cdir)
    dV = np.array((xB - xA, yB - yA)).T
    magaDV = np.power(dV[:, 0] * dV[:, 0] + dV[:, 1] * dV[:, 1], 0.5)
    dotprod = dV[:, 0]
    costheta = dotprod / magaDV   # Prob need to check for div by 0 ??
    dwork = magaDV * hdists_e

    # Work penalty
    costs[2] = np.sum(dwork)

    return costs[2], costs[0], costs[1], costs[3]

def world2grid(lat, lon, transform, nrow):
    """"Convert a (lat, lon) coordinate to (row, column).

    Args:
        lat : (float)
            Latitude value.
        lon : (float)
            Longitude value.
        transform : (float, float, float, float, float, float)
            Tuple containing the affine tranformation of the grid
            coordinate space (row, column) to georeferences (lat, lon).
            Defined by GDAL, see <https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut>.
        nrow : (int)
            Number of rows in raster grid.

    Returns:
        (int, int): The raster grid (row, column)
            that corresponds to the given (lat, lon).
    """
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

if __name__ == "__main__":
    """Perform path planning using metaheuristic algorithm.

    Inputs:
        See <https://github.com/ekrell/conch> for
        documentation on the inputs.

    Outputs:
        Solution path (text):
            If using the '--path' option.
            CSV containing the best-fit path found.
                CSV column 1: path row coordinates in raster space
                CSV column 2: path column coordinates in raster space

            Example:
                $ cat sample_path.txt
                5.790000000000000000e+02,1.870000000000000000e+02
                5.450000000000000000e+02,2.160000000000000000e+02
                4.840000000000000000e+02,5.330000000000000000e+02
                4.770000000000000000e+02,6.130000000000000000e+02
                4.760000000000000000e+02,6.440000000000000000e+02
                4.810000000000000000e+02,7.840000000000000000e+02
                4.850000000000000000e+02,9.890000000000000000e+02

        Solution plot (image):
            If using the '--map' option.
            Figure showing the waypoints and lines between them on the region map.
    """

    parser = OptionParser()
    # Environment
    parser.add_option("-r", "--region",
                      help  = "Path to raster containing occupancy grid (0 -> free space).",
                      default = "test/inputs/full.tif")
    parser.add_option("-m", "--map",
                      help = "Path to save solution path map.",
                      default = "test/metaplan.png")
    parser.add_option("-p", "--path",
                      help = "Path to save solution path.",
                      default = "test/metaplan.txt")
    parser.add_option("-u", "--currents_mag",
                      help = "Path to raster with magnitude of water velocity.",
                      default = None)
    parser.add_option("-v", "--currents_dir",
                      help = "Path to raster with direction of water velocity.",
                      default = None)
    parser.add_option(      "--reward",
                      help = "Path to numpy array (txt) with reward at each cell",
                      default = None)
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
    parser.add_option(      "--generations",     type = "int", default = 500,
                      help = "Number of optimization generations.")
    parser.add_option(      "--pool_size",       type = "int", default = 50,
                      help = "Number of individuals in optimization pool")
    parser.add_option(      "--distance_weight", type = "float", default = 0.0,
                      help = "Weight of distance attribute in fitness.")
    parser.add_option(      "--force_weight",    type = "float", default = 1.0,
                      help = "Weight of force attribute in fitness.")
    parser.add_option(     "--reward_weight",    type = "float", default = 1.0,
                      help = "Weight of reward attribute in fitness")
    parser.add_option(     "--hyperparams",
                      help = "Comma-delimited selection for solver and its options",
                      default = "pso,0.7298,2.05,2.05")
    # Get info for a cached path INSTEAD of solving
    parser.add_option(     "--statpath",
                      help = "Path to list of waypoints to print path information. Will not solve.",
                      default = None)
    # Optional, population initialization
    parser.add_option(     "--init_pop",
                      help = "Path to file with initial population paths.")

    (options, args) = parser.parse_args()

    # Environment
    regionRasterFile = options.region
    mapOutFile = options.map
    pathOutFile = options.path
    currentsRasterFile_u = options.currents_mag
    currentsRasterFile_v = options.currents_dir
    rewardGridFile = options.reward
    startPoint = (float(options.sy), float(options.sx))
    endPoint = (float(options.dy), float(options.dx))
    targetSpeed_mps = float(options.speed)
    timeOffset_s = options.time_offset
    timeInterval_s = options.time_interval

    usingWork = True if currentsRasterFile_u is not None else False
    usingReward = True if rewardGridFile is not None else False

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

    # Cached
    statPathFile = options.statpath

    # Initial population
    initPopFile = options.init_pop

    print("Using input region raster: {}".format(regionRasterFile))
    print("  Start:", startPoint)
    print("    End:", endPoint)

    # Load raster
    regionData = gdal.Open(regionRasterFile)
    regionExtent = getGridExtent(regionData)
    regionTransform = regionData.GetGeoTransform()
    grid = np.nan_to_num(regionData.GetRasterBand(1).ReadAsArray())

    # Bounds
    if options.bounds is None:
        bounds = np.array([0, regionExtent["rows"] - 1, 0, regionExtent["cols"] - 1])
    else:
        bounds = np.array(options.bounds.split(",")).astype(int)

    # Read currents rasters
    elapsedTime = 0
    bandIndex = 0
    usingCurrents = False
    currentsGrid_u = currentsGrid_v = currentsTransform_u = currentsTransform_v = None
    rewardGrid = None
    if currentsRasterFile_u is not None and currentsRasterFile_v is not None:
        bandIndex = 1

        # Load force magnitudes
        currentsData_u = gdal.Open(currentsRasterFile_u)
        currentsExtent_u = getGridExtent(currentsData_u)
        currentsTransform_u = currentsData_u.GetGeoTransform()
        currentsGrid_u = np.nan_to_num(raster2array(currentsData_u))

        # Load force directions
        currentsData_v = gdal.Open(currentsRasterFile_v)
        currentsExtent_v = getGridExtent(currentsData_v)
        currentsTransform_v = currentsData_v.GetGeoTransform()
        currentsGrid_v = np.nan_to_num(raster2array(currentsData_v))

        # Sanity check that the current mag, dir rasters match
        if currentsExtent_u["rows"] != currentsExtent_v["rows"] or \
                currentsExtent_u["cols"] != currentsExtent_v["cols"]:
            print("[-] Spatial extent mismatch between currents u, v")
            exit(-1)

        usingCurrents = True
        print("Incorporating forces into energy cost:")
        print("  u: {}".format(currentsRasterFile_u))
        print("  v: {}".format(currentsRasterFile_v))

    # Read reward matrix
    usingReward = False
    if rewardGridFile is not None:
        rewardGrid = np.loadtxt(rewardGridFile)

    # Convert latlon -> rowcol
    start = world2grid(startPoint[0], startPoint[1], regionTransform, regionExtent["rows"])
    end = world2grid(endPoint[0], endPoint[1], regionTransform, regionExtent["rows"])

    # Calculate pixel distance
    s = grid2world(0, 0, regionTransform, regionExtent["rows"])
    e = grid2world(0, regionExtent["cols"], regionTransform, regionExtent["rows"])
    dist_m = haversine(s, e) * 1000
    pixelsize_m = dist_m / regionExtent["cols"]

    if statPathFile is None:
        # Solve
        print("Begin solving")
        prob = pg.problem(solvePath(start, end, grid, targetSpeed_mps = targetSpeed_mps,
                waypoints = numWaypoints, bounds = bounds, weights = weights,
                currentsGrid_u = currentsGrid_u, currentsGrid_v = currentsGrid_v,
                currentsTransform = currentsTransform_u, regionTransform = regionTransform,
                rewardGrid = rewardGrid,
                timeIn = timeOffset_s, interval = timeInterval_s, pixelsize_m = pixelsize_m))
        algo = pg.algorithm(pg.pso(gen = generations,
                                   omega = hyperparams[1], eta1 = hyperparams[2], eta2 = hyperparams[3]))
        algo.set_verbosity(10)

        # Generate initial population
        if initPopFile is not None:
            # Load initial population from file
            pop = pg.population(prob, 0) # Ignore pool_size; filling population from file

            # Read file
            with open(initPopFile) as f:
                lines = [line.rstrip() for line in f]
                lines.reverse()

            # Process each line
            initPaths = []
            for line in lines:
                line = line.replace(";", ",").replace("(", "").replace(")", "")
                nums = [int(l) for l in line.split(",")][2:-2]
                for rep in repeat([nums[-2], nums[-1]], numWaypoints - int(len(nums) / 2)):
                    nums.extend(rep)
                initPaths.append(nums[:numWaypoints * 2])

            for ip in initPaths:
                pop.push_back(x = ip)
        else:
            # Randomly initial population
            pop = pg.population(prob, poolSize)

        # Begin solving
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

        print("Done solving with {alg}, {s} seconds".format(alg = hyperparams[0], s = t1 - t0))
    else:
        path = np.loadtxt(statPathFile, delimiter=',').astype(int)

    # Path information
    pathlen = len(path)
    work, dist, obs, reward = calcWork(path, pathlen, grid, targetSpeed_mps,
                               currentsGrid_u, currentsGrid_v, regionTransform, regionTransform,
                               rewardGrid, timeOffset_s, timeInterval_s, pixelsize_m)

    # Haversine distance
    path_distance = 0
    prev_point = path[0]
    for point in path[1:]:
        prev_latlon = grid2world(prev_point[0], prev_point[1], regionTransform, regionExtent["rows"])
        point_latlon = grid2world(point[0], point[1], regionTransform, regionExtent["rows"])
        path_distance += haversine(prev_latlon, point_latlon)
        prev_point = point
    path_duration = (path_distance * 1000 / targetSpeed_mps) / 60

    if usingCurrents:
        print("Planning results (with currents):")
    else:
        print("Planning results (no currents):")
    print('    Distance: {:.4f} km'.format(path_distance))
    print('    Duration: {:.4f} min'.format(path_duration))
    print('    Cost: {:.4f}'.format(work))
    print('    Reward: {:.4f}'.format(reward))
    #print('    Fitness: ', fitness[0])

    # Plot solution path
    lats = np.zeros(pathlen)
    lons = np.zeros(pathlen)
    for i in range(pathlen):
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
    if (rewardGridFile is not None):
        plt.imshow(rewardGrid, alpha = 0.75)
    plt.savefig(mapOutFile)
    np.savetxt(pathOutFile, path, delimiter=',')
