#!/usr/bin/python
'''
goto.py
Author: Evan Krell

A metaheuristic-based path planner to travel to a goal location
while miniziming distance, minimizing energy use, avoiding obstacles,
and maximizing target capture.
'''

def getGridExtent (data):
    # Source: gis.stackexchange.com/a/104367

    # data: a gdal object
    cols = data.RasterXSize
    rows = data.RasterYSize
    transform = data.GetGeoTransform ()

    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows

    extent = { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols  }
    return extent


# Typical modules
from math     import acos, cos, sin, ceil
from optparse import OptionParser
import pandas as pd
import os, sys
import matplotlib.pyplot  as plt
import dill as pickle
# Geographic modules
from osgeo import gdal
# Optimization modules
from PyGMO.problem import base
from PyGMO         import algorithm, island, problem, archipelago
# Conch modules
import gridUtils          as GridUtil
import plannerTools       as PlannerTools
import entityFitness      as TargetFitness
import plannerVis         as PlannerVis

# Constants
HYPERPARAMS_DEFAULT = \
    ["pso",   # Choice of optimization solver
     0.7298,  # Inertia weight
     2.05,    # Cognitive coeff.
     2.05,    # Social coeff.
    ]
# Global variables
environment = None
startPoint  = None
endPoint    = None
weights = {
        "distance" : 1,
        "obstacle" : 100,
        "current"  : 0.00001,
        "entity"   : 0.001, # 0.01
    }

def solveProblem (start, target, environment):

        # Options for all solvers
        dim  = environment['plannerGoto']['numWaypoints'] * 2
        prob = highestCurrentProblem(dim = dim)
        gen  = environment['plannerGoto']['generations']
        hyper = environment['plannerGoto']['hyperparams']

        # Need to set default params?
        if hyper is None:
            hyper = HYPERPARAMS_DEFAULT
        # Make algorithm name lower-case
        hyper[0] = hyper[0].lower()

        print("Hyperparameters:", hyper)

        # Select algorithm
        if hyper[0] == "de":
            # Differential Evolution (DE)
            try:
                algo = algorithm.de(gen,
                    f  = float(hyper[1]), # weighting factor in [0,1] (if -1 self-adapt)
                    cr = float(hyper[2]), # crossover in [0,1] (if -1 self-adapt)
                )
            except:
                algo = algorithm.de(gen)  # Default params
        elif hyper[0] == "sga":
            # Simple Genetic Algorithm (SGA)
            try:
                algo = algorithm.sga(gen,
                    r = float(hyper[1]),  # crossover factor in [0,1]
                    m = float(hyper[2]),  # mutation probability in [0,1]
                )
            except:
                algo = algorithm.sga(gen) # Default params
        elif hyper[0] == "pso":
            # Particle swarm optimization (PSO)
            try:
                algo = algorithm.pso(gen,
                    omega  = float(hyper[1]), # inertia weight in [0,1]
                    eta1   = float(hyper[2]), # cognitive component in [0,4]
                    eta2   = float(hyper[3]), # social component in [0,4]
                )
            except:
                algo = algorithm.pso(gen)     # Default params
        elif hyper[0] == "abc":
            # Artificial Bee Colony (ABC)
            try:
                algo = algorithm.bee_colony(gen,
                    limit = int(hyper[1]),       # num tries to improve before drop
                )
            except:
                algo = algorithm.bee_colony(gen) # Default params
        else:
            print("Algorithm {} not supported".format(hyper[0]))
            exit(0)

        # Apply solver
        isl = island(algo, prob, 20)# environment['plannerGoto']['individuals'])
        isl.evolve(1)

        log = []

        pathInfo = { "path"        : isl.population.champion.x,
                     "constraints" : isl.population.champion.c,
                     "fitness"     : isl.population.champion.f,
                     "start"       : (environment['vehicle']['startCoordsArchive']['row'],
                                      environment['vehicle']['startCoordsArchive']['col']),
                     "stop"        : (target['row'], target['col']),
                     "heading"     : None,
                     "distance"    : None,
                     "duration"    : None,
                     "work"        : None,
                     "coords"      : None,
                     "reward"      : None,
                     "optlog"      : log,
        }

        sp = PlannerTools.statPath (environment, isl.population.champion.x,
                start, target)

        pathInfo["heading"]  = sp[0]
        pathInfo["distance"] = sp[1]
        pathInfo["duration"] = sp[2]
        pathInfo["work"]     = sp[3]
        pathInfo["coords"]   = sp[4]
        pathInfo["reward"]   = sp[5]

        pathPD = PlannerTools.path2pandas (pathInfo)

        return (pathInfo, pathPD)

class highestCurrentProblem(base):

        def __init__(self, dim = 10): # 10 is default dims -> 5 waypoints

                global environment
                global startPoint
                global endPoint

                # Set problem dimensions to be twice the number of waypoints,
                #   to account for the x and y coordinates of each
                dim = environment['plannerGoto']['numWaypoints'] * 2

                super (highestCurrentProblem, self).__init__(dim)

                self.yStart = startPoint['row']
                self.xStart = startPoint['col']
                self.yStop  = endPoint['row']
                self.xStop  = endPoint['col']

                # Get size of pixel (resolution in meters)
                self.pixelSize_m = PlannerTools.calcPixelResolution_m (environment['region']['grid'],
                                                    environment['region']['raster'].GetGeoTransform())

                # Set problem bounds.
                upperBounds = [0] * dim
                for i in range (0, dim):
                        if (i % 2 == 0):
                                # Y value bounds
                                upperBounds[i] = environment['region']['extent']['rows'] - 1
                        else:
                                #X value bounds
                                upperBounds[i] = environment['region']['extent']['cols'] - 1

                self.set_bounds ([0] * dim, upperBounds)


        # The actual objective function, X is the N-dimension solution vector.
        def _objfun_impl (self, x):

                ###############
                # Non temporal#
                ###############

                heading = PlannerTools.pathHeading (x, self.yStart,
                        self.xStart,
                        self.yStop,
                        self.xStop)

                distance = PlannerTools.pathDistance (x,
                        self.yStart,
                        self.xStart,
                        self.yStop,
                        self.xStop,
                        environment["region"]["extent"]["rows"],
                        environment["region"]["extent"]["cols"],
                        environment['region']['grid'],
                        environment['plannerGoto']['obstacle_flag'])

                duration = PlannerTools.pathDuration (x,
                        distance,
                        environment["vehicle"]["speed"])

                reward = PlannerTools.pathReward (x,
                        self.yStart,
                        self.xStart,
                        self.yStop,
                        self.xStop,
                        environment["region"]["extent"]["rows"],
                        environment["region"]["extent"]["cols"],
                        environment["logbook"]["grid"],
                        environment['logbook'])

                ############
                # Temporal #
                ############

                work = PlannerTools.pathEnergy (x,
                        self.yStart,
                        self.xStart,
                        self.yStop,
                        self.xStop,
                        environment["region"]["extent"]["rows"],
                        environment["region"]["extent"]["cols"],
                        environment["forces"]["magnitude"]["data"],
                        environment["forces"]["direction"]["data"],
                        self.pixelSize_m,
                        distance,
                        duration,
                        heading,
                        environment["vehicle"],
                        environment["timespan"]["interval"],
                        environment["timespan"]["offset"])

                global weights
                # Combine objectives into single fitness function
                f = distance["total"]            * weights["distance"] \
                  + distance["penaltyWeighted"]  * weights["obstacle"] \
                  + work["total"]                * weights["current"] # \
                  #- reward['weightedTotal']      * weights["entity"]

                return (f, )

def driver(sPoint, ePoint, env, offset = 0):

    startPointArchive  = GridUtil.getArchiveByWorld(
        sPoint["Lat"],
        sPoint["Lon"],
        env["region"]["grid"],
        env["region"]["raster"].GetGeoTransform())

    targetPointArchive = GridUtil.getArchiveByWorld(
        ePoint["Lat"],
        ePoint["Lon"],
        env["region"]["grid"],
        env["region"]["raster"].GetGeoTransform())

    global environment
    environment = env

    # Offset the start time
    environment["timespan"]["offset"] = offset

    global weights
    if float(environment["plannerGoto"]["distanceWeight"]) >= 0:
        weights["distance"] = float(environment["plannerGoto"]["distanceWeight"])
    if float(environment["plannerGoto"]["obstacleWeight"]) >= 0:
        weights["obstacle"] = float(environment["plannerGoto"]["obstacleWeight"])
    if float(environment["plannerGoto"]["currentWeight"]) >= 0:
        weights["current"] = float(environment["plannerGoto"]["currentWeight"])
    if float(environment["plannerGoto"]["entityWeight"]) >= 0:
        weights["entity"] = float(environment["plannerGoto"]["entityWeight"])

    global startPoint
    startPoint = startPointArchive

    global endPoint
    endPoint = targetPointArchive

    (solutionPath, pathPandas) = solveProblem (startPointArchive,
                                           targetPointArchive,
                                           environment)

    solution = { "solutionPath" : solutionPath,
                 "pathPandas"   : pathPandas,
                 "startPoint"   : sPoint,
                 "endPoint"     : ePoint,
    }

    return solution, pathPandas["DURATION"].sum()


def main():
    ###########
    # Options #
    ###########
    parser = OptionParser()
    # Planning options
    parser.add_option("-a", "--start_lat",  type = "float",
        help = "Latitude of robot start position.")
    parser.add_option("-b", "--start_lon",  type = "float",
        help = "Latitude of robot start position.")
    parser.add_option("-c", "--target_lat", type = "float",
        help = "Latitude of robot start position.")
    parser.add_option("-d", "--target_lon", type = "float",
        help = "Latitude of robot start position.")
    parser.add_option("-s", "--speed",      type = "float",    default = 154,
        help = "Speed of vehicle.")
    # Program options
    parser.add_option("-F", "--figure_out",                    default = None,
        help = "Output file for map image with path.")
    parser.add_option("-T", "--table_out",                     default = None,
        help = "Output file for path csv.")
    parser.add_option("-P", "--pickle_out",                    default = None,
        help = "Output file for pickled path results.")
    # Data source options
    parser.add_option("-r", "--region_file",
        help = "Region as occupancy grid (Numpy-compatible TXT or GeoTIFF).")
    parser.add_option("-m", "--magnitude_force_file",
        help = "Magnitude of force (Numpy-compatible TXT or GeoTIFF).")
    parser.add_option("-z", "--direction_force_file",
        help = "Direction of force (Numpy-compatible TXT or GeoTIFF).")
    parser.add_option(      "--obstacle_flag",   type = "int", default = 1,
        help = "Integer flag to indicate obstacle in occupancy grid.")
    # Optimization parameter options
    parser.add_option("-n", "--num_waypoints",   type = "int", default = 5,
        help = "Number of solution waypoints to generate.")
    parser.add_option("-g", "--generations",     type = "int", default = 1000,
        help = "Number of optimization generations.")
    parser.add_option("-p", "--pool_size",       type = "int", default = 100,
        help = "Number of individuals in optimization pool")
    parser.add_option(      "--dist_weight",     type = "int", default = 1,
        help = "Weight of distance attribute in fitness.")
    parser.add_option(      "--obstacle_weight", type = "int", default = 100,
        help = "Weight of obstacle attribute in fitness.")
    parser.add_option(      "--force_weight",    type = "int", default = 0.00001,
        help = "Weight of force attribute in fitness.")
    parser.add_option(     "--entity_weight",    type = "int", default = 0.001,
        help = "Weight of entity reward attribute in fitness")
    parser.add_option(     "--hyperparams",
        help = "Comma-delimited selection for solver and its options")
    # Cached path options
    parser.add_option(     "--cached",  action = "store_true", default = False,
        help = "Will print the fitness of a cached path. Requires ('--path_file').e")
    parser.add_option(     "--path_file",
        help = "Path as file where each line is a waypoint.")
    parser.add_option(     "--rowcol", action = "store_true", default = False,
        help = "Will treat waypoints in path file ('--path_file') as row, col. Default is lat, lon.")
    (options, args) = parser.parse_args()

    #########
    # Setup #
    #########
    environment = dict()
    raster = gdal.Open(options.region_file)
    grid = raster.GetRasterBand(1).ReadAsArray()
    extent = getGridExtent(raster)
    magnitudeForceRaster = gdal.Open(options.magnitude_force_file)
    directionForceRaster = gdal.Open(options.direction_force_file)

    if options.hyperparams is not None:
        hyperparams = options.hyperparams.split(',')
    else:
        hyperparams = HYPERPARAMS_DEFAULT

    environment["region"] = {
            "file"   : options.region_file,
            "raster" : raster,
            "grid"   : grid,
            "extent" : extent,
        }
    environment["timespan"] = {
            "interval" : 3000,
            "offset"   : 0,
        }
    environment["forces"] = {
            "magnitude" : {
                    "file"   : options.magnitude_force_file,
                    "raster" : magnitudeForceRaster,
                    "data"   : magnitudeForceRaster.ReadAsArray(),
                },
            "direction" : {
                    "file"   : options.direction_force_file,
                    "raster" : directionForceRaster,
                    "data"   : directionForceRaster.ReadAsArray(),
                }
        }
    environment["vehicle"] = {
            "startCoordinates_lat" : float(options.start_lat),
            "startCoordinates_lon" : float(options.start_lon),
            "startCoords"          : (options.start_lon, options.start_lat),
            "startCoordsArchive"   : GridUtil.getArchiveByWorld(
                                        options.start_lat, options.start_lon,
                                        grid, raster.GetGeoTransform()),
            "speed"                : options.speed,
        }
    environment["plannerGoto"] = {
            "numWaypoints"   : options.num_waypoints,
            "obstacle_flag"  : options.obstacle_flag,
            "generations"    : options.generations,
            "individuals"    : options.pool_size,
            "distanceWeight" : options.dist_weight,
            "obstacleWeight" : options.obstacle_weight,
            "currentWeight"  : options.force_weight,
            "entityWeight"   : options.entity_weight,
            "hyperparams"    : hyperparams,
        }
    environment["logbook"] = {
            "grid"    : None,
            "tWeight" : None,
        }

    start  = {"Lat" : options.start_lat,  "Lon" : options.start_lon}
    target = {"Lat" : options.target_lat, "Lon" : options.target_lon}

    startArchive = GridUtil.getArchiveByWorld(start["Lat"], start["Lon"],
        environment["region"]["grid"],
        environment["region"]["raster"].GetGeoTransform())
    targetArchive = GridUtil.getArchiveByWorld(target["Lat"], target["Lon"],
        environment["region"]["grid"],
        environment["region"]["raster"].GetGeoTransform())

    if options.cached is True:
        #################
        # Load Solution #
        #################
        with open(options.path_file) as f:
            path = f.read().splitlines()

        path = [p.split(',') for p in path]
        for p in path:
            p[0] = int(p[0])
            p[1] = int(p[1])

        path2 = []
        for p in path:
            path2.append(p[0])
            path2.append(p[1])

        heading, distance, duration, work, coord, reward = PlannerTools.statPath (environment, path2,
                targetArchive, startArchive)
        f = distance["total"]            * weights["distance"]  \
          + distance["penaltyWeighted"]  * weights["obstacle"]  \
          + work["total"]                * weights["current"] * 1000  \
          - reward['weightedTotal']      * weights["entity"]

        print (path)
        map_ax = PlannerVis.makeGotoMap_simple(environment["region"]["raster"],
            environment["region"]["file"], path, 5, options.figure_out)
        plt.show()

        print("work", work)
        print("f", f)

    #######
    # Run #
    #######
    solution, solutionDuration = driver(start, target, environment)

    #############
    # Visualize #
    #############
    print("Start:  (lat {}, lon {})".format(
        solution["startPoint"]["Lat"], solution["startPoint"]["Lon"]))
    print("Target: (lat {}, lon {})".format(
        solution["endPoint"]["Lat"], solution["endPoint"]["Lon"]))
    pd.set_option('display.precision', 3)
    print(solution["pathPandas"])
    print("Fitness: {}".format(solution["solutionPath"]["fitness"][0]))

    # Map (and save) the solution
    map_ax = PlannerVis.makeGotoMap(environment["region"]["raster"],
        environment["region"]["file"], [solution], 5, options.figure_out)

    #################
    # Store results #
    #################
    # Save pandas table as csv
    if options.table_out is not None:
        solution["pathPandas"].to_csv(options.table_out)

    # Save pickled solution data
    if options.pickle_out is not None:
        with open(options.pickle_out, 'wb') as outfile:
            pickle.dump(solution, outfile, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
