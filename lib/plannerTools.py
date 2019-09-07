'''
PlannerTools.py
Author: Evan Krell

Various utility functions for the planner modules.
'''

from math import acos, cos, sin, ceil
from numpy import inf

import pandas as pd
import numpy as np
import yaml as yaml
import os
import sys

import gridUtils as GridUtil
import rasterSetInterface as rsi
import entityFitness as AnalystTools

def idx2rowcol(idx, ncol):
    row =  int(idx / ncol)
    col =  int(idx % ncol)
    return row, col

def rowcol2idx(row, col, ncol):
    idx = (row * ncol) + col
    return idx

def euclideanDistance (xi, yi, xj, yj):
        """Compute the Euclidean distance between two 2D points, i and j.

        Args:
                xi (double): The x-coordinate of point i.
                yi (double): The y-coordinate of point i.
                xj (double): The x-coordinate of point j.
                yj (double): The y-coordinate of point j.

        Returns:
                double: The Euclidian distance between i and j.
        """

        return pow ( (pow (xi - xj, 2) + pow (yi - yj, 2)) , 0.5)

def getBandIdxByElapsedTime (elapsedTime, bandInterval):
        if elapsedTime == 0:
            return 1
        idx = int (ceil ((elapsedTime * 1.0) / bandInterval))
        return idx

def angle (vecA, vecB):
        dotProd = sum( [vecA[i] * vecB[i] for i in range(len(vecA))] )
        magA = pow (sum( [vecA[i] * vecA[i] for i in range(len(vecA))] ), 0.5)
        magB = pow (sum( [vecB[i] * vecB[i] for i in range(len(vecB))] ), 0.5)

        if (magA * magB == 0):
                return 0.0

        cosTheta = dotProd / (magA * magB)
        theta_rad = acos (cosTheta)
        return theta_rad

def vSub (vecA, vecB):
        vecS = (vecB[0] - vecA[0], vecB[1] - vecA[1])
        return vecS

def getPathCoords (path, start, stop, grid, transform):
    # path = [ y1, x1, y2, x2, ....., yn, xn ]

        coordArchive = []
        coordArchive.append (GridUtil.getArchiveByGrid (start[0], start[1], grid, transform))
        iterations = len (path) / 2
        count = 0

        for i in range (0, iterations * 2, 2):
                coordArchive.append (GridUtil.getArchiveByGrid (path[i], path[i + 1], grid, transform))
        coordArchive.append (GridUtil.getArchiveByGrid (stop[0], stop[1], grid, transform))
        return coordArchive

def calcPixelResolution_m (grid, transform):
        import geopy.distance

        i = 10
        # Pixel 1: [0, 0]
        p1 = GridUtil.getArchiveByGrid (0, 0, grid, transform)
        # Pixel 2: [0, i]
        p2 = GridUtil.getArchiveByGrid (0, i, grid, transform)
        # Distance between them in kilometers
        distance = geopy.distance.vincenty( (p1['lon'], p1['lat']), (p2['lon'], p2['lat'])).m
        # Pixel resolution
        res = distance / i

        return res

def writePandas (df, filename):
	df.to_csv (filename, sep= ',', index=False)

def initTargetsTableByFile(targetsTableFile):
    import pandas as pd
    targetsTable = pd.read_csv(targetsTableFile)
    targetsTable = targetsTable.set_index('ID')
    return targetsTable

def countLinePenalties (y0, x0, y1, x1, grid, yLimit, xLimit, penaltyCase):
        """ Visits each element on a 2D grid along a line from point 0 to point 1
        , and counts the number of times that the element is the penaltyCase.

        Args:
                x0 (double): The x-coordinate of point 0.
                y0 (double): The y-coordinate of point 0.
                y1 (double): The x-coordinate of point 1.
                y2 (double): The y-coordinate of point 1.
                grid (int[][]): The grid whose values are being checked
                        for penaltyCase along a line from point 0 to point 1.
                xLimit (int): The length of the x-axis.
                yLimit (int): The length of the y-axis.
                penaltyCase (int): A numeric flag that is being counted.
        """
        from bresenham import bresenham
        hits = 0
        y0 = int (ceil (y0))
        x0 = int (ceil (x0))
        y1 = int (ceil (y1))
        x1 = int (ceil (x1))
        b = list(bresenham(x0, y0, x1, y1))
        for p in b:
                if grid[p[1]][p[0]] == penaltyCase:
                        hits = hits + 1
        return hits

# Path Info #
def line (y0, x0, y1, x1, grid, yLimit, xLimit, penaltyCase):
    """ Visits each element on a 2D grid along a line from point 0 to point 1
    , and counts the number of times that the element is the penaltyCase.

    Args:
            x0 (double): The x-coordinate of point 0.
            y0 (double): The y-coordinate of point 0.
            y1 (double): The x-coordinate of point 1.
            y2 (double): The y-coordinate of point 1.
            grid (int[][]): The grid whose values are being checked
                    for penaltyCase along a line from point 0 to point 1.
            xLimit (int): The length of the x-axis.
            yLimit (int): The length of the y-axis.
            penaltyCase (int): A numeric flag that is being counted.
    """
    from bresenham import bresenham
    hits = 0
    y0 = int (ceil (y0))
    x0 = int (ceil (x0))
    y1 = int (ceil (y1))
    x1 = int (ceil (x1))

    b = list(bresenham(x0, y0, x1, y1))

    for p in b:
            if grid[p[1]][p[0]] == penaltyCase:
                    hits = hits + 1
    return hits


def pathCountPenalties(path, yStart, xStart, yStop, xStop, yLimit, xLimit, grid, penaltyCase, distance):
    """ Calculates the penalty for each path segment.
    The penalty is the number of times that an element along a line segment is equal to the penaltyCase.

    Args:
            path ((double,double)[]): Sequence of interior waypoints as (x, y) pairs.
            xStart (double): The x-coordinate of the path's start location.
            yStart (double): The y-coordinate of the path's start location.
            xStop (double): The x-coordinate of the path's stop location.
            yStop (double): The y-coordinate of the path's stop location.
            xLimit (int): The length of the x-axis
            yLimit (int): The length 0f the y-axis
            grid (int[][]): The grid whose values are being checked
                    for penaltyCase along each line segment.
            penaltyCase (int): A numeric flag that is being counted.
            distance ({ 'segmentPenalties':(double)[] }): Dict where penalty information is stored.

    Returns:
            dict: containing penalty information, both per-segment and a summation.

            {'segmentPenalties':(double)[]: List of penalty count for each segment.
             'penaltyCase':(double): Summation of penalty counts.}
    """
    penaltyCount = 0
    count = 0

    yi = path[0]
    xi = path[1]
    yj = None
    xj = None

    distance["segmentPenalties"].append (line (yStart, xStart, yi, xi, grid, yLimit, xLimit, penaltyCase))

    count = 2
    iterations = len (path) / 2 - 1
    for i in range (0, iterations):
        yj = path[count]
        count = count + 1
        xj = path[count]
        count = count + 1
        distance["segmentPenalties"].append (line(yi, xi, yj, xj, grid, yLimit, xLimit, penaltyCase))
        yi = yj
        xi = xj

    distance["segmentPenalties"].append (line (yi, xi, yStop, xStop, grid, yLimit, xLimit, penaltyCase))
    distance["penalty"] = sum (distance["segmentPenalties"])
    return distance


def pathPenalty(path, yStart, xStart, yStop, xStop, yLimit, xLimit, grid, penaltyCase, distance):
    """ Calculates the penalty of the entire path.
    The penalty is the number of times that an element along the path is equal to the penaltyCase.

    Args:
            path ((double,double)[]): Sequence of interior waypoints as (x, y) pairs.
            xStart (double): The x-coordinate of the path's start location.
            yStart (double): The y-coordinate of the path's start location.
            xStop (double): The x-coordinate of the path's stop location.
            yStop (double): The y-coordinate of the path's stop location.
            xLimit (int): The length of the x-axis
            yLimit (int): The length 0f the y-axis
            grid (int[][]): The grid whose values are being checked
                    for penaltyCase along each line segment.
            penaltyCase (int): A numeric flag that is being counted.
            distance ({ 'segmentPenalties':(double)[] }): Dict where penalty information is stored.

    Returns:
            dict: containing penalty information, both per-segment and a summation. Also a weighted summation.

            {'segmentPenalties':(double)[]: List of penalty count for each segment.
             'penaltyCase':(double): Summation of penalty counts.
             'penaltyWeighted':(double): Weighted summation of penalty counts.}
    """

    obstructions = pathCountPenalties(path, yStart, xStart, yStop,
                   xStop, yLimit, xLimit, grid, penaltyCase, distance)
    distance["penaltyWeighted"] = pow (distance["penalty"], 2)
    return distance


def pathHeading (path, yStart, xStart, yStop, xStop):
    # path = [ y1, x1, y2, x2, ....., yn, xn ]

    headings = []

    yi = path[0]
    xi = path[1]
    yj = None
    xj = None

    v = vSub ( (yStart, xStart), (yi, xi) )
    a = angle (v, (1, 0))
    headings.append (a)

    count = 2
    iterations = len (path) / 2 - 1
    for i in range (0, iterations):
         yj = path[count]
         count = count + 1
         xj = path[count]
         count = count + 1

         v = vSub ( (yi, xi), (yj, xj) )
         a = angle (v, (1, 0))
         headings.append (a)

         yi = yj
         xi = xj

    v = vSub ( (yj, xj), (yStop, xStop) )
    a = angle (v, (1, 0))
    headings.append (a)

    return headings


def pathDistance (path, yStart, xStart, yStop, xStop, yLimit, xLimit, grid, penaltyCase):
    # path = [ y1, x1, y2, x2, ....., yn, xn ]

    distance = {"total" : 0.0, "penalty" : 0.0, "penaltyWeighted" : 0.0, "segments" : [], "segmentPenalties" : []}

    yi = path[0]
    xi = path[1]
    yj = None
    xj = None

    distance["segments"].append (euclideanDistance (yStart, xStart, yi, xi))

    count = 2
    iterations = len (path) / 2 - 1
    for i in range (0, iterations):
         yj = path[count]
         count = count + 1
         xj = path[count]
         count = count + 1
         distance["segments"].append (euclideanDistance (yi, xi, yj, xj))
         yi = yj
         xi = xj

    distance["segments"].append (euclideanDistance (yi, xi, yStop, xStop))
    distance["total"] = sum (distance["segments"])

    pathPenalty (path, yStart, xStart, yStop, xStop, yLimit, xLimit, grid, penaltyCase, distance)
    return distance

def calcDuration (distance, speed):
                return distance / speed

def pathDuration (path, distance, speed):
    # path = [ y1, x1, y2, x2, ....., yn, xn ]


    duration = {"total" : 0.0, "segments" : []}

    iterations = len (path) / 2 + 1
    for i in range (0, iterations):
        duration["segments"].append(calcDuration (distance["segments"][i], speed))

    duration["total"] = sum (duration["segments"])
    return duration

def getWorkAlongLine (y0, x0, y1, x1, gridM, gridD, yLimit, xLimit, targetVelocity, pixelResolution, heading):
    # Calculates "work" relative to velocity.

    from bresenham import bresenham

    def calcVelocityDiff (vecA, vecB):
        #
        # vecA: Required velocity vector in (magnitude, direction)
        # vecB: Applied velocity vector in (magnitude, direction)

        # Break both angles into components
        magA = vecA[0]
        dirA = vecA[1]
        magB = vecB[0]
        dirB = vecB[1]

        yA = magA * cos (dirA)
        xA = magA * sin (dirA)
        yB = magB * cos (dirB)
        xB = magB * sin (dirB)

        yDeltaVelocity = xA - xB
        xDeltaVelocity = yA - yB

        dirDeltaVelocity = angle ( (yDeltaVelocity, xDeltaVelocity), (1, 0) )
        magDeltaVelocity = pow ( (yDeltaVelocity * yDeltaVelocity + xDeltaVelocity * xDeltaVelocity), 0.5 )

        return (magDeltaVelocity, dirDeltaVelocity, yDeltaVelocity, xDeltaVelocity)

    def calcRelativeWork (speed, distance):
        # speed is velocity magnitude
        return speed * distance

    work = 0
    y0 = int (ceil (y0))
    x0 = int (ceil (x0))
    y1 = int (ceil (y1))
    x1 = int (ceil (x1))
    b = list(bresenham(x0, y0, x1, y1))
    for p in b:
        # Calculate work
        appliedVelocity = calcVelocityDiff ( (targetVelocity, heading) ,
                          (gridM[p[1]][p[0]], gridD[p[1]][p[0]]))
        work = work + calcRelativeWork (appliedVelocity[0], pixelResolution)
    return work


def pathEnergy (path, yStart, xStart, yStop, xStop,
    yLimit, xLimit, gridMagnitude, gridDirection,
    pixelResolution, distance, duration, headings,
   USV, timeInterval, offset = 0):

    # path = [ y1, x1, y2, x2, ....., yn, xn ]

    work = {"total" : 0.0, "segments" : []}

    yi = path[0]
    xi = path[1]
    yj = None
    xj = None

    ii = 0

    # Duration = 0 -> band index = 1
    d        = 0
    bandIdx  = 1
    bandIdxC = bandIdx
    # Init grids using band
    gridM = np.flipud(gridMagnitude.GetRasterBand(bandIdx).ReadAsArray())
    gridD = np.flipud(gridDirection.GetRasterBand(bandIdx).ReadAsArray())

    # First segment (between start and waypopint one)
    work["segments"].append (getWorkAlongLine (yStart, xStart, yi, xi,
        gridM, gridD, yLimit, xLimit, USV["speed"], pixelResolution, headings[ii]))
    # All segments between waypoints
    ii = ii + 1
    count = 2
    iterations = len (path) / 2 - 1
    for i in range (0, iterations):
         yj = path[count]
         count = count + 1
         xj = path[count]
         count = count + 1

         # Update bandIdx
         d = d + duration['segments'][ii]
         bandIdx = getBandIdxByElapsedTime (d + offset, timeInterval)
         # Only update grids if new bandIdx
         if bandIdx != bandIdxC:
             bandIdxC = bandIdx
             gridM = np.flipud(gridMagnitude.GetRasterBand(bandIdx).ReadAsArray())
             gridD = np.flipud(gridDirection.GetRasterBand(bandIdx).ReadAsArray())

         work["segments"].append (getWorkAlongLine (yi, xi, yj, xj,
             gridM, gridD, yLimit, xLimit, USV["speed"], pixelResolution, headings[ii]))

         ii = ii + 1
         yi = yj
         xi = xj

    # Final segment (between waypoint last and goal)
    d = d + duration['segments'][ii]
    bandIdx = getBandIdxByElapsedTime (d, timeInterval)
    # Only update grids if new bandIdx
    if bandIdx != bandIdxC:
        #print("update", bandIdx, bandIdxC)
        bandIdxC = bandIdx
        gridM = gridMagnitude.GetRasterBand(bandIdx).ReadAsArray()
        gridD = gridDirection.GetRasterBand(bandIdx).ReadAsArray()

    work["segments"].append (getWorkAlongLine (yj, xj, yStop, xStop,
        gridM, gridD, yLimit, xLimit, USV["speed"], pixelResolution, headings[ii]))

    work["total"] = sum (work["segments"])

    return work


def pathReward (path, yStart, xStart, yStop, xStop, yLimit, xLimit,
    gridTargets, logbook):
    # path = [ y1, x1, y2, x2, ....., yn, xn ]
    if gridTargets is not None:
        reward = AnalystTools.rewardLine (path, yStart, xStart, yStop, xStop,
            yLimit, xLimit, gridTargets, logbook)
        reward['weightedTotal'] = reward['total'] * logbook['tWeight']
    else:
        reward = { "total" : 0, "weightedTotal" : 0, "segments" : [] }
    return reward


def getPathLongForm(pathSimple):
    # Convert from
    # pathSimple = [ (y1, x1), (y2, x2), ... (yn-1, xn-1), (yn, xn) ]
    # to
    # path = [ (y2, x2), ... (yn-1, xn-1) ] ; start = (y1, x1) ; target = (yn, xn)
    # In this version, the path is the sequence _between_ start and stop

    path = []

    # Path start
    start =  { 'row' : pathSimple[0]["row"],
               'col' : pathSimple[0]["col"],
    }
    # Path end
    target = { 'row' : pathSimple[-1]["row"],
               'col' : pathSimple[-1]["col"],
    }

    # Interior of path
    pathSimpleInterior = pathSimple[1:-1]
    for psi in pathSimpleInterior:
        path.append(psi["row"])
        path.append(psi["col"])

    return path, start, target


def statPath (environment, path, start, target, offset = 0):

    yStart = int(start["row"])
    xStart = int(start["col"])
    yStop  = int(target["row"])
    xStop  = int(target["col"])

    print(yStart)

    # Need grid resolution to determine actual distance and energy
    pixelSize_m = calcPixelResolution_m(
        environment['region']['grid'], environment['region']['raster'].GetGeoTransform())

    path = [int(p) for p in path]

    # Get coordinates
    coord = getPathCoords (path,
        (yStart, xStart),
        (yStop, xStop),
        environment['region']['grid'],
        environment['region']['raster'].GetGeoTransform ())

    # Get headings
    heading = pathHeading (path,
        yStart,
        xStart,
        yStop,
        xStop)

    # Distance travelled over entire path
    distance = pathDistance (path,
        yStart,
        xStart,
        yStop,
        xStop,
        environment['region']['extent']['rows'],
        environment['region']['extent']['cols'],
        environment['region']['grid'],
        environment['plannerGoto']['obstacle_flag'])

    duration = pathDuration (path,
        distance,
        environment['vehicle']['speed'])

    work = pathEnergy (path,
        yStart,
        xStart,
        yStop,
        xStop,
        environment["region"]["extent"]["rows"],
        environment["region"]["extent"]["cols"],
        environment["forces"]["magnitude"]["raster"],
        environment["forces"]["direction"]["raster"],
        pixelSize_m,
        distance,
        duration,
        heading,
        environment["vehicle"],
        environment["timespan"]["interval"],
        offset)

    reward = pathReward(path,
        yStart,
        xStart,
        yStop,
        xStop,
        environment["region"]["extent"]["rows"],
        environment["region"]["extent"]["cols"],
        environment["logbook"]["grid"],
        environment["logbook"])

    return (heading, distance, duration, work, coord, reward)



def path2pandas(pathInfo):
    # pathInfo:
    # Init df
    pathDF = pd.DataFrame ()

    # Break path into x and y components
    yInner = pd.Series (pathInfo["path"][::2])
    xInner = pd.Series (pathInfo["path"][1::2])
    y1 = pd.Series ([pathInfo['start'][0]])
    y2 = pd.Series ([pathInfo['stop'][0]])
    x1 = pd.Series ([pathInfo['start'][1]])
    x2 = pd.Series ([pathInfo['stop'][1]])
    y = y1
    y = y.append (yInner)
    y = y.append (y2)
    x = x1
    x = x.append (xInner)
    x = x.append (x2)
    y.index = range(len(y.index))
    x.index = range(len(x.index))

    heading = pd.Series ([0])
    heading = heading.append (pd.Series (pathInfo['heading']))
    heading.index = range(len(heading.index))

    distance = pd.Series ([0])
    distance = distance.append (pd.Series (pathInfo['distance']['segments']))
    distanceAccum = distance.cumsum ()
    distance.index = range(len(distance.index))
    distanceAccum.index = range(len(distanceAccum.index))

    duration = pd.Series ([0])
    duration = duration.append (pd.Series (pathInfo['duration']['segments']))
    durationAccum = duration.cumsum ()
    duration.index = range(len(duration.index))
    durationAccum.index = range(len(durationAccum.index))

    work = pd.Series ([0])
    work = work.append (pd.Series (pathInfo['work']['segments']))
    workAccum = work.cumsum ()
    work.index = range(len(work.index))
    workAccum.index = range(len(workAccum.index))

    reward = pd.Series ([0])
    reward = reward.append (pd.Series (pathInfo['reward']['segments']))
    rewardAccum = reward.cumsum ()
    reward.index = range(len(reward.index))
    rewardAccum.index = range(len(rewardAccum.index))


    lat = [t["lat"] for t in pathInfo["coords"]]
    lon = [t["lon"] for t in pathInfo["coords"]]
    lat = pd.Series (lat)
    lat.index = range (len(lat.index))
    lon= pd.Series (lon)
    lon.index = range (len(lon.index))

    # Combine into data frame
    pathDF['X']              = x
    pathDF['Y']              = y
    pathDF['LONGITUDE']      = lon
    pathDF['LATITUDE']       = lat
    pathDF['HEADING']        = heading
    pathDF['DISTANCE']       = distance
    pathDF['DISTANCE_ACCUM'] = distanceAccum
    pathDF['DURATION']       = duration
    pathDF['DURATION_ACCUM'] = durationAccum
    pathDF['WORK']           = work
    pathDF['WORK_ACCUM']     = workAccum
    pathDF['REWARD']         = reward
    pathDF['REWARD_ACCUM']   = rewardAccum

    return pathDF



def pathPandasConcat(resultSequenceLog):
    IDs = range(len(resultSequenceLog))

    extendedPandas = []
    for ID in IDs:
        IDpanda = resultSequenceLog[ID]["pathPandas"].copy()
        IDpanda.insert(0, 'SEGMENT', range(IDpanda.shape[0]))
        IDpanda.insert(0, 'TYPE', resultSequenceLog[ID]["type"])
        IDpanda.insert(0, 'RESULT',  ID)
        extendedPandas.append(IDpanda)

    pathConcat = pd.concat(extendedPandas, ignore_index=True)

    # Replace existing accum columns,
    # since they are relative to each segment
    distanceAccum            = pathConcat["DISTANCE"].cumsum()
    distanceAccum.index      = range(len(distanceAccum.index))
    pathConcat["DISTANCE_ACCUM"] = distanceAccum
    durationAccum            = pathConcat["DURATION"].cumsum()
    durationAccum.index      = range(len(durationAccum.index))
    pathConcat["DURATION_ACCUM"] = durationAccum
    workAccum                = pathConcat["WORK"].cumsum()
    workAccum.index          = range(len(workAccum.index))
    pathConcat["WORK_ACCUM"] = workAccum
    rewardAccum                = pathConcat["REWARD"].cumsum()
    rewardAccum.index          = range(len(rewardAccum.index))
    pathConcat["REWARD_ACCUM"] = rewardAccum


    return pathConcat

def pathSequenceSummary2pandas(resultSequenceLog):
    testnode = resultSequenceLog[0]
    if ("distance" not in testnode) or ("duration" not in testnode) or ("work" not in testnode):
        return pd.DataFrame()

    segment     = []
    plannertype = []
    startLat    = []
    startLon    = []
    endLat      = []
    endLon      = []
    distance    = []
    duration    = []
    work        = []
    reward      = []

    for r in resultSequenceLog:
        segment.append(r["seqnum"])
        plannertype.append(r["type"])
        startLat.append(float(r["start"]["Lat"]))
        startLon.append(float(r["start"]["Lon"]))
        endLat.append(float(r["end"]["Lat"]))
        endLon.append(float(r["end"]["Lon"]))
        distance.append(r["distance"])
        duration.append(r["duration"])
        work.append(r["work"])
        reward.append(r["reward"])

    pathSequenceLogPandas = pd.DataFrame(
        { 'SEGMENT'    : segment,
           'TYPE'      : plannertype,
           'START_LAT' : startLat,
           'START_LON' : startLon,
           'END_LAT'   : endLat,
           'END_LON'   : endLon,
           'DISTANCE'  : distance,
           'DURATION'  : duration,
           'WORK'      : work,
           'REWARD'    : reward,
        })


    return pathSequenceLogPandas
