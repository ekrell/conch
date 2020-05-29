from pylab import *
import yaml
import netCDF4
import matplotlib.tri as Tri
import matplotlib.pyplot as plt
import netCDF4
import pandas as pd
import datetime as dt
from datetime import date, datetime, timedelta
from dateutil.rrule import rrule, DAILY
from osgeo import gdal
from math import acos, cos, sin
import time
import os
import sys

from scipy.interpolate import griddata
import georaster
from mpl_toolkits.basemap import Basemap

import gridUtils as GridUtil
import entityFitness as TargetFitness

def components2MagDir (vecA, vecB):
	dotProd = sum( [vecA[i] * vecB[i] for i in range(len(vecA))] )
	magA = pow (sum( [vecA[i] * vecA[i] for i in range(len(vecA))] ), 0.5)
	magB = pow (sum( [vecB[i] * vecB[i] for i in range(len(vecB))] ), 0.5)

	if (magA * magB < 0.001):
		return (0.0, 1)

	cosTheta = dotProd / (magA * magB)

        if ( abs (cosTheta) >= 1):
               print (vecA)
               return (0.0, 1)

	theta_rad = acos (cosTheta)

        # This part is repeditive because of debugging..
	magnitude = pow (vecA[0] * vecA[0] + vecA[1] * vecA[1],  0.5)
        direction = theta_rad

        return magnitude, direction

def getMagDir (u, v):
	c = [ components2MagDir(v, (1, 0)) for v in zip (u, v)]
	return c

def getGridExtent (data):
	# Sources:
	#	https://gis.stackexchange.com/a/104367

	# data: a gdal object
	cols = data.RasterXSize
	rows = data.RasterYSize
	transform = data.GetGeoTransform ()

	minx = transform[0]
	maxy = transform[3]
	maxx = minx + transform[1] * cols
	miny = maxy + transform[5] * rows

	extent = { 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'rows' : rows, 'cols' : cols }
	return extent


def getNcByTime (nc, time, extent):
	# Sources:
	#    https://stackoverflow.com/a/29136166

	itime = netCDF4.date2index (time, nc.variables['time'], select = 'nearest')

	# Get latitude and longitude values
	lat = nc.variables['lat'][:]
	lon = nc.variables['lon'][:]
        latc = nc.variables['latc'][:]
        lonc = nc.variables['lonc'][:]

        # Find velocity points in bounding box
        ind = np.argwhere((lonc >= extent["minx"]) & (lonc <= extent["maxx"]) & (latc >= extent["miny"]) & (latc <= extent["maxy"]))
        subsample=3
        np.random.shuffle(ind)
        Nvec = int(len(ind) / subsample)
        idv = ind[:Nvec]
        #idv = ind

	# Get water currents as u and v component variables
	u = nc.variables['u'][itime, 0, :]
	v = nc.variables['v'][itime, 0, :]

        slatc = latc[idv]
        slonc = lonc[idv]
        su    = u[idv]
        sv    = v[idv]

        slatc = [latc[0] for latc in slatc]
        slonc = [lonc[0] for lonc in slonc]
        su    = [u[0]    for u    in su]
        sv    = [v[0]    for v    in sv]

        points = pd.DataFrame(list(zip(slatc, slonc, su, sv)), columns = ['latc', 'lonc', 'u', 'v'])

	points = points.loc[points['latc'] >= extent['miny']]
	points = points.loc[points['latc'] <= extent['maxy']]
	points = points.loc[points['lonc'] >= extent['minx']]
	points = points.loc[points['lonc'] <= extent['maxx']]

	points = points.sample(frac=0.5, replace=False)
	return points

def interpolate (y, x, z, gridy, gridx):
	# Source: https://gis.stackexchange.com/a/150881
	from scipy import interpolate

	f = interpolate.Rbf(x, y, z, function='linear')
	gridz = f (gridx, gridy)

	return gridz

def array2raster (grid, extent, rasterFile, baseRegion):
	# Source:
	# 	https://gis.stackexchange.com/a/150881

	drv = gdal.GetDriverByName ('GTiff')
	ds = drv.Create (rasterFile, extent['cols'], extent['rows'], 1 ,gdal.GDT_Float32)
	band = ds.GetRasterBand (1)
	band.SetNoDataValue (-3e30)
	ds.SetGeoTransform (baseRegion.GetGeoTransform ())
	ds.SetProjection (baseRegion.GetProjection ())
	band.WriteArray (grid)
	return ds

def initRaster (baseRegion, extent, rasterFile, numBands):
	drv = gdal.GetDriverByName ('GTiff')
	ds = drv.Create (rasterFile, extent['cols'], extent['rows'], numBands, gdal.GDT_Float32)
	ds.SetGeoTransform (baseRegion.GetGeoTransform ())
	ds.SetProjection (baseRegion.GetProjection ())
	return ds

def addRasterBand (grid, gdalRaster, i):

	band = gdalRaster.GetRasterBand (i)
	band.SetNoDataValue (0)
	band.WriteArray (grid)

def raster2plot (gRaster):
	# Plots a gdal raster with matplotlib
	# Returns the plot, rather than showing it

	return None

def getNcByDuration (sourceNC, extent, start, stop, interval_min):
	pointsSeries = []
	# For each time stamp...
	for time in rrule (MINUTELY, dtstart=start, until=stop, interval = interval_min):
		pointsSeries.append (getNcByTime (sourceNC, time, extent))
	return pointsSeries

def initialize (regionFile, currentsMagFile, currentsDirFile,
	ncURL, startDate, days, interval_min):

	# Use start date and duration to determine stop date
	stopDate = startDate + timedelta (days = days)

	# Load the region of interest raster
	regionData = gdal.Open (regionFile)
	regionExtent = getGridExtent (regionData)

	# Init connection to DAP data source
	# Open as NetCDF
	sourceNC = None# netCDF4.Dataset (ncURL)

	params = {
		'region' : {
			'file' : regionFile,
			'raster' : regionData,
			'grid' : regionData.GetRasterBand (1).ReadAsArray(),
			'extent' : regionExtent },
		'nc' : {
			'url' : ncURL,
			'data' : sourceNC },
		'timespan' : {
			'start' : startDate,
			'stop' : stopDate,
			'interval' : interval_min },
		'currents' : {
			'magnitude' : {
				'file'   : currentsMagFile,
				'raster' : None },
			'direction' : {
				'file'   : currentsDirFile,
				'raster' : None } } }

	# Can save the grid for debugging
	#np.savetxt('trash.out', params["region"]["grid"], delimiter='', fmt='%i')

	return params

def initializeTargets (targetsFile, targetsTable, targetsWeights):
	# A separate function since targets are optional

	if targetsFile == "None":
		targets = { 'file' : None, 'raster' : None, 'grid' : None, 'rasters' : None }
	else:
		targetsData = gdal.Open (targetsFile)
		targetsGrid = targetsData.GetRasterBand (1).ReadAsArray()

		targets = {
			'file' : targetsFile,
			'raster' : targetsData,
			'grid' : targetsGrid }

        targets["table"]   = targetsTable
        targets["weights"] = TargetFitness.initLogbookByFile(targetsWeights)
	targets["rasters"] = [None] + [gdal.Open(g) for g in targets["weights"]["geotiffs"][1:]]

	print (targets["weights"])

	targets["grids"] = \
	    [ { "density" : targets["rasters"][e].GetRasterBand(1).ReadAsArray(),
	        "speed"   : targets["rasters"][e].GetRasterBand(2).ReadAsArray(),
	        "acc"     : targets["rasters"][e].GetRasterBand(3).ReadAsArray() } \
	    for e in range(1, len(targets["rasters"]))]

	targets["weights"]["grids"] = targets["grids"]

	return targets

def initializeByFile (configFile):
	with open(configFile) as f:
		config = yaml.safe_load(f)

	regionFile = config['config']['dataSources']['regionRaster']
	sourceURL = config['config']['dataSources']['ncSourceURL']
	currentsMagnitudeFile = config['config']['dataSources']['currentsMagnitudeFile']
	currentsDirectionFile = config['config']['dataSources']['currentsDirectionFile']


	startDate = config['config']['missionTime']['startDate']
	if (startDate == "None"):
		startDate = datetime.now ()
	numDays = config['config']['missionTime']['numDays']
	interval_min = config['config']['missionTime']['interval']

	params = initialize (regionFile, currentsMagnitudeFile,
		currentsDirectionFile, sourceURL,
		startDate, numDays, interval_min)

	# Add targets data, if available
	targetsFile    = config['config']['dataSources']['targetsImg']
        targetsTable   = config['config']['dataSources']['targetsTbl']
        targetsWeights = config['config']['dataSources']['targetsWeights']

        params['targets'] = initializeTargets (targetsFile, targetsTable, targetsWeights)

	return params

def main (configFile, haveCurrents = False):
	data = initializeByFile (configFile)

	if (haveCurrents == False):
		getCurrentsRasterSet (data, showTiming = True)
	else:

		data['currents']['magnitude']['raster'] = gdal.Open (data['currents']['magnitude']['file'])
		data['currents']['direction']['raster'] = gdal.Open (data['currents']['direction']['file'])

	return data


def getGrid (points, zName, YI, XI):
	ZI = interpolate (points['lat'].tolist (),
		points['lon'].tolist (), points[zName].tolist (), YI, XI)
	return ZI

def getCurrentsRasterSet (params, showTiming = False):


	# Get all the points, each list elem corresponds to a timestamp
	start_time = time.time ()

	pointsSeries = getNcByDuration (
		params['nc']['data'],
		params['region']['extent'],
		params['timespan']['start'],
		params['timespan']['stop'],
		params['timespan']['interval'])

	end_time = time.time ()
	if (showTiming == True):
		print ("lapsed time was %g seconds" % (end_time - start_time))

	# Init gdal rasters

        drv = gdal.GetDriverByName ('GTiff')
        geotransform = list(params['region']['raster'].GetGeoTransform ())
        geotransform = [params["region"]["extent"]["minx"], geotransform[1], 0, params["region"]["extent"]["maxy"], 0, -geotransform[5]]

        params['currents']['magnitude']['raster'] = \
             drv.Create(params['currents']['magnitude']['file'],
                        params['region']['extent']['cols'],
                        params['region']['extent']['rows'],
                        len (pointsSeries),
                        gdal.GDT_Float32)
        params['currents']['magnitude']['raster'].SetGeoTransform(geotransform)
        params['currents']['magnitude']['raster'].SetProjection(params['region']['raster'].GetProjection ())

        params['currents']['direction']['raster'] = \
             drv.Create(params['currents']['direction']['file'],
                        params['region']['extent']['cols'],
                        params['region']['extent']['rows'],
                        len (pointsSeries),
                        gdal.GDT_Float32)
        params['currents']['direction']['raster'].SetGeoTransform(geotransform)
        params['currents']['direction']['raster'].SetProjection(params['region']['raster'].GetProjection ())


	XI, YI = np.mgrid [params['region']['extent']['minx']:params['region']['extent']['maxx']:complex(0, params['region']['extent']['cols']),
		params['region']['extent']['miny']:params['region']['extent']['maxy']:complex(0, params['region']['extent']['rows'])]

	start_time = time.time ()

	i = 1 # Band index. Count starts at 1.
	for points in pointsSeries:
		print (i)

                cols = params['region']['extent']['cols']
                rows = params['region']['extent']['rows']

                # Interpolate the u and v components of current vector points
                zu = griddata((points["lonc"], points["latc"]), points["u"], (XI, YI), method='cubic').T
                zv = griddata((points["lonc"], points["latc"]), points["v"], (XI, YI), method='cubic').T

                # Init empty grids to store current magnitude and direction
                zm = np.zeros(shape=(rows, cols))
                zd = np.zeros(shape=(rows, cols))

                # Fill in magnitude and direction grids
                for row in range(rows):
                    for col in range(cols):
                        if np.isnan(zu[row][col]) or np.isnan(zv[row][col]):
                            m = 0
                            d = 0
                        else:
                            m, d = components2MagDir((zu[row][col], zv[row][col]), (1, 0))
                        zm[row][col] = m
                        zd[row][col] = d

                zm = np.flipud(zm)
                zd = np.flipud(zd)

                np.savetxt("beforeMag.txt", zm[40:60,40:60], fmt="%g.2")


                # Save to GDAL raster band
                band = params['currents']['magnitude']['raster'].GetRasterBand(i)
	        band.SetNoDataValue(0)
	        band.WriteArray(zm, 0, 0)

                band = params['currents']['direction']['raster'].GetRasterBand(i)
	        band.SetNoDataValue(0)
	        band.WriteArray(zd, 0, 0)

                minx = params['region']['extent']['minx']
                maxx = params['region']['extent']['maxx']
                miny = params['region']['extent']['miny']
                maxy = params['region']['extent']['maxy']

                m = Basemap (projection='cyl',
                    llcrnrlon=minx-.005, \
                    llcrnrlat=miny-.005, \
                    urcrnrlon=maxx+.005, \
                    urcrnrlat=maxy+.005, \
                    resolution='i')

                # Plot
                ###plt.close("all")
                ###image = georaster.SingleBandRaster( params["region"]["file"],
                ###    load_data=(minx, maxx, miny, maxy), latlon=True)
                ###fig1 = plt.figure(figsize=(8,8))
                ###ax1 = fig1.add_subplot(111)
                ###ax1.imshow(image.r, extent=(minx, maxx, miny, maxy), alpha=0.6)
                ###ax1.imshow(zm, extent=(minx, maxx, miny, maxy), alpha = 0.75)
                ###Q = ax1.quiver(points["lonc"],points["latc"],points["u"],points["v"],scale=20)
                ###qk = plt.quiverkey(Q,0.92,0.08,0.50,'0.5 m/s',labelpos='W')
                ###plt.savefig("currents_magnitude_" + str(i) + ".png")

                ###plt.close("all")
                ###image = georaster.SingleBandRaster( params["region"]["file"],
                ###    load_data=(minx, maxx, miny, maxy), latlon=True)
                ###fig1 = plt.figure(figsize=(8,8))
                ###ax1 = fig1.add_subplot(111)
                ###ax1.imshow(image.r, extent=(minx, maxx, miny, maxy), alpha=0.6)
                ###ax1.imshow(zd, extent=(minx, maxx, miny, maxy), alpha = 0.75)
                ###Q = ax1.quiver(points["lonc"],points["latc"],points["u"],points["v"],scale=20)
                ###qk = plt.quiverkey(Q,0.92,0.08,0.50,'0.5 m/s',labelpos='W')
                ###plt.savefig("currents_direction_" + str(i) + ".png")

		# Increment band
		i = i + 1

        params['currents']['magnitude']['raster'].FlushCache()  # Write to disk.
        params['currents']['direction']['raster'].FlushCache()  # Write to disk.

	end_time = time.time ()
	if (showTiming == True):
		print ("lapsed time was %g seconds" % (end_time - start_time))
