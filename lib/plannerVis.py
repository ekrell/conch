#!/usr/bin/python
'''
PlannerViz.py
Author: Evan Krell

Plotting and other visualization tools for the path planner.
'''

import sys

from osgeo                  import gdal
from mpl_toolkits.basemap   import Basemap
from matplotlib.collections import LineCollection
from matplotlib.colors      import ListedColormap, BoundaryNorm

import georaster
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import numpy              as np

import gridUtils         as GridUtil
import plannerTools      as PlannerTools

def initMap(raster, width, rasterFile):

    cols = raster.RasterXSize
    rows = raster.RasterYSize
    height = ( float(rows) / float(cols) ) * float(width)

    f = plt.figure(figsize = (width, height))
    ax = f.add_subplot(111)

    region = georaster.SingleBandRaster(rasterFile, load_data=False)

    # Get region extent
    minx, maxx, miny, maxy = region.extent

    # set Basemap with slightly larger extents
    m = Basemap (projection='cyl',
        llcrnrlon=minx-.005, \
        llcrnrlat=miny-.005, \
        urcrnrlon=maxx+.005, \
        urcrnrlat=maxy+.005, \
        resolution='i')        # 'h' : high res

    # load the geotiff image, assign it a variable
    image = georaster.SingleBandRaster( rasterFile,
                load_data=(minx, maxx, miny, maxy),
                latlon=True)

    # plot the image on matplotlib active axes
    # set zorder to put the image on top of coastlines and continent areas
    # set alpha to let the hidden graphics show through
    #ax.imshow(image.r, extent=(minx, maxx, miny, maxy), zorder=10, alpha=0.6)
    ax.imshow(image.r, extent=(0, cols, 0, rows), zorder=10, alpha=0.6)

    return ax, m


def addImage(ax, imgFile, raster):


    # Get raster extent (in cols, rows)
    extent = 0, raster.RasterXSize, 0, raster.RasterYSize

    # Open image
    img = plt.imread(imgFile)

    # Plot image
    imgplot = plt.imshow(img, interpolation = 'nearest', extent = extent)

    return ax



def addVectorField(regionGrid, magnitudeGrid, directionGrid, sampleInterval):
    from math import cos, sin

    rows = len(regionGrid)
    cols = len(regionGrid[0])
    mrows = len(magnitudeGrid)
    mcols = len(magnitudeGrid[0])
    drows = len(directionGrid)
    dcols = len(directionGrid[0])

    xSamples = [] # X coords of sample points
    ySamples = [] # Y coords of sample points
    uSamples = [] # u component of sample points
    vSamples = [] # v component of sample points

    for y in range(0, rows - 1, sampleInterval):
        for x in range(0, cols - 1, sampleInterval):
            xSamples.append(x)
            ySamples.append(y)

            #np.savetxt("yes", magnitudeGrid * 100, "%01d")
            #exit()

            if (regionGrid[y][x] == 0):
                uSamples.append(magnitudeGrid[y][x] * cos (directionGrid[y][x]))
                vSamples.append(magnitudeGrid[y][x] * sin (directionGrid[y][x]))
            else:
                uSamples.append(0)
                vSamples.append(0)

    # Overlay the vector field
    ySamples = [(-1) * y + rows for y in ySamples]
    plt.quiver(xSamples[::], ySamples[::], uSamples[::], vSamples[::])

    return None



def makeBlankMap(raster, width, rasterFile, outFile):

    # Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    # Save map
    if outFile is not None:
        plt.savefig(outFile)

    return ax

def makeSafezonesMap(raster, width, rasterFile, safezones, outFile):

    # Clear figures
    plt.close("all")

    grid = raster.GetRasterBand(1).ReadAsArray()
    rows = grid.shape[0]

    # Legend
    safezoneMarker = "s"
    safezoneColor  = "#73ec90"
    safezoneSize   = 100

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    x = []
    y = []
    for idx, row in safezones.iterrows():
        pointArchive = GridUtil.getArchiveByWorld(row["Lat"], row["Lon"],
                           grid, raster.GetGeoTransform())
        y.append(pointArchive["rowFromBottom"])
        x.append(pointArchive["col"])

    plt.scatter(x, y, s = safezoneSize, marker = safezoneMarker, color = safezoneColor)

    # Save map
    if outFile is not None:
        plt.savefig(outFile)

    return ax

def makeTargetsMap(raster, width, rasterFile, targets, outFile):

    # Clear figures
    plt.close("all")

    grid = raster.GetRasterBand(1).ReadAsArray()
    rows = grid.shape[0]

    # Legend
    targetMarker = "8"
    targetColor  = "#ec73c9"
    targetSize   = 10

    tareaColor   = "#6ae7ce"
    tareaLColor  = "#39776f"

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    x = []
    y = []
    w = []
    l = []

    for idx, row in targets.iterrows():
        pointArchive = GridUtil.getArchiveByWorld(row["Lat"], row["Lon"],
                           grid, raster.GetGeoTransform())

        x.append(pointArchive["col"])
        y.append(pointArchive["rowFromBottom"])


        ul = (pointArchive["rowFromBottom"] + 0.5 * row["Length"],
              pointArchive["col"] - 0.5 * row["Width"])
        ll = (pointArchive["rowFromBottom"] - 0.5 * row["Length"],
              pointArchive["col"] - 0.5 * row["Width"])
        ur = (pointArchive["rowFromBottom"] + 0.5 * row["Length"],
              pointArchive["col"] + 0.5 * row["Width"])
        lr = (pointArchive["rowFromBottom"] - 0.5 * row["Length"],
              pointArchive["col"] + 0.5 * row["Width"])

        xbox = [ul[1], ur[1], lr[1], ll[1], ul[1]]
        ybox = [ul[0], ur[0], lr[0], ll[0], ul[0]]

        plt.plot(xbox, ybox)

        font = { 'family' : 'serif',
                 'color'  : 'mediumseagreen',
                 'weight' : 'normal',
                 'size'   : 9,
        }

        # Text
        plt.text(pointArchive["col"] + 5,
                 pointArchive["rowFromBottom"] - 5,
                 "(" + str(pointArchive["lat"]) + ",\n" +
                       str(pointArchive["lon"]) + ")",
                 fontdict = font)

        w.append(row["Width"])
        l.append(row["Length"])

    plt.scatter(x, y, s = targetSize, marker = targetMarker, color = targetColor)

    #for idx in range(len(w)):
    #    llx = x[idx] - (0.5) * w[idx]
    #    lly = y[idx] - (0.5) * l[idx]
    #    tarea = patches.Rectangle((llx, lly), w[idx], l[idx],
    #                linewidth = 1, edgecolor = tareaLColor, facecolor = tareaColor)
    #    ax.add_patch(tarea)

    # Save map
    if outFile is not None:
        plt.savefig(outFile)

    return ax

def makeEntitiesMap(raster, width, rasterFile, entities, outFile):

    # Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    # Add entities image
    ax = addImage(ax, entities, raster)

    # Save map
    if outFile is not None:
        plt.savefig(outFile)

    return ax


def makeWaterCurrentMap(regionRaster, magnitudeRaster, directionRaster,
                        width, regionRasterFile, outFile, band = 1, sampleInterval = 10):

    # Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(regionRaster, width, regionRasterFile)

    # Prepare data for vector field
    regionGrid    = regionRaster.GetRasterBand(1).ReadAsArray()
    magnitudeGrid = np.flipud(magnitudeRaster.GetRasterBand(band).ReadAsArray())
    directionGrid = np.flipud(directionRaster.GetRasterBand(band).ReadAsArray())


    # Add water currents as vector field
    addVectorField(regionGrid, magnitudeGrid, directionGrid, sampleInterval)

    # Save map
    if outFile is not None:
        plt.savefig(outFile)

    return ax



def makeTargetSelectMap(raster, rasterFile, targets, targetSequence, width, outfile):

    #Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    grid = raster.GetRasterBand(1).ReadAsArray()
    rows = grid.shape[0]


    # Legend
    targetMarker = "8"
    targetColor  = "#ec73c9"
    targetSize   = 100

    # Add targets
    x = []
    y = []
    w = []
    l = []

    for TID in targetSequence:
        lat    = targets.loc[[TID]]["Lat"]
        lon    = targets.loc[[TID]]["Lon"]
	width  = targets.loc[[TID]]["Width"]
        length = targets.loc[[TID]]["Length"]
        pointArchive = GridUtil.getArchiveByWorld(lat, lon,
                           grid, raster.GetGeoTransform())
        x.append(pointArchive["col"])
        y.append(pointArchive["rowFromBottom"])
        w.append(width)
        l.append(length)

    plt.scatter(x, y, s = targetSize, marker = targetMarker, color = targetColor)

    n = range(1, len(y) + 1)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    # Save map
    if outfile is not None:
        plt.savefig(outfile)

    return ax


def makeCoverageZoomMaps(raster, rasterFile, coverage, width, outfiles):

    coverageZoomMaps = []

    #Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    grid = raster.GetRasterBand(1).ReadAsArray()

    region = georaster.SingleBandRaster(rasterFile, load_data=False)
    nrows = len(grid)


    for c in coverage:
        coveragePoints = c["points"]
        path           = c["path"]["path"]

        # Extract latitude and longitude from the path coordinates
        lat = []
        lon = []
        cols = []
        rows = []

        for idx in path:
            if idx < 0:
                continue   # Skip nulls

            ncols = len(coveragePoints[0])
            row, col = PlannerTools.idx2rowcol(idx, ncols)

            worldrow, worldcol = GridUtil.grid2world(
                coveragePoints[row][col]["rowFromBottom"],
                coveragePoints[row][col]["col"],
                raster.GetGeoTransform(),
		nrows)

            lat.append(worldrow)
            lon.append(worldcol)

            cols.append(coveragePoints[row][col]["col"])
            rows.append(coveragePoints[row][col]["rowFromBottom"])

        plt.plot (cols, rows, 'rx')
        #plt.savefig (outfile)

        npoints = len(rows)
        for i in range(npoints - 1):
            a = (float(i) / float(npoints - 1))
            plt.plot(cols[i:i+2], rows[i:i+2],
                     alpha = a, color = "green")


    count = 0
    for c in coverage:
        coveragePoints = c["points"]

        outfile = outfiles[count]

        lowerleft  = (coveragePoints[0][0]["rowFromBottom"],
                      coveragePoints[0][0]["col"])
        upperleft  = (coveragePoints[len(coveragePoints) - 1][0]["rowFromBottom"],
                      coveragePoints[len(coveragePoints) - 1][0]["col"])
        upperright = (coveragePoints[len(coveragePoints) - 1] \
                              [len(coveragePoints[0]) - 1]["rowFromBottom"],
          coveragePoints[len(coveragePoints) - 1] \
                                    [len(coveragePoints[0]) - 1]["col"])
        lowerright = (coveragePoints[0][len(coveragePoints[0]) - 1]["rowFromBottom"],
              coveragePoints[0][len(coveragePoints[0]) - 1]["col"])

        xmin = lowerleft[1]
        xmax = lowerright[1]
        ymin = upperleft[0]
        ymax = lowerleft[0]

        plt.axis([xmin,xmax,ymin,ymax])
        plt.savefig(outfile)
        count = count + 1

    return coverageZoomMaps


def makeCoverageMap(raster, rasterFile, coverage, width, outfile):

    #Clear figures
    plt.close("all")

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    grid = raster.GetRasterBand(1).ReadAsArray()

    region = georaster.SingleBandRaster(rasterFile, load_data=False)
    nrows = len(grid)
    for c in coverage:
        coveragePoints = c["points"]
        path           = c["path"]["path"]

        # Extract latitude and longitude from the path coordinates
        lat = []
        lon = []
        cols = []
        rows = []

        for idx in path:
            if idx < 0:
                continue   # Skip nulls

            ncols = len(coveragePoints[0])
            row, col = PlannerTools.idx2rowcol(idx, ncols)

            worldrow, worldcol = GridUtil.grid2world(
                coveragePoints[row][col]["rowFromBottom"],
                coveragePoints[row][col]["col"],
                raster.GetGeoTransform(),
		nrows)

            lat.append(worldrow)
            lon.append(worldcol)

            cols.append(coveragePoints[row][col]["col"])
            rows.append(coveragePoints[row][col]["rowFromBottom"])

        plt.plot (cols, rows, 'rx--')
        plt.savefig (outfile)

    return ax

def makeGotoMap(raster, rasterFile, gotoResults, width, outfile):

    #Clear figures
    plt.close("all")

    # Legend
    marker = "8"
    size   = 100
    startColor   = "#ec73c9"
    targetColor  = "orange"
    endColor     = "seagreen"

    # Build terrain map
    ax, m = initMap(raster, width, rasterFile)

    # Init region grid
    grid = raster.GetRasterBand(1).ReadAsArray()

    # Number of path segments
    numSegments = len(gotoResults)

    # USV start location
    print (gotoResults)
    startCoord = gotoResults[0]["startPoint"]
    startArchive = GridUtil.getArchiveByWorld(startCoord["Lat"],
                       startCoord["Lon"], grid, raster.GetGeoTransform())
    ystart = [startArchive["rowFromBottom"]]
    xstart = [startArchive["col"]]

    # Target locations
    xTargets = []
    yTargets = []


    # Plot paths
    for gotoResult in gotoResults:
        y = []
        x = []
        # Waypoints
        y = [c["rowFromBottom"] for c in gotoResult["solutionPath"]["coords"]]
        x = [c["col"]           for c in gotoResult["solutionPath"]["coords"]]
        # Endpoint
        endArchive = GridUtil.getArchiveByWorld(gotoResult["endPoint"]["Lat"],
                           gotoResult["endPoint"]["Lon"], grid, raster.GetGeoTransform())
        yTargets.append(endArchive["rowFromBottom"])
        xTargets.append(endArchive["col"])


        plt.plot(x, y, 'x--')

    plt.scatter(xstart, ystart, s = size,
        marker = marker, color = startColor)
    plt.scatter(xTargets, yTargets, s = size,
        marker = marker, color = targetColor)
    plt.scatter(xTargets[len(xTargets) -1], yTargets[len(yTargets) -1],
        s = size, marker = marker, color = endColor)

    plt.savefig(outfile)

    return ax
