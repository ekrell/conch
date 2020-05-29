#!/usr/bin/python
from osgeo import gdal

# Function for translating world and grid points and accessing grid elements

## !!!!!! ##
# ORIGIN is at Upper Left #
## !!!!!! ##


def world2grid (lat, lon, transform, nrow):
	row = int ( (lat - transform[3]) / transform[5] )
	col = int ( (lon - transform[0]) / transform[1] )
	return (row, col)

def getElemByGrid (row, col, grid):
	return grid[row][col]

def getElemByWorld (lat, lon, grid, transform):
	gridPoint = world2grid (lat, lon, transform, len(grid))
	return getElemByGrid (gridPoint[0], gridPoint[1], grid)

def grid2world (row, col, transform, nrow):
	lon = transform[1] * col + transform[2] * row + transform[0]
	lat = transform[4] * col + transform[5] * row + transform[3]
	return (lat, lon)

def getArchiveByGrid (row, col, grid, transform):
	archivePoint = { "row" : row, "col" : col, "lat" : None, "lon" : None, "elem" : None }
	worldPoint = grid2world (archivePoint["row"], archivePoint["col"], transform, len(grid))
	archivePoint["lat"] = worldPoint[0]
	archivePoint["lon"] = worldPoint[1]
	elem = getElemByGrid (archivePoint["row"], archivePoint["col"], grid)
	archivePoint["elem"] = elem

        # A flipped row for when you have the origin at lower left.
        archivePoint["rowFromBottom"] = (-1) * archivePoint["row"] + len(grid)

	return archivePoint

def getArchiveByWorld (lat, lon, grid, transform):
	archivePoint = { "row" : None, "col" : None, "lat" : lat, "lon" : lon, "elem" : None }
	gridPoint = world2grid (archivePoint["lat"], archivePoint["lon"], transform, len(grid))
	archivePoint["row"] = gridPoint[0]
	archivePoint["col"] = gridPoint[1]
	elem = getElemByGrid (archivePoint["row"], archivePoint["col"], grid)
	archivePoint["elem"] = elem

        # A flipped row for when you have the origin at lower left.
        archivePoint["rowFromBottom"] = (-1) * archivePoint["row"] + len(grid)

	return archivePoint

# Functions for loading grids

def loadLayerByFile (filename):
	data = gdal.Open (filename)
	band = data.GetRasterBand (1)
	cols = data.RasterXSize
	rows = data.RasterYSize
	transform = data.GetGeoTransform ()
	grid = band.ReadAsArray (0, 0, cols, rows).T
	layer = { "grid" : grid, "transform" : transform, "rows" : rows, "cols" : cols }
	return layer

def testLayer (layer, points_list):

	for point in points_list:
		gridPoint = world2grid (point[0], point[1], layer["transform"])
		elem = getElemByWorld (point[0], point[1], layer["grid"], layer["transform"])
		pointCopy = grid2world (gridPoint[0], gridPoint[1], layer["transform"])
		print (point)
		print (gridPoint)
		print (pointCopy)
		print (elem)
		aPoint = getArchiveByGrid(gridPoint[0], gridPoint[1], layer["grid"], layer["transform"])
		print (aPoint)
		aPoint2 = getArchiveByWorld(point[0], point[1], layer["grid"], layer["transform"])
		print (aPoint2)
		print ("---")


def test (gridBrick):
	points_list = [ (-70.89, 42.54138), (-70.885, 42.54257) ] #list of X,Y coordinates

	testLayer (gridBrick["terrain"], points_list)
	testLayer (gridBrick["currentDirection"], points_list)
	testLayer (gridBrick["currentMagnitude"], points_list)


def initBrick ():
	driver = gdal.GetDriverByName('GTiff')

	terrain_filename = "../DataLayers/scenario1/Map/regionLand.tif"
	currentDirection_filename = "../DataLayers/scenario1/Current/currents_direction.tif"
	currentMagnitude_filename = "../DataLayers/scenario1/Current/currents_magnitude.tif"

	gridBrick = { "terrain" : None, "currentDirection" : None, "currentMagnitude" : None }
	gridBrick["terrain"] = loadLayerByFile (terrain_filename)
	gridBrick["currentDirection"] = loadLayerByFile (currentDirection_filename)
	gridBrick["currentMagnitude"] = loadLayerByFile (currentMagnitude_filename)

	return gridBrick
