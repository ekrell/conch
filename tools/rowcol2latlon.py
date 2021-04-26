#!/usr/bin/python
from optparse import OptionParser
from osgeo import gdal

def world2grid (lat, lon, transform, nrow):
    row = int ((lat - transform[3]) / transform[5])
    col = int ((lon - transform[0]) / transform[1])
    return (row, col)

def grid2world (row, col, transform, nrow):
    lon = transform[1] * col + transform[2] * row + transform[0]
    lat = transform[4] * col + transform[5] * row + transform[3]
    return (lat, lon)

def main():
    # Options
    parser = OptionParser()
    parser.add_option("-g", "--geotiff", default = None,
        help = "Raster (geotiff) of region.")
    parser.add_option("-r", "--row", type = "float",
        help = "Position row.")
    parser.add_option("-c", "--col", type = "float",
        help = "Position column.")
    parser.add_option("-a", "--lat", type = "float",
        help = "Position latitude.")
    parser.add_option("-o", "--lon", type = "float",
        help = "Position longitude.")
    (options, args) = parser.parse_args()

    if options.geotiff is None:
        print("Invalid arguments: Geotiff raster region required (-g)")

    rowcol2latlon = True
    # Check if converting (row, col) to (lat, lon)
    if options.row is not None and \
       options.col is not None and \
       options.lat is     None and \
       options.lon is     None:
        rowcol2latlon = True
    elif options.row is   None and \
       options.col is     None and \
       options.lat is not None and \
       options.lon is not None:
        rowcol2latlon = False
    else:
        print("Invalid arguments: specify both row (-r) and col (-c)")
        print("                OR specify both lat (-a) and lon (-o)")
        exit(1)

    # Process geotiff
    raster = gdal.Open(options.geotiff)
    grid = raster.GetRasterBand(1).ReadAsArray()
    transform = raster.GetGeoTransform()

    # Perform position conversion
    if rowcol2latlon:
        (row, col) = (options.row, options.col)
        (lat, lon) = grid2world(options.row, options.col,
                                transform, grid.shape[0])
    else:
        (lat, lon) = (options.lat, options.lon)
        (row, col) = world2grid(options.lat, options.lon,
                                transform, grid.shape[0])

    # Print coordinates
    print("(row, col) : ({}, {}) <---> (lat, lon) : ({}, {})".format(row, col, lat, lon))

if __name__ == '__main__':
    main()
