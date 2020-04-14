#!/usr/bin/python3

import pyvisgraph as vg
import rasterio
from rasterio.features import shapes
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gp
from optparse import OptionParser

###########
# Options #
###########

parser = OptionParser()
parser.add_option("-r", "--region",
        help = "Path to raster containing binary occupancy region (1 -> obstacle, 0 -> free)",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.tif")
parser.add_option("-g", "--graph",
        help = "Path to save graph",
        default = "/home/ekrell/Downloads/ADVGEO_DL/sample_region_mini.graph")
parser.add_option("-m", "--map",
        help = "Path to save graph map",
parser.add_option("-b", "--build",
        help = "Build the graph",
        action = "store_true",
        default = False)

(options, args) = parser.parse_args()

regionRasterFile = options.region
graphOutFile = options.graph



