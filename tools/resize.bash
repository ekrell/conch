# Resize a raster
# If all rasters have same CRS, they can be clipped to the same size
# So, all rasters will have the same size in pixels, and these pixels will be the same spatial location

# Need to finish this script

gdalwarp -te xmin ymin xmax ymax input.tif output.tif


