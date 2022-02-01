#/usr/bin/python3

"""Generate a plot of the data source & path.

Plot as many elements are are provided by user.
* Region raster  (required)
* Water currents
* Reward
* Start, goal
* Waypoints
"""

from optparse import OptionParser
from osgeo import gdal, ogr, osr, gdalconst
from gdalconst import GA_ReadOnly
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from inspect import getmembers, isclass

# Color maps
sea = (153.0/255.0, 188.0/255.0, 182.0/255.0)
land = (239.0/255.0, 209.0/255.0, 171.0/255.0)
cm_landsea = LinearSegmentedColormap.from_list(
    "landsea", [sea, land], N = 2
)
cm_landonly = LinearSegmentedColormap.from_list(
    "landsea", [(0, 0, 0, 0), land], N = 2
)


def grid2world(row, col, transform, nrow):
    lat = transform[4] * col + transform[5] * row + transform[3]
    lon = transform[1] * col + transform[2] * row + transform[0]
    return (lat, lon)


def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None, savefig_kw={}):
     # Behave like pyplot and act on current figure if no figure is specified
     fig = plt.gcf() if fig is None else fig

     # Need to set_rasterization_zorder in order for rasterizing to work
     zorder = -5  # Somewhat arbitrary, just ensuring less than 0

     if rasterize_list is None:
         # Have a guess at stuff that should be rasterised
         types_to_raster = ['QuadMesh',]  # 'Contour', 'collections']
         rasterize_list = []

         print("""
         No rasterize_list specified, so the following objects will
         be rasterized: """)
         # Get all axes, and then get objects within axes
         for ax in fig.get_axes():
             for item in ax.get_children():
                 if any(x in str(item) for x in types_to_raster):
                     rasterize_list.append(item)
         print('\n'.join([str(x) for x in rasterize_list]))
     else:
         # Allow rasterize_list to be input as an object to rasterize
         if type(rasterize_list) != list:
             rasterize_list = [rasterize_list]

     for item in rasterize_list:

         # Whether or not plot is a contour plot is important
         is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                       isinstance(item, matplotlib.tri.TriContourSet))

         # Whether or not collection of lines
         # This is commented as we seldom want to rasterize lines
         # is_lines = isinstance(item, matplotlib.collections.LineCollection)

         # Whether or not current item is list of patches
         all_patch_types = tuple(
             x[1] for x in getmembers(matplotlib.patches, isclass))
         try:
             is_patch_list = isinstance(item[0], all_patch_types)
         except TypeError:
             is_patch_list = False

         # Convert to rasterized mode and then change zorder properties
         if is_contour:
             curr_ax = item.ax.axes
             curr_ax.set_rasterization_zorder(zorder)
             # For contour plots, need to set each part of the contour
             # collection individually
             for contour_level in item.collections:
                 contour_level.set_zorder(zorder - 1)
                 contour_level.set_rasterized(True)
         elif is_patch_list:
             # For list of patches, need to set zorder for each patch
             for patch in item:
                 curr_ax = patch.axes
                 curr_ax.set_rasterization_zorder(zorder)
                 patch.set_zorder(zorder - 1)
                 patch.set_rasterized(True)
         else:
             # For all other objects, we can just do it all at once
             #curr_ax = item.axes
             #curr_ax.set_rasterization_zorder(zorder)
             item.set_rasterized(True)
             #item.set_zorder(zorder - 1)

     # dpi is a savefig keyword argument, but treat it as special since it is
     # important to this function
     if dpi is not None:
         savefig_kw['dpi'] = dpi

     # Save resulting figure
     fig.savefig(fname, **savefig_kw)


def main():

    # Options
    parser = OptionParser()
    parser.add_option("-r", "--region",
                      help = "Path to raster occupancy grid",
                      default = None)
    parser.add_option("-u", "--currents_mag",
                      help = "Path to raster with magnitude of water velocity.",
                      default = None)
    parser.add_option("-v", "--currents_dir",
                      help = "Path to raster with direction of water velocity.",
                      default = None)
    parser.add_option("-b", "--currents_band",
                      help = "Index of water currents raster band to plot.",
                      default = 1)
    parser.add_option("-w", "--reward",
                      help = "Path to numpy array (txt) with reward at each cell",
                      default = None)
    parser.add_option("-p", "--path",
                      help = "Path to waypoints file (txt).",
                      default = None)
    parser.add_option(      "--sx",
                      help = "Start longitude.",
                      default = None)
    parser.add_option(      "--sy",
                      help = "Start latitude.",
                      default = None)
    parser.add_option(      "--dx",
                      help = "Destination longitude.",
                      default = None)
    parser.add_option(      "--dy",
                      help = "Destination longitude.",
                      default = None)
    parser.add_option("-m", "--map",
                      help = "Path to save output plot",
                      default = None)
    (options, args) = parser.parse_args()

    region = options.region
    currents_mag = options.currents_mag
    currents_dir = options.currents_dir
    currents_band = options.currents_band
    reward = options.reward
    path = options.path
    sy = options.sy
    sx = options.sx
    dy = options.dy
    dx = options.dx

    outmap = options.map

    #startPoint = (float(options.sy), float(options.sx))
    #endPoint = (float(options.dy), float(options.dx))

    # Initalize plot
    fig, ax = plt.subplots()

    # Plot region map
    if region is None:
        print("Must provide a region raster ('-r')\nExiting...")
        exit(1)
    # Open raster
    ds = gdal.Open(region, GA_ReadOnly)
    # Determine bounding coordinates
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    lrx = ulx + (ds.RasterXSize * xres)
    lry = uly + (ds.RasterYSize * yres)
    llx = lrx - (ds.RasterXSize * xres)
    lly = lry
    urx = ulx + (ds.RasterXSize * xres)
    ury = uly
    # Initialize basemap
    m = Basemap( \
                llcrnrlon = llx,
                llcrnrlat = lly,
                urcrnrlon = urx,
                urcrnrlat = ury,
                resolution = "i",
                epsg = "4269")
    m.drawmapboundary(fill_color = sea)
    # Draw axes
    parallels = m.drawparallels(np.arange(-90.,120.,.05),
                                labels = [1, 0, 0, 0],
                                color = "gray")
    m.drawmeridians(np.arange(-180.,180.,.05),
                    labels = [0, 0, 0, 1],
                    color = "gray")
    for p in parallels:
        try:
            parallels[p][1][0].set_rotation(90)
        except:
            pass
    # Plot region raster
    data = ds.GetRasterBand(1).ReadAsArray()
    lon = np.linspace(llx, urx, data.shape[1])
    lat = np.linspace(ury, lly, data.shape[0])
    xx, yy = m(*np.meshgrid(lon,lat))
    m.pcolormesh(xx, yy, data, cmap = cm_landsea)
    nrows, ncols = data.shape

    # Plot reward
    if reward is not None:
        # Colormap
        ncolors = 256
        color_array = plt.get_cmap('Purples')(range(ncolors))
        # change alpha values
        color_array[0:1, -1] = 0.0
        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='reward',colors=color_array)
        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)
        # Plot reward values
        rewardGrid = np.flipud(np.loadtxt(reward))
        m.imshow(rewardGrid * rewardGrid, interpolation='nearest',
                 cmap = "reward", zorder=10)
        # Plot regions (again, so that reward is below)
        data = ds.GetRasterBand(1).ReadAsArray()
        lon = np.linspace(llx, urx, data.shape[1])
        lat = np.linspace(ury, lly, data.shape[0])
        xx, yy = m(*np.meshgrid(lon,lat))
        m.pcolormesh(xx, yy, data, cmap = cm_landonly, zorder=20)
        nrows, ncols = data.shape

    # Plot water currents
    if currents_mag is None and currents_dir is not None:
        print("Provided water currents direction ('-v'), but no magnitude ('-u').\nExiting...")
        exit(1)
    if currents_mag is not None and currents_dir is None:
        print("Provided water currents magnitude ('-u'), but no direction ('-v').\nExiting...")
        exit(1)
    if currents_mag is not None and currents_dir is not None:
        ds_mag = gdal.Open(currents_mag, GA_ReadOnly)
        ds_dir = gdal.Open(currents_dir, GA_ReadOnly)
        data_mag = ds_mag.GetRasterBand(currents_band).ReadAsArray()
        data_dir = ds_dir.GetRasterBand(currents_band).ReadAsArray()
        lon_water = np.linspace(llx, urx, data_mag.shape[1])
        lat_water = np.linspace(ury, lly, data_mag.shape[0])
        xx_water, yy_water = m(*np.meshgrid(lon_water, lat_water))
        data_u = data_mag * np.cos(data_dir)
        data_v = data_mag * np.sin(data_dir)
        xgrid = np.arange(0, data_u.shape[1], 30)
        ygrid = np.arange(0, data_u.shape[0], 30)
        points = np.meshgrid(ygrid, xgrid)
        Q = m.quiver(xx_water[tuple(points)], yy_water[tuple(points)],
                     data_u[tuple(points)], data_v[tuple(points)],
                     scale = 9, alpha = 1, latlon = True, color = "black", headwidth=4)
        keyVelocity = 0.5
        keyStr = 'Water: %3.1f m/s' % keyVelocity
        qk = plt.quiverkey(Q, 1.15, 1.06, keyVelocity, keyStr, labelpos = 'W')

    # Plot path
    if path is not None:
        points = [grid2world(p[0], p[1], ds.GetGeoTransform(), nrows) for p in \
                  np.loadtxt(path, delimiter = ",").astype("int")]
        m.plot(list(zip(*points))[1], list(zip(*points))[0],
               color = "black", linestyle = "--", linewidth = 3, zorder=30)

    # Plot start coordinates
    if sx is not None and sy is not None:
        sx = float(sx)
        sy = float(sy)
        m.scatter([sx], [sy],  marker = 'D', color = "red", s = 30,
                  edgecolors="black", zorder=10000)

    # Plot goal coordinates
    if dx is not None and dy is not None:
        dx = float(dx)
        dy = float(dy)
        m.scatter([dx], [dy],  marker = 'X', color = "red", s = 45,
                  edgecolors="black", zorder=10000)

    if options.map is None:
        print("Save this plot with the -m <filename> option")
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(outmap)


if __name__ == '__main__':
    main()
