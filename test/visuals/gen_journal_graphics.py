#!/usr/bin/python3

# The purpose is to generate figures for paper:
#   Autonomous Surface Vehicle Energy-Efficient and Reward-Based
#   Path Plannning using Particle Swarm Optimization and Visibility Graphs

from osgeo import gdal, ogr, osr, gdalconst
from gdalconst import GA_ReadOnly
from shapely.geometry import shape
import shapefile
import numpy as np
import pandas as pd
from optparse import OptionParser
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import dill as pickle
import matplotlib
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from itertools import repeat
import seaborn as sns
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


from inspect import getmembers, isclass
def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.
    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

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



def grid2world(row, col, transform, nrow):
    lat = transform[4] * col + transform[5] * row + transform[3]
    lon = transform[1] * col + transform[2] * row + transform[0]
    return (lat, lon)

#############
# Constants #
#############

bestPSO = [
    [4, 3, 1, 7, 9,],
    [3, 2, 6, 1, 5,],
    [1, 10, 7, 5, 5,],
]

bestPSO_VG = [
    [3, 8, 9, 2, 2,],
    [8, 4, 3, 5, 5,],
    [9, 4, 7, 10, 1,],
]

bestPSO_r500 = [
    [2, 5, 2, 9, 2,],
    [3, 3, 3, 7, 8,],
    [4, 10, 7, 10, 2,],
]

bestPSO_r1000 = [
    [10, 2, 9, 2, 3,],
    [7, 8, 1, 10, 2,],
    [3, 2, 9, 3, 6,],
]

bestPSO_VG_r500 = [
    [4, 6, 8, 3, 7,],
    [3, 3, 6, 7, 1,],
    [10, 5, 10, 10, 1,],
]

bestPSO_VG_r1000 = [
    [5, 6, 8, 3, 7,],
    [3, 3, 10, 10, 4,],
    [6, 4, 4, 7, 10,],
]

# Color maps
sea = (153.0/255.0, 188.0/255.0, 182.0/255.0)
land = (239.0/255.0, 209.0/255.0, 171.0/255.0)
c_vispnt = (56.0/255.0, 56.0/255.0, 56.0/255.0)
cm_landsea = LinearSegmentedColormap.from_list(
    "landsea", [sea, land], N = 2
)
cm_landonly = LinearSegmentedColormap.from_list(
    "landsea", [(0, 0, 0, 0), land], N = 2
)


# Global font size
plt.rcParams.update({'font.size': 7})

###########
# Options #
###########

TRIALS = 10

# Path tasks
P1 = [-70.99428, 42.32343, -70.88737, 42.33600]
P2 = [-70.97322, 42.33283, -70.903406, 42.27183]
P3 = [-70.95617, 42.36221, -70.97952, 42.35282]

C1 = (231.0/255.0, 80.0/255.0, 105.0/255.0)
C2 = (130.0/255.0, 80.0/255.0, 231.0/255.0)
C3 = (55.0/255.0, 137.0/255.0, 68.0/255.0)

# Water currents
currentsRasterFile_mag_W1 = "test/inputs/20170503_magwater.tiff"
currentsRasterFile_dir_W1 = "test/inputs/20170503_dirwater.tiff"
currentsRasterFile_mag_W2 = "test/inputs/20170801_magwater.tiff"
currentsRasterFile_dir_W2 = "test/inputs/20170801_dirwater.tiff"
currentsRasterFile_mag_W3 = "test/inputs/20191001_magwater.tiff"
currentsRasterFile_dir_W3 = "test/inputs/20191001_dirwater.tiff"
currentsRasterFile_mag_W4 = "test/inputs/20200831_magwater.tiff"
currentsRasterFile_dir_W4 = "test/inputs/20200831_dirwater.tiff"
# Output files
currentsRasterOut_W1_0 = "test/visuals/20170503_currents_0.pdf"
currentsRasterOut_W1_1 = "test/visuals/20170503_currents_1.pdf"
currentsRasterOut_W1_2 = "test/visuals/20170503_currents_2.pdf"
currentsRasterOut_W2_0 = "test/visuals/20170801_currents_0.pdf"
currentsRasterOut_W2_1 = "test/visuals/20170801_currents_1.pdf"
currentsRasterOut_W2_2 = "test/visuals/20170801_currents_2.pdf"
currentsRasterOut_W3_0 = "test/visuals/20191001_currents_0.pdf"
currentsRasterOut_W3_1 = "test/visuals/20191001_currents_1.pdf"
currentsRasterOut_W3_2 = "test/visuals/20191001_currents_2.pdf"
currentsRasterOut_W4_0 = "test/visuals/20200831_currents_0.pdf"
currentsRasterOut_W4_1 = "test/visuals/20200831_currents_1.pdf"
currentsRasterOut_W4_2 = "test/visuals/20200831_currents_2.pdf"

# Reward raster
rewardFile = "test/inputs/reward.txt"

# Full region
lower_left = {"lat" : 42.2647, "lon" : -70.9996}
upper_right = {"lat" : 42.3789, "lon" : -70.87151}
regionRasterFile = "test/inputs/full.tif"
regionShapeFile = "test/outputs/visgraph_build/visgraph.shp"
# Output files
regionOut = "test/visuals/region.pdf"
polyOut = "test/visuals/poly.pdf"

# Graphs
visgraphFile_P1 = "test/outputs/graphplanner_visgraph/visgraph_P1.pickle"
visgraphFile_P2 = "test/outputs/graphplanner_visgraph/visgraph_P2.pickle"
visgraphFile_P3 = "test/outputs/graphplanner_visgraph/visgraph_P3.pickle"
# Output files
visgraphOut_P1 = "test/visuals/visgraph_P1.png"
visgraphOut_P2 = "test/visuals/visgraph_P2.png"
visgraphOut_P3 = "test/visuals/visgraph_P3.png"

# Dijkstra results (graph)
gpathResDir = "test/outputs/graphplanner/haversine_s.5"
gpathResStr = "path-dijkstra-{}-P{}W{}.txt"
# Dijkstra results (VG)
vgpathResDir = "test/outputs/graphplanner_visgraph/haversine_s.5"
vgpathResStr = "path-dijkstra-{}-P{}W{}.txt"
# PSO results (random)
rPSOpathResDir = "test/outputs/metaplanner/"
rPSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.txt"
rPSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.out"
r_10kgens_PSOpathResDir = "test/outputs/metaplanner_10kgens/"
r_10kgens_PSOpathResStr = "PSO_P{}_W{}_S0.5_G2500_I100_N5__T{}.txt"
r_10kgens_PSOstatResStr = "PSO_P{}_W{}_S0.5_G2500_I100_N5__T{}.out"
# PSO results (VG init)
vgPSOpathResDir = "test/outputs/metaplanner_initPop/"
vgPSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.txt"
vgPSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.out"
vg_10kgens_PSOpathResDir = "test/outputs/metaplanner_10kgens_initPop/"
vg_10kgens_PSOpathResStr = "PSO_P{}_W{}_S0.5_G2500_I100_N5__T{}.txt"
vg_10kgens_PSOstatResStr = "PSO_P{}_W{}_S0.5_G2500_I100_N5__T{}.out"
# PSO results (random, reward)
r_r_PSOpathResDir = "test/outputs/metaplanner_reward/"
r_r500_PSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R1500__T{}.txt"
r_r500_PSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R1500__T{}.out"
r_r1000_PSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R2000__T{}.txt"
r_r1000_PSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R2000__T{}.out"
# PSO results (VG init, reward)
vg_r_PSOpathResDir = "test/outputs/metaplanner_initPop_reward/"
vg_r500_PSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R1500__T{}.txt"
vg_r500_PSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R1500__T{}.out"
vg_r1000_PSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R2000__T{}.txt"
vg_r1000_PSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5_R2000__T{}.out"

def init(width=1.9, height=1.9):
    # Init basemap
    fig = plt.figure(figsize=(width, height))
    m = Basemap( \
                llcrnrlon = lower_left["lon"],
                llcrnrlat = lower_left["lat"],
                urcrnrlon = upper_right["lon"],
                urcrnrlat = upper_right["lat"],
                resolution = "i",
                epsg = "4269")
    m.drawmapboundary(fill_color = sea)

    #zorder = -5
    #types_to_raster = ['Spine']
    #rasterize_list = []
    #for ax in fig.get_axes():
    #    for item in ax.get_children():
    #        print(str(item))
    #        #if any(x in str(item) for x in types_to_raster):
    #        rasterize_list.append(item)
    #print(rasterize_list)

    #for item in rasterize_list:
    #    curr_ax = item.axes
    #    if curr_ax is not None:
    #        curr_ax.set_rasterization_zorder(zorder)
    #        item.set_rasterized(True)
    #        item.set_zorder(zorder - 1)

    parallels = m.drawparallels(np.arange(-90.,120.,.05), labels = [1, 0, 0, 0], color = "gray")
    m.drawmeridians(np.arange(-180.,180.,.05), labels = [0, 0, 0, 1], color = "gray")
    for p in parallels:
        try:
            parallels[p][1][0].set_rotation(90)
        except:
            pass
    return m

def region(m, colors = cm_landsea, z = None):
    # Plot region raster
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    lrx = ulx + (ds.RasterXSize * xres)
    lry = uly + (ds.RasterYSize * yres)
    llx = lrx - (ds.RasterXSize * xres)
    lly = lry
    urx = ulx + (ds.RasterXSize * xres)
    ury = uly
    data = ds.GetRasterBand(1).ReadAsArray()
    lon = np.linspace(llx, urx, data.shape[1])
    lat = np.linspace(ury, lly, data.shape[0])
    xx, yy = m(*np.meshgrid(lon,lat))
    if z is not None:
        m.pcolormesh(xx, yy, data, cmap = colors, zorder = z)
    else:
        m.pcolormesh(xx, yy, data, cmap = colors)


def currents(magFile, dirFile, band = 1):
    # Plot water currents
    ds_mag = gdal.Open(magFile, GA_ReadOnly)
    ds_dir = gdal.Open(dirFile, GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres = ds_mag.GetGeoTransform()
    lrx = ulx + (ds_mag.RasterXSize * xres)
    lry = uly + (ds_mag.RasterYSize * yres)
    llx = lrx - (ds_mag.RasterXSize * xres)
    lly = lry
    urx = ulx + (ds_mag.RasterXSize * xres)
    ury = uly
    data_mag = ds_mag.GetRasterBand(band).ReadAsArray()
    data_dir = ds_dir.GetRasterBand(band).ReadAsArray()
    lon_water = np.linspace(llx, urx, data_mag.shape[1])
    lat_water = np.linspace(ury, lly, data_mag.shape[0])
    xx_water, yy_water = m(*np.meshgrid(lon_water, lat_water))
    data_u = data_mag * np.cos(data_dir)
    data_v = data_mag * np.sin(data_dir)
    xgrid = np.arange(0, data_u.shape[1], 30)
    ygrid = np.arange(0, data_u.shape[0], 30)
    points = np.meshgrid(ygrid, xgrid)
    Q = m.quiver(xx_water[tuple(points)], yy_water[tuple(points)], data_u[tuple(points)], data_v[tuple(points)],
             scale = 5,
             alpha = 1, latlon = True, color = "black", headwidth=7)
    keyVelocity = 0.5
    keyStr = 'Water: %3.1f m/s' % keyVelocity
    qk = plt.quiverkey(Q, 1.05, 1.06, keyVelocity, keyStr, labelpos = 'W')

def graphplt(graphFile):
    graph = pickle.load(open(graphFile, 'rb'))
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    for s in list(graph.keys())[1:]:
        v = grid2world(s[0], s[1], transform, nrows)
        for d in graph[s]:
            w = grid2world(d[0], d[1], transform, nrows)
        m.plot([v[1], w[1]], [v[0], w[0]], marker = '.', color = c_vispnt, linewidth = 0.15)
    worldPoints = [grid2world(p[0], p[1], transform, nrows) for p in graph]
    x, y = m(list(zip(*worldPoints))[1], list(zip(*worldPoints))[0])
    m.scatter(x, y, color = c_vispnt, s = 0.5)

def graphtask(P, color, z = 10000):
    # Plot start location
    m.scatter([P[0]], [P[1]], marker = 'D', color = color, s = 60, zorder = z, edgecolors= "black")
    # Plot goal location
    m.scatter([P[2]], [P[3]], marker = "X", color = color, s = 70, zorder = z, edgecolors= "black")


def poly():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = init()
    m.readshapefile(regionShapeFile.replace(".shp", ""), "BH")
    patches = []
    for info, shape in zip(m.BH_info, m.BH):
        patches.append(Polygon(np.array(shape), True))
    ax.add_collection(PatchCollection(patches, facecolor = land, edgecolor = 'k', linewidths=1, zorder = 2))

def statPSO(pathOutFile):
    with open(pathOutFile) as f:
        lines = [line.rstrip() for line in f]
    convlines = []
    lines.pop(0)
    lines.pop(0)
    for line in lines:
        words = line.split()
        try:
            if words[0] == "Gen:":
                continue
            if words[0] == "Exit":
                break
        except:
            continue
        convlines.append(','.join(words))
    convdata = [[int(cline.split(",")[0]),
                 int(cline.split(",")[1]),
                 float(cline.split(",")[2]),
                 float(cline.split(",")[3]),
                 float(cline.split(",")[4]),
                 float(cline.split(",")[5]),
                 ] for cline in convlines]
    df = pd.DataFrame(convdata, columns = ["gen", "fevals", "gbest", "mean_vel", "meanlbest", "avg_dist"])
    return df

def resPSO(pathOutFile):
    import re
    costStr = None
    rewardStr = None
    file = open(pathOutFile, "r")
    for line in file:
        if re.search("Cost: ", line):
            costStr = line
        if re.search("Reward: ", line):
            rewardStr = line
    cost = float(costStr.split()[-1])
    reward = float(rewardStr.split()[-1])
    return cost, reward

def histPSO(path, work, outFile, trials = 1):
    r500_costs = []
    r1000_costs = []
    vg500_costs = []
    vg1000_costs = []
    r500_rewards = []
    r1000_rewards = []
    vg500_rewards = []
    vg1000_rewards = []
    # Collect stats from each trial
    for t in range(1, trials + 1):
        c, r = resPSO(r_r_PSOpathResDir + "/" + r_r500_PSOstatResStr.format(path, work, t))
        r500_costs.append(c)
        r500_rewards.append(r)
        c, r = resPSO(r_r_PSOpathResDir + "/" + r_r1000_PSOstatResStr.format(path, work, t))
        r1000_costs.append(c)
        r1000_rewards.append(r)
        c, r = resPSO(vg_r_PSOpathResDir + "/" + vg_r500_PSOstatResStr.format(path, work, t))
        vg500_costs.append(c)
        vg500_rewards.append(r)
        c, r = resPSO(vg_r_PSOpathResDir + "/" + vg_r1000_PSOstatResStr.format(path, work, t))
        vg1000_costs.append(c)
        vg1000_rewards.append(r)
    # Plot
    data_costs = [r500_costs, vg500_costs, r1000_costs, vg1000_costs]
    data_rewards = [r500_rewards, vg500_rewards, r1000_rewards, vg1000_rewards]
    fig, axs = plt.subplots(2, figsize=(6, 3))
    bp = axs[0].boxplot(data_costs, patch_artist=True)
    rColor = "tab:blue"
    vgColor = "tab:green"
    # Work
    bp['boxes'][0].set(facecolor = rColor)
    bp['boxes'][1].set(facecolor = rColor)
    bp['boxes'][2].set(facecolor = vgColor)
    bp['boxes'][3].set(facecolor = vgColor)
    for median in bp['medians']:
        median.set(color = "black", linewidth = 2)
    axs[0].set_ylabel("Work")
    axs[0].set_xticklabels([])
    # Reward
    bp = axs[1].boxplot(data_rewards, patch_artist=True)
    axs[1].set_ylabel("Reward")
    axs[1].set_xticklabels([r'${PSO}_{R}$,1x', r'${PSO}_{VG}$,1x', '${PSO}_{R}$,2x', '${PSO}_{VG}$,2x'], fontsize = 15)
    bp['boxes'][0].set(facecolor = rColor)
    bp['boxes'][1].set(facecolor = rColor)
    bp['boxes'][2].set(facecolor = vgColor)
    bp['boxes'][3].set(facecolor = vgColor)
    for median in bp['medians']:
        median.set(color = "black", linewidth = 2)
    axs[0].set_title("$\mathcal{{T}}_{{{}}}, \mathcal{{W}}_{{{}}}$".format(path + 1, work))
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(outFile)

def pltPSOconvergence(path, work, pathResDir, pathResStr, ax, trials = 1):
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(path, work, trial)
        dfPSO = statPSO(pathFile)
        ax.plot(dfPSO["gen"], dfPSO["gbest"])

def pltPSOconvergence_2(path, work, pathResDir, pathResStr, timePerGen, ax, trials = 1, color = "green"):
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(path, work, trial)
        dfPSO = statPSO(pathFile)
        ax.plot(dfPSO["gen"] * timePerGen, dfPSO["gbest"], color = color)

def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid", linewidth = 1, alpha = 1.0, z = None):
    path = [grid2world(p[0], p[1], transform, nrows) for p in \
            np.loadtxt(pathfile, delimiter = ",").astype("int")]
    if z is None:
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle, linewidth = linewidth, alpha = alpha)
    else:
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle, linewidth = linewidth, alpha = alpha, zorder = z)

def pltPSO(work, pathResDir, pathResStr, trials = 1, z = None, best = None):
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    bigLinewidth = 3
    bigAlpha = 1

    # Path 1
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(0, work, trial)
        linewidth = 1
        alpha = 0.6
        if best is not None:
            if trial == best[0][work]:
                linewidth = bigLinewidth
                alpha = bigAlpha
        if z is None:
            pathplt(pathFile, transform, nrows, color = C1, linestyle = "solid", alpha = alpha, linewidth = linewidth)
        else:
            pathplt(pathFile, transform, nrows, color = C1, linestyle = "solid", alpha = alpha, linewidth = linewidth, z = z)
    graphtask(P1, C1)
    # Path 2
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(1, work, trial)
        linewidth = 1
        alpha = 0.6
        if best is not None:
            if trial == best[0][work]:
                linewidth = bigLinewidth
                alpha = bigAlpha
        if z is None:
            pathplt(pathFile, transform, nrows, color = C2, linestyle = "solid", alpha = alpha, linewidth = linewidth)
        else:
            pathplt(pathFile, transform, nrows, color = C2, linestyle = "solid", alpha = alpha, linewidth = linewidth, z = z)
    graphtask(P2, C2)
    # Path 3
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(2, work, trial)
        linewidth = 1
        alpha = 0.6
        if best is not None:
            if trial == best[0][work]:
                linewidth = bigLinewidth
                alpha = bigAlpha
        if z is None:
            pathplt(pathFile, transform, nrows, color = C3, linestyle = "solid", alpha = alpha, linewidth = linewidth)
        else:
            pathplt(pathFile, transform, nrows, color = C3, linestyle = "solid", alpha = alpha, linewidth = linewidth, z = z)
    graphtask(P3, C3)
    # Legend
    #ax = plt.gca()
    #custom_lines = [Line2D([0], [0], linestyle = "solid", color = C1, alpha = 1.0, lw = 1.5),
    #                Line2D([0], [0], linestyle = "solid",  color = C2,  alpha = 1.0,  lw = 1.5),
    #                Line2D([0], [0], linestyle = "solid",  color = C3,  alpha = 1.0,  lw = 1.5),
    #                ]
    #custom_labels = ["$\mathcal{T}_1$ trials", "$\mathcal{T}_2$ trials", "$\mathcal{T}_3$ trials",
    #]
    #legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, prop={'size':6}, handlelength = 5)
    #legend.get_frame().set_alpha(6.0)
    #legend.get_frame().set_facecolor("aliceblue")
    #legend.get_frame().set_linewidth(0.0)

# Plot Dijkstra results (cost, times)
resDijkstraFile = "test/results_dijkstra.csv"
dfDijk = pd.read_csv(resDijkstraFile)
dfDijk["Time_Grid"] = dfDijk["Time_Grid"] / 60
dfDijk["Time_Graph"] = dfDijk["Time_Graph"] / 60
#dfDijk = dfDijk.replace(np.nan, 0)
dfDijk_P1 = dfDijk[dfDijk["Path"] == "P1"]
dfDijk_P2 = dfDijk[dfDijk["Path"] == "P2"]
dfDijk_P3 = dfDijk[dfDijk["Path"] == "P3"]
df = dfDijk_P1.groupby(['Work'])
plt.clf()
# get colormap
ncolors = 256
color_array = plt.get_cmap('Purples')(range(ncolors))
# change alpha values
color_array[0:1, -1] = 0.0
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='reward',colors=color_array)
# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)


def pltDijkCost(df):
    fig, ax1 = plt.subplots(figsize=(2.8, 3))
    ax2=ax1.twinx()
    sns.barplot(
        ax = ax1,
        data=df,
        x="Work", y="Cost", hue="Connection", alpha=0.75,
        #ci="sd", palette="dark", alpha=.6, height=6
    )

    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value
            # we change the bar width
            patch.set_width(new_value * 4)
            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)
    change_width(ax1, .05)

    sns.scatterplot(
        ax = ax2,
        data=dfDijk_P1,
        x="Work", y="Time_Graph", hue="Connection",
        alpha = 0.8, marker="*", s = 200, edgecolor="black", linewidth=1,
    )
    ax1.set_ylim(0, 5500)
    ax1.set_ylabel("Path cost")
    ax1.set_xticklabels(["$\mathcal{W}_1$", "$\mathcal{W}_2$", "$\mathcal{W}_3$", "$\mathcal{W}_4$"])
    sns.scatterplot(
        ax = ax2,
        data=dfDijk_P1,
        x="Work", y="Time_Grid", hue="Connection",
        alpha = 0.5, marker="D", s = 200,  edgecolor="black", linewidth=1,
    )
    ax2.set_ylim(0, 25)
    ax2.set_ylabel("Solution time (minutes)")
    ax1.legend().remove()
    ax2.legend().remove()
    #ax1.legend(loc='upper center', bbox_to_anchor=(0.475, 1.15),
    #                 ncol=4, fancybox=False, shadow=False, handletextpad=0.1, frameon=False, handlelength=1)
    #ax2.legend().set_visible(False)
    plt.tight_layout()




def pltReward(rFile):
    rewardGrid = np.flipud(np.loadtxt(rFile))
    region(m)
    img = m.imshow(rewardGrid * rewardGrid, interpolation='nearest', zorder = 100, alpha = 1, cmap = "reward")
    region(m, colors = cm_landonly, z = 1000)

# Plot reward
m = init()
pltReward(rewardFile)
plt.tight_layout()
rasterize_and_save("test/visuals/reward.pdf", dpi=300)
plt.clf()

#P1
pltDijkCost(dfDijk_P1)
plt.tight_layout()
plt.savefig('test/visuals/dijkstra_res_P1.pdf', dpi=300)
plt.clf()
# P2
pltDijkCost(dfDijk_P2)
plt.tight_layout()
plt.savefig('test/visuals/dijkstra_res_P2.pdf', dpi=300)
plt.clf()
# P3
pltDijkCost(dfDijk_P3)
plt.tight_layout()
plt.savefig('test/visuals/dijkstra_res_P3.pdf', dpi=300)
plt.clf()
# Legend
fig, ax = plt.subplots(figsize=(4.5, 1))
snsColors = sns.color_palette()
custom_lines = [Line2D([0], [0], linestyle = "solid", color = snsColors[0], alpha = 1.0, lw = 4),
                    Line2D([0], [0], linestyle = "solid",  color = snsColors[1],  alpha = 1.0,  lw = 4),
                    Line2D([0], [0], linestyle = "solid",  color = snsColors[2],  alpha = 1.0,  lw = 4),
                    Line2D([0], [0], linestyle = "solid",  color = snsColors[3],  alpha = 1.0,  lw = 4),
                    Line2D([], [], color= snsColors[0], marker='*', linestyle='None', markersize=8, label='Purple triangles'),
                    Line2D([], [], color= snsColors[1], marker='*', linestyle='None', markersize=8, label='Purple triangles'),
                    Line2D([], [], color= snsColors[2], marker='*', linestyle='None', markersize=8, label='Purple triangles'),
                    Line2D([], [], color= snsColors[3], marker='*', linestyle='None', markersize=8, label='Purple triangles'),
                    Line2D([], [], color= snsColors[0], marker='D', linestyle='None', markersize=4, label='Purple triangles'),
                    Line2D([], [], color= snsColors[1], marker='D', linestyle='None', markersize=4, label='Purple triangles'),
                    Line2D([], [], color= snsColors[2], marker='D', linestyle='None', markersize=4, label='Purple triangles'),
                    #Line2D([], [], color= snsColors[3], marker='D', linestyle='None', markersize=11, label='Purple triangles'),
                    ]
custom_labels = [
        "Cost, 4-way", "Cost, 8-way", "Cost, 16-way", "Cost, VG",
        "Graph speed, 4-way", "Graph speed, 8-way", "Graph speed, 16-way", "Graph speed, VG",
        "Grid speed, 4-way", "Grid speed, 8-way", "Grid speed, 16-way",
    ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, handlelength = 3)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
ax.axis('off')
plt.tight_layout()
plt.savefig("test/visuals/dijkstra_cost_legend.pdf", dpi=300)
plt.clf()


# Plot Dijkstra results (Astar results are the same)
def pathplt2(pathfile, transform, nrows, color = "white", linestyle = "solid", linewidth = 1, alpha = 1.0):
    path = [grid2world(p[0], p[1], transform, nrows) for p in \
            np.loadtxt(pathfile, delimiter = ",").astype("int")]
    m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle, linewidth = linewidth, alpha = alpha)

def pltdijkstra(work, pltlegend = False):
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    # Path 1
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 1, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 40.0/255.0, 100.0/255.0), linestyle = "solid", linewidth = 2.5, alpha = 1)
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 1, work)
    pathplt2(pathFile, transform, nrows, color = (200.0/255.0, 60.0/255.0, 35.0/255.0), linestyle = "solid", linewidth = 2.5, alpha = 1)
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 1, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 100.0/255.0, 30.0/255.0), linestyle = "solid", linewidth = 2.5, alpha = 1)
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 1, work)
    pathplt2(pathFile, transform, nrows, color = (10.0/255.0, 10.0/255.0, 10.0/255.0), linestyle = "solid", linewidth = 2.5, alpha = 1)
    graphtask(P1, C1, z = 100)
    # Path 2
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 2, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 40.0/255.0, 100.0/255.0), linestyle = "dashed", linewidth = 2.5, alpha = 0.9)
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 2, work)
    pathplt2(pathFile, transform, nrows, color = (200.0/255.0, 60.0/255.0, 35.0/255.0), linestyle = "dashed", linewidth = 2.5, alpha = 0.9)
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 2, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 100.0/255.0, 30.0/255.0), linestyle = "dashed", linewidth = 2.5, alpha = 0.9)
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 2, work)
    pathplt2(pathFile, transform, nrows, color = (10.0/255.0, 10.0/255.0, 10.0/255.0), linestyle = "dashed", linewidth = 2.5, alpha = 0.9)
    graphtask(P2, C2, z = 100)
    # Path 3
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 3, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 40.0/255.0, 100.0/255.0), linestyle = "dotted", linewidth = 2, alpha = 0.85)
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 3, work)
    pathplt2(pathFile, transform, nrows, color = (200.0/255.0, 60.0/255.0, 35.0/255.0), linestyle = "dotted", linewidth = 2, alpha = 0.85)
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 3, work)
    pathplt2(pathFile, transform, nrows, color = (100.0/255.0, 100.0/255.0, 30.0/255.0), linestyle = "dotted", linewidth = 2, alpha = 0.85)
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 3, work)
    pathplt2(pathFile, transform, nrows, color = (10.0/255.0, 10.0/255.0, 10.0/255.0), linestyle = "dotted", linewidth = 2, alpha = 0.85)
    graphtask(P3, C3, z = 100)

# PSO, random, W0
m = init()
region(m)
pltPSO(0, rPSOpathResDir, rPSOpathResStr, trials = TRIALS, best = bestPSO)
plt.title(r'PSO results ($\mathcal{W}_0)$')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_W0.pdf", dpi=300)
plt.clf()
# PSO, random, W1
m = init()
region(m)
pltPSO(1, rPSOpathResDir, rPSOpathResStr, trials = TRIALS, best = bestPSO)
plt.title(r'PSO results ($\mathcal{W}_1)$')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_W1.pdf", dpi=300)
plt.clf()
# PSO, random, W2
m = init()
region(m)
pltPSO(2, rPSOpathResDir, rPSOpathResStr, trials = TRIALS, best = bestPSO)
plt.title(r'PSO results ($\mathcal{W}_2)$')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_W2.pdf", dpi=300)
plt.clf()
# PSO, random, W3
m = init()
region(m)
pltPSO(3, rPSOpathResDir, rPSOpathResStr, trials = TRIALS, best = bestPSO)
plt.title(r'PSO results ($\mathcal{W}_3)$')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_W3.pdf", dpi=300)
plt.clf()
# PSO, random, W4
m = init()
region(m)
pltPSO(4, rPSOpathResDir, rPSOpathResStr, trials = TRIALS, best = bestPSO)
plt.title(r'PSO results ($\mathcal{W}_4)$')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_W4.pdf", dpi=300)
plt.clf()

# PSO, VG init, W0
m = init()
region(m)
pltPSO(0, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS, best = bestPSO_VG)
plt.title(r'PSO results, VG-init ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vg_W0.pdf", dpi=300)
plt.clf()
# PSO, VG init, W1
m = init()
region(m)
pltPSO(1, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS, best = bestPSO_VG)
plt.title(r'PSO results, VG-init ($\mathcal{W}_1$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vg_W1.pdf", dpi=300)
plt.clf()
# PSO, VG init, W2
m = init()
region(m)
pltPSO(2, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS, best = bestPSO_VG)
plt.title(r'PSO results, VG-init ($\mathcal{W}_2$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vg_W2.pdf", dpi=300)
plt.clf()
# PSO, VG init, W3
m = init()
region(m)
pltPSO(3, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS, best = bestPSO_VG)
plt.title(r'PSO results, VG-init ($\mathcal{W}_3$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vg_W3.pdf", dpi=300)
plt.clf()
# PSO, VG init, W4
m = init()
region(m)
pltPSO(4, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS, best = bestPSO_VG)
plt.title(r'PSO results, VG-init ($\mathcal{W}_4$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vg_W4.pdf", dpi=300)
plt.clf()

# Dijksta, graph, W0
m = init()
region(m)
pltdijkstra(0)
plt.title(r'Dijkstra results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/dijkstra_graph_W0.pdf", dpi=300)
plt.clf()
# Dijksta, graph,p W1
m = init()
region(m)
pltdijkstra(1)
plt.title(r'Dijkstra results ($\mathcal{W}_1$)')
plt.tight_layout()
rasterize_and_save("test/visuals/dijkstra_graph_W1.pdf", dpi=300)
plt.clf()
# Dijksta, graph, W2
m = init()
region(m)
pltdijkstra(2)
plt.title(r'Dijkstra results ($\mathcal{W}_2$)')
plt.tight_layout()
rasterize_and_save("test/visuals/dijkstra_graph_W2.pdf", dpi=300)
plt.clf()
# Dijksta, graph, W3
m = init()
region(m)
pltdijkstra(3)
plt.title(r'Dijkstra results ($\mathcal{W}_3$)')
plt.tight_layout()
rasterize_and_save("test/visuals/dijkstra_graph_W3.pdf", dpi=300)
plt.clf()
# Dijksta, graph, W4
m = init()
region(m)
pltdijkstra(4)
plt.title(r'Dijkstra results ($\mathcal{W}_4$)')
plt.tight_layout()
rasterize_and_save("test/visuals/dijkstra_graph_W4.pdf", dpi=300)
plt.clf()


# Legends
fig, ax = plt.subplots(figsize=(3.5, 0.75))
custom_lines = [
    Line2D([0], [0], linestyle = "solid", color =  (100.0/255.0, 40.0/255.0, 100.0/255.0), alpha = 1.0, lw = 3),
    Line2D([0], [0], linestyle = "solid",  color = (200.0/255.0, 60.0/255.0, 35.0/255.0),  alpha = 1.0,  lw =3),
    Line2D([0], [0], linestyle = "solid",  color = (100.0/255.0, 100.0/255.0, 30.0/255.0),  alpha = 1.0,  lw = 3),
    Line2D([0], [0], linestyle = "solid", color =  (10.0/255.0, 10.0/255.0, 10.0/255.0), alpha = 1.0, lw = 3),
    Line2D([0], [0], linestyle = "dashed", color =  (100.0/255.0, 40.0/255.0, 100.0/255.0), alpha = 1.0, lw = 3),
    Line2D([0], [0], linestyle = "dashed",  color = (200.0/255.0, 60.0/255.0, 35.0/255.0),  alpha = 1.0,  lw = 3),
    Line2D([0], [0], linestyle = "dashed",  color = (100.0/255.0, 100.0/255.0, 30.0/255.0),  alpha = 1.0,  lw = 3),
    Line2D([0], [0], linestyle = "dashed", color =  (10.0/255.0, 10.0/255.0, 10.0/255.0), alpha = 1.0, lw = 3),
    Line2D([0], [0], linestyle = "dotted", color =   (100.0/255.0, 40.0/255.0, 100.0/255.0), alpha = 1.0, lw = 3),
    Line2D([0], [0], linestyle = "dotted",  color =  (200.0/255.0, 60.0/255.0, 35.0/255.0),  alpha = 1.0,  lw = 3),
    Line2D([0], [0], linestyle = "dotted",  color =  (100.0/255.0, 100.0/255.0, 30.0/255.0),  alpha = 1.0,  lw = 3),
    Line2D([0], [0], linestyle = "dotted", color =   (10.0/255.0, 10.0/255.0, 10.0/255.0), alpha = 1.0, lw = 3),
                        ]
custom_labels = ["$\mathcal{T}_1$, 4-way", "$\mathcal{T}_1$, 8-way", "$\mathcal{T}_1$, 16-way", "$\mathcal{T}_1$, VG",
                         "$\mathcal{T}_2$, 4-way", "$\mathcal{T}_2$, 8-way", "$\mathcal{T}_2$, 16-way", "$\mathcal{T}_2$, VG",
                         "$\mathcal{T}_3$, 4-way", "$\mathcal{T}_3$, 8-way", "$\mathcal{T}_3$, 16-way", "$\mathcal{T}_3$, VG",
        ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3,  handlelength = 3)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
ax.axis('off')
plt.tight_layout()
plt.savefig("test/visuals/dijkstra_legend.pdf", dpi=300)
plt.clf()

fig, ax = plt.subplots(figsize=(1.2, 0.75))
custom_lines = [Line2D([0], [0], linestyle = "solid", color = C1, alpha = 1.0, lw = 5),
                    Line2D([0], [0], linestyle = "solid",  color = C2,  alpha = 1.0,  lw = 3),
                    Line2D([0], [0], linestyle = "solid",  color = C3,  alpha = 1.0,  lw = 3),
                    ]
custom_labels = ["$\mathcal{T}_1$ trials", "$\mathcal{T}_2$ trials", "$\mathcal{T}_3$ trials",
    ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 1, handlelength = 5)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
ax.axis('off')
plt.tight_layout()
plt.savefig("test/visuals/pso_legend.pdf", dpi=300)
plt.clf()


fig, ax = plt.subplots(figsize=(12, 3))
custom_lines = [Line2D([0], [0], linestyle = "dashed", color = C1, alpha = 1.0, lw = 5),
                    Line2D([0], [0], linestyle = "dashed",  color = C3,  alpha = 1.0,  lw = 5),
                    Line2D([0], [0], linestyle = "dashed",  color = C2,  alpha = 1.0,  lw = 5),
                    ]
custom_labels = ["4-way cost", "8-way cost", "16-way cost",
    ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, handlelength = 4)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.savefig("test/visuals/pso_conv_legend.pdf")
plt.clf()

# Plot VG-based initial populations
ds = gdal.Open(regionRasterFile, GA_ReadOnly)
transform = ds.GetGeoTransform()
nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
# Path 1
m = init()
region(m)
initPopFile = "test/outputs/getNpaths/paths_P1.txt"
with open(initPopFile) as f:
    lines = [line.rstrip() for line in f]
    lines.reverse()
initPaths = []
for line in lines:
    line = line.replace(";", ",").replace("(", "").replace(")", "")
    nums = [int(l) for l in line.split(",")][2:-2]
    initPaths.append(nums)
for ip in initPaths:
    lp = []
    x = ip[1::2]
    y = ip[::2]
    for i in range(len(x)):
        lp.append(grid2world(y[i], x[i], transform, nrows)[0])
        lp.append(grid2world(y[i], x[i], transform, nrows)[1])
    lx = lp[1::2]
    ly = lp[::2]
    plt.plot(lx, ly)
plt.tight_layout()
rasterize_and_save("test/visuals/getNpaths_P1.pdf", dpi=300)
plt.clf()
# Path 2
m = init()
region(m)
initPopFile = "test/outputs/getNpaths/paths_P2.txt"
with open(initPopFile) as f:
    lines = [line.rstrip() for line in f]
    lines.reverse()
initPaths = []
for line in lines:
    line = line.replace(";", ",").replace("(", "").replace(")", "")
    nums = [int(l) for l in line.split(",")][2:-2]
    initPaths.append(nums)
for ip in initPaths:
    lp = []
    x = ip[1::2]
    y = ip[::2]
    for i in range(len(x)):
        lp.append(grid2world(y[i], x[i], transform, nrows)[0])
        lp.append(grid2world(y[i], x[i], transform, nrows)[1])
    lx = lp[1::2]
    ly = lp[::2]
    plt.plot(lx, ly)
plt.tight_layout()
rasterize_and_save("test/visuals/getNpaths_P2.pdf", dpi=300)
plt.clf()
# Path 3
m = init()
region(m)
initPopFile = "test/outputs/getNpaths/paths_P3.txt"
with open(initPopFile) as f:
    lines = [line.rstrip() for line in f]
    lines.reverse()
initPaths = []
for line in lines:
    line = line.replace(";", ",").replace("(", "").replace(")", "")
    nums = [int(l) for l in line.split(",")][2:-2]
    initPaths.append(nums)
for ip in initPaths:
    lp = []
    x = ip[1::2]
    y = ip[::2]
    for i in range(len(x)):
        lp.append(grid2world(y[i], x[i], transform, nrows)[0])
        lp.append(grid2world(y[i], x[i], transform, nrows)[1])
    lx = lp[1::2]
    ly = lp[::2]
    plt.plot(lx, ly)
plt.tight_layout()
rasterize_and_save("test/visuals/getNpaths_P3.pdf", dpi=300)
plt.clf()

# Plot visibility graphs
# P1
m = init()
region(m)
graphplt(visgraphFile_P1)
graphtask(P1, C1, z = 100)
plt.title(r'Visibility graph (P1)')
plt.tight_layout()
plt.savefig(visgraphOut_P1)
plt.clf()
# P2
m = init()
region(m)
graphplt(visgraphFile_P2)
graphtask(P2, C2, z = 100)
plt.title(r'Visibility graph (P2)')
plt.tight_layout()
plt.savefig(visgraphOut_P2)
plt.clf()
# P3
m = init()
region(m)
graphplt(visgraphFile_P3)
graphtask(P3, C3, z = 100)
plt.title(r'Visibility graph (P3)')
plt.tight_layout()
plt.savefig(visgraphOut_P3)
plt.clf()

# Plot Dijkstra results (Astar results are the same)
def pltdijkstra(work):
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    # Path 1
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 1, work)
    pathplt(pathFile, transform, nrows, color = C1, linestyle = "dashdot")
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 1, work)
    pathplt(pathFile, transform, nrows, color = C1, linestyle = "dashed")
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 1, work)
    pathplt(pathFile, transform, nrows, color = C1, linestyle = "dotted")
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 1, work)
    pathplt(pathFile, transform, nrows, color = C1, linestyle = "solid")
    graphtask(P1, C1)
    # Path 2
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 2, work)
    pathplt(pathFile, transform, nrows, color = C2, linestyle = "dashdot")
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 2, work)
    pathplt(pathFile, transform, nrows, color = C2, linestyle = "dashed")
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 2, work)
    pathplt(pathFile, transform, nrows, color = C2, linestyle = "dotted")
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 2, work)
    pathplt(pathFile, transform, nrows, color = C2, linestyle = "solid")
    graphtask(P2, C2, z = 100)
    # Path 3
    pathFile = gpathResDir + "/" + gpathResStr.format(4, 3, work)
    pathplt(pathFile, transform, nrows, color = C3, linestyle = "dashdot")
    pathFile = gpathResDir + "/" + gpathResStr.format(8, 3, work)
    pathplt(pathFile, transform, nrows, color = C3, linestyle = "dashed")
    pathFile = gpathResDir + "/" + gpathResStr.format(16, 3, work)
    pathplt(pathFile, transform, nrows, color = C3, linestyle = "dotted")
    pathFile = vgpathResDir + "/" + vgpathResStr.format("vg", 3, work)
    pathplt(pathFile, transform, nrows, color = C3, linestyle = "solid")
    graphtask(P3, C3)
    # Legend
    ax = plt.gca()
    custom_lines = [Line2D([0], [0], linestyle = "dashdot", color = C1, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "dashed",  color = C1,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "dotted",  color = C1,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "solid", color = C1, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "dashdot", color = C2, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "dashed",  color = C2,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "dotted",  color = C2,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "solid", color = C2, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "dashdot", color = C3, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "dashed",  color = C3,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "dotted",  color = C3,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "solid", color = C3, alpha = 1.0, lw = 1.5),
                    ]
    custom_labels = ["$\mathcal{T}_1$, 4-way", "$\mathcal{T}_1$, 8-way", "$\mathcal{T}_1$, 16-way", "$\mathcal{T}_1$, VG",
                     "$\mathcal{T}_2$, 4-way", "$\mathcal{T}_2$, 8-way", "$\mathcal{T}_2$, 16-way", "$\mathcal{T}_2$, VG",
                     "$\mathcal{T}_3$, 4-way", "$\mathcal{T}_3$, 8-way", "$\mathcal{T}_3$, 16-way", "$\mathcal{T}_3$, VG",
    ]
    #legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, prop={'size':6}, handlelength = 5)
    #legend.get_frame().set_alpha(6.0)
    #legend.get_frame().set_facecolor("aliceblue")
    #legend.get_frame().set_linewidth(0.0)

# Plot water currents
# W1 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1)
plt.title(r'Time = +0h ($\mathcal{W}_1$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W1_0, dpi=300)
plt.clf()
# W1 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1, band = 2)
plt.title(r'Time = +1h ($\mathcal{W}_1$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W1_1, dpi=300)
plt.clf()
# W1 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1, band = 3)
plt.title(r'Time = +2h ($\mathcal{W}_1$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W1_2, dpi=300)
plt.clf()

# W2 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2)
plt.title(r'Time = +0h ($\mathcal{W}_2$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W2_0, dpi=300)
plt.clf()
# W2 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2, band = 2)
plt.title(r'Time = +1h ($\mathcal{W}_2$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W2_1, dpi=300)
plt.clf()
# W2 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2, band = 3)
plt.title(r'Time = +2h ($\mathcal{W}_2$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W2_2, dpi=300)
plt.clf()

# W3 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3)
plt.title(r'Time = +0h ($\mathcal{W}_3$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W3_0, dpi=300)
plt.clf()
# W3 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3, band = 2)
plt.title(r'Time = +1h ($\mathcal{W}_3$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W3_1, dpi=300)
plt.clf()
# W3 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3, band = 3)
plt.title(r'Time = +2h ($\mathcal{W}_3$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W3_2, dpi=300)
plt.clf()

# W4 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4)
plt.title(r'Time = +0h ($\mathcal{W}_4$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W4_0, dpi=300)
plt.clf()
# W4 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4, band = 2)
plt.title(r'Time = +1h ($\mathcal{W}_4$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W4_1, dpi=300)
plt.clf()
# W4 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4, band = 3)
plt.title(r'Time = +2h ($\mathcal{W}_4$)')
plt.tight_layout()
rasterize_and_save(currentsRasterOut_W4_2, dpi=300)
plt.clf()


rc = (153/255, 107/255, 184/255, 1)
vgc = (199/255, 169/255, 0/255, 1)
# PSO convergence (VG init)
# (Time-based)
# PSO convergence - path 1
fig, axs = plt.subplots(1, 4, figsize = (5.5, 2.25))
# Work 1
pltPSOconvergence_2(0, 1, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 234/500,  ax = axs[0], trials = TRIALS, color = rc)
pltPSOconvergence_2(0, 1, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 236/500,  ax = axs[0], trials = TRIALS, color = vgc)
axs[0].axhline(y=4815.3878, color=C1, linestyle='--', label = "4-way")
axs[0].axvline(x=161.703, color=C1, linestyle=':', label = "4-way")
axs[0].axhline(y=4400.0321, color=C2, linestyle='--', label = "8-way")
axs[0].axvline(x=409.625, color=C2, linestyle=':', label = "8-way")
axs[0].axhline(y=4325.1572, color=C3, linestyle='--', label = "16-way")
axs[0].axvline(x=1112.597, color=C3, linestyle=':', label = "16-way")
axs[0].set_ylim(4000, 8000)
axs[0].set_xlim(0,1200)
axs[0].set_title("$\mathcal{T}_1, \mathcal{W}_1$")

# Work 2
pltPSOconvergence_2(0, 2, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 211/500, ax = axs[1], trials = TRIALS, color = rc)
pltPSOconvergence_2(0, 2, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 196/500, ax = axs[1], trials = TRIALS, color = vgc)
axs[1].axhline(y=3507.0305, color=C1, linestyle='--', label = "4-way")
axs[1].axvline(x=177.138, color=C1, linestyle=':', label = "4-way")
axs[1].axhline(y=2097.3748, color=C2, linestyle='--', label = "8-way")
axs[1].axvline(x=317.843, color=C2, linestyle=':', label = "8-way")
axs[1].axhline(y=1675.0157, color=C3, linestyle='--', label = "16-way")
axs[1].axvline(x=949.563, color=C3, linestyle=':', label = "16-way")
axs[1].set_ylim(1500, 8000)
axs[1].set_xlim(0,1050)
axs[1].get_yaxis().set_visible(False)
axs[1].set_title("$\mathcal{T}_1, \mathcal{W}_2$")
# Work 3
pltPSOconvergence_2(0, 3, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 221/500, ax = axs[2], trials = TRIALS, color = rc)
pltPSOconvergence_2(0, 3, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 195/500, ax = axs[2], trials = TRIALS, color = vgc)
axs[2].axhline(y=3196.3011, color=C1, linestyle='--', label = "4-way")
axs[2].axvline(x=155.049, color=C1, linestyle=':', label = "4-way")
axs[2].axhline(y=1701.4989, color=C2, linestyle='--', label = "8-way")
axs[2].axvline(x=348.816, color=C2, linestyle=':', label = "8-way")
axs[2].axhline(y=1151.26, color=C3, linestyle='--', label = "16-way")
axs[2].axvline(x=983.849, color=C3, linestyle=':', label = "16-way")
axs[2].set_ylim(1050, 8000)
axs[2].set_xlim(0,1050)
axs[2].get_yaxis().set_visible(False)
axs[2].set_title("$\mathcal{T}_1, \mathcal{W}_3$")
# Work 4
pltPSOconvergence_2(0, 4, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 214/500, ax = axs[3], trials = TRIALS, color = rc)
pltPSOconvergence_2(0, 4, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 197/500, ax = axs[3], trials = TRIALS, color = vgc)
axs[3].axhline(y=5025.9262, color=C1, linestyle='--', label = "4-way")
axs[3].axvline(x=197.270, color=C1, linestyle=':', label = "4-way")
axs[3].axhline(y=4330.592, color=C2, linestyle='--', label = "8-way")
axs[3].axvline(x=482.383, color=C2, linestyle=':', label = "8-way")
axs[3].axhline(y=4151.432, color=C3, linestyle='--', label = "16-way")
axs[3].axvline(x=1295.317, color=C3, linestyle=':', label = "16-way")
axs[3].set_ylim(4000, 8000)
axs[3].set_xlim(0,1400)
axs[3].get_yaxis().set_visible(False)
axs[3].set_title("$\mathcal{T}_1, \mathcal{W}_4$")
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                        #bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.savefig("test/visuals/pso_vg_P1_conv_2.pdf", dpi=300)
plt.clf()

# PSO convergence - path 2
fig, axs = plt.subplots(1, 4, figsize = (5.5, 2.25))
# Work 1
pltPSOconvergence_2(1, 1, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 252/500, ax = axs[0], trials = TRIALS, color = rc)
pltPSOconvergence_2(1, 1, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 278/500, ax = axs[0], trials = TRIALS, color = vgc)
axs[0].axhline(y=4590.4769, color=C1, linestyle='--', label = "4-way")
axs[0].axvline(x=198, color=C1, linestyle=':', label = "4-way")
axs[0].axhline(y=3900.0763, color=C2, linestyle='--', label = "8-way")
axs[0].axvline(x=452, color=C2, linestyle=':', label = "8-way")
axs[0].axhline(y=3745.1477, color=C3, linestyle='--', label = "16-way")
axs[0].axvline(x=1236, color=C3, linestyle=':', label = "16-way")
axs[0].set_ylim(3500, 8000)
axs[0].set_xlim(0,1350)
axs[0].set_title("$\mathcal{T}_2, \mathcal{W}_1$")
# Work 2
pltPSOconvergence_2(1, 2, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 225/500, ax = axs[1], trials = TRIALS, color = rc)
pltPSOconvergence_2(1, 2, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 208/500, ax = axs[1], trials = TRIALS, color = vgc)
axs[1].axhline(y=4327.5204, color=C1, linestyle='--', label = "4-way")
axs[1].axvline(x=195, color=C1, linestyle=':', label = "4-way")
axs[1].axhline(y=3082.7551, color=C2, linestyle='--', label = "8-way")
axs[1].axvline(x=458, color=C2, linestyle=':', label = "8-way")
axs[1].axhline(y=2773.9114, color=C3, linestyle='--', label = "16-way")
axs[1].axvline(x=1330, color=C3, linestyle=':', label = "16-way")
axs[1].set_ylim(2500, 8000)
axs[1].set_xlim(0,1450)
axs[1].get_yaxis().set_visible(False)
axs[1].set_title("$\mathcal{T}_2, \mathcal{W}_2$")
# Work 3
pltPSOconvergence_2(1, 3, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 231/500, ax = axs[2], trials = TRIALS, color = rc)
pltPSOconvergence_2(1, 3, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 207/500, ax = axs[2], trials = TRIALS, color = vgc)
axs[2].axhline(y=4201.3305, color=C1, linestyle='--', label = "4-way")
axs[2].axvline(x=212, color=C1, linestyle=':', label = "4-way")
axs[2].axhline(y=2858.711, color=C2, linestyle='--', label = "8-way")
axs[2].axvline(x=507, color=C2, linestyle=':', label = "8-way")
axs[2].axhline(y=2636.9706, color=C3, linestyle='--', label = "16-way")
axs[2].axvline(x=1585, color=C3, linestyle=':', label = "16-way")
axs[2].set_ylim(2500, 8000)
axs[2].set_xlim(0,1650)
axs[2].get_yaxis().set_visible(False)
axs[2].set_title("$\mathcal{T}_2, \mathcal{W}_3$")
# Work 4
pltPSOconvergence_2(1, 4, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 234/500, ax = axs[3], trials = TRIALS, color = rc)
pltPSOconvergence_2(1, 4, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 198/500, ax = axs[3], trials = TRIALS, color = vgc)
axs[3].axhline(y=4768.2483, color=C1, linestyle='--', label = "4-way")
axs[3].axvline(x=179, color=C1, linestyle=':', label = "4-way")
axs[3].axhline(y=3646.9265, color=C2, linestyle='--', label = "8-way")
axs[3].axvline(x=450, color=C2, linestyle=':', label = "8-way")
axs[3].axhline(y=3440.4205, color=C3, linestyle='--', label = "16-way")
axs[3].axvline(x=1045, color=C3, linestyle=':', label = "16-way")
axs[3].set_ylim(3250, 8000)
axs[3].set_xlim(0,1100)
axs[3].get_yaxis().set_visible(False)
axs[3].set_title("$\mathcal{T}_2, \mathcal{W}_4$")
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.savefig("test/visuals/pso_vg_P2_conv_2.pdf", dpi=300)
plt.clf()
# PSO convergence - path 3
fig, axs = plt.subplots(1, 4, figsize = (5.5, 2.25))
# Work 1
pltPSOconvergence_2(2, 1, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 167/500, ax = axs[0], trials = TRIALS, color = rc)
pltPSOconvergence_2(2, 1, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 218/500, ax = axs[0], trials = TRIALS, color = vgc)
axs[0].axhline(y=2447.6688, color='C1', linestyle='--', label = "4-way")
axs[0].axvline(x=61, color='C1', linestyle=':', label = "4-way")
axs[0].axhline(y=2058.698, color=C2, linestyle='--', label = "8-way")
axs[0].axvline(x=130, color=C2, linestyle=':', label = "8-way")
axs[0].axhline(y=1922.7959, color=C3, linestyle='--', label = "16-way")
axs[0].axvline(x=333, color=C3, linestyle=':', label = "16-way")
axs[0].set_ylim(1700, 8000)
axs[0].set_xlim(0,350)
axs[0].set_title("$\mathcal{T}_3, \mathcal{W}_1$")
# Work 2
pltPSOconvergence_2(2, 2, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 159/500, ax = axs[1], trials = TRIALS, color = rc)
pltPSOconvergence_2(2, 2, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 168/500, ax = axs[1], trials = TRIALS, color = vgc)
axs[1].axhline(y=2885.3459, color=C1, linestyle='--', label = "4-way")
axs[1].axvline(x=76, color=C1, linestyle=':', label = "4-way")
axs[1].axhline(y=2584.5256, color=C2, linestyle='--', label = "8-way")
axs[1].axvline(x=236, color=C2, linestyle=':', label = "8-way")
axs[1].axhline(y=2489.5882, color=C3, linestyle='--', label = "16-way")
axs[1].axvline(x=701, color=C3, linestyle=':', label = "16-way")
axs[1].set_ylim(2200, 8000)
axs[1].set_xlim(0,750)
axs[1].get_yaxis().set_visible(False)
axs[1].set_title("$\mathcal{T}_3, \mathcal{W}_2$")
# Work 3
pltPSOconvergence_2(2, 3, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 156/500, ax = axs[2], trials = TRIALS, color = rc)
pltPSOconvergence_2(2, 3, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 175/500, ax = axs[2], trials = TRIALS, color = vgc)
axs[2].axhline(y=2948.1269, color=C1, linestyle='--', label = "4-way")
axs[2].axvline(x=77, color=C1, linestyle=':', label = "4-way")
axs[2].axhline(y=2675.4716, color=C2, linestyle='--', label = "8-way")
axs[2].axvline(x=256, color=C2, linestyle=':', label = "8-way")
axs[2].axhline(y=2593.5196, color=C3, linestyle='--', label = "16-way")
axs[2].axvline(x=743, color=C3, linestyle=':', label = "16-way")
axs[2].set_ylim(2300, 8000)
axs[2].set_xlim(0,775)
axs[2].get_yaxis().set_visible(False)
axs[2].set_title("$\mathcal{T}_3, \mathcal{W}_3$")
# Work 4
pltPSOconvergence_2(2, 4, r_10kgens_PSOpathResDir, r_10kgens_PSOstatResStr, 169/500, ax = axs[3], trials = TRIALS, color = rc)
pltPSOconvergence_2(2, 4, vg_10kgens_PSOpathResDir, vg_10kgens_PSOstatResStr, 168/500, ax = axs[3], trials = TRIALS, color = vgc)
axs[3].axhline(y=2698, color=C1, linestyle='--', label = "4-way")
axs[3].axvline(x=70, color=C1, linestyle=':', label = "4-way")
axs[3].axhline(y=2687.1266, color=C2, linestyle='--', label = "8-way")
axs[3].axvline(x=180, color=C2, linestyle=':', label = "8-way")
axs[3].axhline(y=2109.6257, color=C3, linestyle='--', label = "16-way")
axs[3].axvline(x=453, color=C3, linestyle=':', label = "16-way")
axs[3].set_ylim(1900, 8000)
axs[3].set_xlim(0,480)
axs[3].get_yaxis().set_visible(False)
axs[3].set_title("$\mathcal{T}_3, \mathcal{W}_4$")
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.tight_layout()
plt.savefig("test/visuals/pso_vg_P3_conv_2.pdf", dpi=300)
plt.clf()

fig, ax = plt.subplots(figsize=(5.25, 0.5))
custom_lines = [
    Line2D([0], [0], linestyle = "solid", color = rc, alpha = 1.0, lw = 3.75),
    Line2D([0], [0], linestyle = "solid", color = vgc, alpha = 1.0, lw = 3.75),
    Line2D([0], [0], linestyle = "dashed", color = C1, alpha = 1.0, lw = 3.75),
    Line2D([0], [0], linestyle = ":", color = C1, alpha = 1.0, lw = 3.75),
    Line2D([0], [0], linestyle = "dashed",  color = C3,  alpha = 1.0,  lw = 3.75),
    Line2D([0], [0], linestyle = ":",  color = C3,  alpha = 1.0,  lw = 3.75),
    Line2D([0], [0], linestyle = "dashed",  color = C2,  alpha = 1.0,  lw = 3.75),
    Line2D([0], [0], linestyle = ":",  color = C2,  alpha = 1.0,  lw = 3.75),
                    ]
custom_labels = [
    '${PSO}_{R}$', '${PSO}_{VG}$',
    "4-way cost", "4-way time", "8-way cost",
    "8-way time", "16-way cost", "16-way time",
    ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 4, handlelength = 5.25)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
ax.axis('off')
plt.tight_layout()
plt.savefig("test/visuals/pso_conv_legend_2.pdf", dpi=300)
plt.clf()

# Plot PSO with reward
# PSO reward=500, random, W0
m = init()
pltReward(rewardFile)
pltPSO(0, r_r_PSOpathResDir, r_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R500_W0.pdf", dpi=300)
plt.clf()
# PSO reward=500, random, W1
m = init()
pltReward(rewardFile)
pltPSO(1, r_r_PSOpathResDir, r_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R500_W1.pdf", dpi=300)
plt.clf()
# PSO reward=500, random, W2
m = init()
pltReward(rewardFile)
pltPSO(2, r_r_PSOpathResDir, r_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R500_W2.pdf", dpi=300)
plt.clf()
# PSO reward=500, random, W3
m = init()
pltReward(rewardFile)
pltPSO(3, r_r_PSOpathResDir, r_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R500_W3.pdf", dpi=300)
plt.clf()
# PSO reward=500, random, W4
m = init()
pltReward(rewardFile)
pltPSO(4, r_r_PSOpathResDir, r_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R500_W4.pdf", dpi=300)
plt.clf()
# PSO reward=1000, random, W0
m = init()
pltReward(rewardFile)
pltPSO(0, r_r_PSOpathResDir, r_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R1000_W0.pdf", dpi=300)
plt.clf()
# PSO reward=1000, random, W1
m = init()
pltReward(rewardFile)
pltPSO(1, r_r_PSOpathResDir, r_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R1000_W1.pdf", dpi=300)
plt.clf()
# PSO reward=1000, random, W2
m = init()
pltReward(rewardFile)
pltPSO(2, r_r_PSOpathResDir, r_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R1000_W2.pdf", dpi=300)
plt.clf()
# PSO reward=1000, random, W3
m = init()
pltReward(rewardFile)
pltPSO(3, r_r_PSOpathResDir, r_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R1000_W3.pdf", dpi=300)
plt.clf()
# PSO reward=1000, random, W4
m = init()
pltReward(rewardFile)
pltPSO(4, r_r_PSOpathResDir, r_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_rand_R1000_W4.pdf", dpi=300)
plt.clf()

# Plot PSO with reward
# PSO reward=500, VG, W0
m = init()
pltReward(rewardFile)
pltPSO(0, vg_r_PSOpathResDir, vg_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR500_W0.pdf", dpi=300)
plt.clf()
# PSO reward=500, VG, W1
m = init()
pltReward(rewardFile)
pltPSO(1, vg_r_PSOpathResDir, vg_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR500_W1.pdf", dpi=300)
plt.clf()
# PSO reward=500, VG, W2
m = init()
pltReward(rewardFile)
pltPSO(2, vg_r_PSOpathResDir, vg_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR500_W2.pdf", dpi=300)
plt.clf()
# PSO reward=500, VG, W3
m = init()
pltReward(rewardFile)
pltPSO(3, vg_r_PSOpathResDir, vg_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR500_W3.pdf", dpi=300)
plt.clf()
# PSO reward=500, VG, W4
m = init()
pltReward(rewardFile)
pltPSO(4, vg_r_PSOpathResDir, vg_r500_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r500)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR500_W4.pdf", dpi=300)
plt.clf()
# PSO reward=1000, VG, W0
m = init()
pltReward(rewardFile)
pltPSO(0, vg_r_PSOpathResDir, vg_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR1000_W0.pdf", dpi=300)
plt.clf()
# PSO reward=1000, VG, W1
m = init()
pltReward(rewardFile)
pltPSO(1, vg_r_PSOpathResDir, vg_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR1000_W1.pdf", dpi=300)
plt.clf()
# PSO reward=1000, VG, W2
m = init()
pltReward(rewardFile)
pltPSO(2, vg_r_PSOpathResDir, vg_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR1000_W2.pdf", dpi=300)
plt.clf()
# PSO reward=1000, VG, W3
m = init()
pltReward(rewardFile)
pltPSO(3, vg_r_PSOpathResDir, vg_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR1000_W3.pdf", dpi=300)
plt.clf()
# PSO reward=1000, VG, W4
m = init()
pltReward(rewardFile)
pltPSO(4, vg_r_PSOpathResDir, vg_r1000_PSOpathResStr, trials = TRIALS, z = 10000, best = bestPSO_VG_r1000)
plt.title(r'PSO results ($\mathcal{W}_0$)')
plt.tight_layout()
rasterize_and_save("test/visuals/pso_vgR1000_W4.pdf", dpi=300)
plt.clf()
# Legend
fig, ax = plt.subplots(figsize=(3.5, 0.4))
custom_lines = [Line2D([0], [0], linestyle = "solid", color = C1, alpha = 1.0, lw = 5),
                    Line2D([0], [0], linestyle = "solid",  color = C2,  alpha = 1.0,  lw = 5),
                    Line2D([0], [0], linestyle = "solid",  color = C3,  alpha = 1.0,  lw = 5),
                    ]
custom_labels = ["$\mathcal{T}_1$ trials", "$\mathcal{T}_2$ trials", "$\mathcal{T}_3$ trials",
    ]
legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, handlelength = 4)
legend.get_frame().set_alpha(6.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_linewidth(0.0)
ax.axis('off')
plt.tight_layout()
plt.savefig("test/visuals/pso_legend_2.pdf")
plt.clf()


# PSO convergence (random init)
# PSO convergence - path 1
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(0, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4815.3878, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=4400.0321, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=4325.1572, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(0, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=3507.0305, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=2097.3748, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=1675.0157, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(0, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=3196.3011, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=1701.4989, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=1151.26, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(0, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=5025.9262, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=4330.592, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=4151.432, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_rand_P1_conv.pdf", dpi=300)
plt.clf()
# PSO convergence - path 2
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(1, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4590.4769, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=3900.0763, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=3745.1477, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(1, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=4327.5204, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=3082.7551, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=2773.9114, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(1, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=4201.3305, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=2858.711, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=2636.9706, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(1, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=4768.2483, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=3646.9265, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=3440.4205, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_rand_P2_conv.pdf", dpi=300)
plt.clf()
# PSO convergence - path 3
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(2, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=2447.6688, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=2058.698, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=1922.7959, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(2, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=2885.3459, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=2584.5256, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=2489.5882, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(2, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=2948.1269, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=2675.4716, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=2593.5196, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(2, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=2698, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=2687.1266, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=2109.6257, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_rand_P3_conv.pdf", dpi=300)
plt.clf()

# PSO convergence (VG init)
# PSO convergence - path 1
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(0, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4815.3878, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=4400.0321, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=4325.1572, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(0, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=3507.0305, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=2097.3748, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=1675.0157, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(0, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=3196.3011, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=1701.4989, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=1151.26, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(0, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=5025.9262, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=4330.592, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=4151.432, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                        #bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_vg_P1_conv.pdf", dpi=300)
plt.clf()
# PSO convergence - path 2
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(1, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4590.4769, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=3900.0763, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=3745.1477, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(1, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=4327.5204, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=3082.7551, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=2773.9114, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(1, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=4201.3305, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=2858.711, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=2636.9706, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(1, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=4768.2483, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=3646.9265, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=3440.4205, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
#plt.subplots_adjust(wspace = 0.1)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_vg_P2_conv.pdf", dpi=300)
plt.clf()
# PSO convergence - path 3
fig, axs = plt.subplots(1, 4, figsize = (5.5, 4))
# Work 1
pltPSOconvergence(2, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=2447.6688, color=C1, linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=2058.698, color=C2, linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=1922.7959, color=C3, linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("$\mathcal{W}_1$")
# Work 2
pltPSOconvergence(2, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=2885.3459, color=C1, linestyle='--', label = "4-way")
axs[1].axhline(y=2584.5256, color=C2, linestyle='--', label = "8-way")
axs[1].axhline(y=2489.5882, color=C3, linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("$\mathcal{W}_2$")
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(2, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=2948.1269, color=C1, linestyle='--', label = "4-way")
axs[2].axhline(y=2675.4716, color=C2, linestyle='--', label = "8-way")
axs[2].axhline(y=2593.5196, color=C3, linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("$\mathcal{W}_3$")
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(2, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=2698, color=C1, linestyle='--', label = "4-way")
axs[3].axhline(y=2687.1266, color=C2, linestyle='--', label = "8-way")
axs[3].axhline(y=2109.6257, color=C3, linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("$\mathcal{W}_4$")
axs[3].get_yaxis().set_visible(False)
#axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
#                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.25)
plt.savefig("test/visuals/pso_vg_P3_conv.pdf", dpi=300)
plt.clf()




###################
# Unused in paper #
###################

# Exit because we don't want these
exit(0)

# Plot region
m = init()
region(m)
plt.title(r'Boston Harbor')
plt.tight_layout()
rasterize_and_save(regionOut, dpi=300)
plt.clf()

# Plot region shapefile
poly()
plt.title(r'Boston Harbor - polygons')
plt.tight_layout()
rasterize_and_save(polyOut, dpi=300)
plt.clf()

# PSO reward & work histograms
# W1 - P1
histPSO(0, 1, "test/visuals/pso_hist_P1_W1", TRIALS)
# W1- P2
histPSO(1, 1, "test/visuals/pso_hist_P2_W1", TRIALS)
# W1 - P3
histPSO(2, 1, "test/visuals/pso_hist_P3_W1", TRIALS)
# W1 - P1
histPSO(0, 2, "test/visuals/pso_hist_P1_W2", TRIALS)
# W2- P2
histPSO(1, 2, "test/visuals/pso_hist_P2_W2", TRIALS)
# W2 - P3
histPSO(2, 2, "test/visuals/pso_hist_P3_W2", TRIALS)
# W2 - P1
histPSO(0, 3, "test/visuals/pso_hist_P1_W3", TRIALS)
# W3- P2
histPSO(1, 3, "test/visuals/pso_hist_P2_W3", TRIALS)
# W3 - P3
histPSO(2, 3, "test/visuals/pso_hist_P3_W3", TRIALS)
# W3 - P1
histPSO(0, 4, "test/visuals/pso_hist_P1_W4", TRIALS)
# W4- P2
histPSO(1, 4, "test/visuals/pso_hist_P2_W4", TRIALS)
# W4 - P3
histPSO(2, 4, "test/visuals/pso_hist_P3_W4", TRIALS)
plt.clf()

