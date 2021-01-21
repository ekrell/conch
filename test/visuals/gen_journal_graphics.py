#!/usr/bin/python3

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
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from itertools import repeat

def grid2world(row, col, transform, nrow):
    lat = transform[4] * col + transform[5] * row + transform[3]
    lon = transform[1] * col + transform[2] * row + transform[0]
    return (lat, lon)

#############
# Constants #
#############

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
currentsRasterOut_W1_0 = "test/visuals/20170503_currents_0.png"
currentsRasterOut_W1_1 = "test/visuals/20170503_currents_1.png"
currentsRasterOut_W1_2 = "test/visuals/20170503_currents_2.png"
currentsRasterOut_W2_0 = "test/visuals/20170801_currents_0.png"
currentsRasterOut_W2_1 = "test/visuals/20170801_currents_1.png"
currentsRasterOut_W2_2 = "test/visuals/20170801_currents_2.png"
currentsRasterOut_W3_0 = "test/visuals/20191001_currents_0.png"
currentsRasterOut_W3_1 = "test/visuals/20191001_currents_1.png"
currentsRasterOut_W3_2 = "test/visuals/20191001_currents_2.png"
currentsRasterOut_W4_0 = "test/visuals/20200831_currents_0.png"
currentsRasterOut_W4_1 = "test/visuals/20200831_currents_1.png"
currentsRasterOut_W4_2 = "test/visuals/20200831_currents_2.png"

# Full region
lower_left = {"lat" : 42.2647, "lon" : -70.9996}
upper_right = {"lat" : 42.3789, "lon" : -70.87151}
regionRasterFile = "test/inputs/full.tif"
regionShapeFile = "test/outputs/visgraph_build/visgraph.shp"
# Output files
regionOut = "test/visuals/region.png"
polyOut = "test/visuals/poly.png"

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
# PSO results (VG init)
vgPSOpathResDir = "test/outputs/metaplanner_initPop/"
vgPSOpathResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.txt"
vgPSOstatResStr = "PSO_P{}_W{}_S0.5_G500_I100_N5__T{}.out"

def init():
    # Init basemap
    m = Basemap( \
                llcrnrlon = lower_left["lon"],
                llcrnrlat = lower_left["lat"],
                urcrnrlon = upper_right["lon"],
                urcrnrlat = upper_right["lat"],
                resolution = "i",
                epsg = "4269")
    m.drawmapboundary(fill_color = sea)
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
             scale = 20,
             alpha = 1, latlon = True, color = "black")
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

def graphtask(P, color):
    # Plot start location
    m.scatter([P[0]], [P[1]], marker = 'D', color = color, s = 60, zorder = 10, edgecolors= "black")
    # Plot goal location
    m.scatter([P[2]], [P[3]], marker = "X", color = color, s = 70, zorder = 10, edgecolors= "black")

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
        if words[0] == "Exit":
            break
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


def pltPSOconvergence(path, work, pathResDir, pathResStr, ax, trials = 1):
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(path, work, trial)
        dfPSO = statPSO(pathFile)
        ax.plot(dfPSO["gen"], dfPSO["gbest"])

# get colormap
ncolors = 256
color_array = plt.get_cmap('Purples')(range(ncolors))
# change alpha values
color_array[0:1, -1] = 0.0
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='reward',colors=color_array)
# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)
def pltReward(rFile):
    rewardGrid = np.flipud(np.loadtxt(rFile))
    m = init()
    region(m)
    img = m.imshow(rewardGrid * rewardGrid, interpolation='nearest', zorder = 100, alpha = 1, cmap = "reward")
    region(m, colors = cm_landonly, z = 1000)

# Plot reward
rewardFile = "test/inputs/reward.txt"
pltReward(rewardFile)
plt.savefig("test/visuals/reward.png")

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
plt.savefig("test/visuals/getNpaths_P1.png")
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
plt.savefig("test/visuals/getNpaths_P2.png")
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
plt.savefig("test/visuals/getNpaths_P3.png")
plt.clf()

# PSO convergence (random init)
# PSO convergence - path 1
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(0, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4815.3878, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=4400.0321, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=4325.1572, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(0, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=3507.0305, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=2097.3748, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=1675.0157, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(0, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=3196.3011, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=1701.4989, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=1151.26, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(0, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=5025.9262, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=4330.592, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=4151.432, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_rand_P1_conv.png")
plt.clf()
# PSO convergence - path 2
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(1, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4590.4769, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=3900.0763, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=3745.1477, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(2300, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(1, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=4327.5204, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=3082.7551, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=2773.9114, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(2300, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(1, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=4201.3305, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=2858.711, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=2636.9706, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(2300, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(1, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=4768.2483, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=3646.9265, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=3440.4205, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(2300, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_rand_P2_conv.png")
plt.clf()
# PSO convergence - path 3
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(2, 1, rPSOpathResDir, rPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=2447.6688, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=2058.698, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=1922.7959, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(1900, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(2, 2, rPSOpathResDir, rPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=2885.3459, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=2584.5256, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=2489.5882, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(1900, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(2, 3, rPSOpathResDir, rPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=2948.1269, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=2675.4716, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=2593.5196, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(1900, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(2, 4, rPSOpathResDir, rPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=2698, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=2687.1266, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=2109.6257, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(1900, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_rand_P3_conv.png")
plt.clf()

# Plot region
m = init()
region(m)
plt.title("Boston Harbor")
plt.savefig(regionOut)
plt.clf()

# Plot region shapefile
poly()
plt.title("Boston Harbor - polygons")
plt.savefig(polyOut)
plt.clf()

# Plot Dijkstra results (Astar results are the same)
def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid", alpha = 1.0):
    path = [grid2world(p[0], p[1], transform, nrows) for p in \
            np.loadtxt(pathfile, delimiter = ",").astype("int")]
    m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle, alpha = alpha)

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
    graphtask(P2, C2)
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
    custom_labels = ["T-1, 4-way", "T-1, 8-way", "T-1, 16-way", "T-1, VG",
                     "T-2, 4-way", "T-2, 8-way", "T-2, 16-way", "T-2, VG",
                     "T-3, 4-way", "T-3, 8-way", "T-3, 16-way", "T-3, VG",
    ]
    legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, prop={'size':6}, handlelength = 5)
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

# PSO convergence (VG init)
# PSO convergence - path 1
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(0, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4815.3878, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=4400.0321, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=4325.1572, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(1000, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(0, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=3507.0305, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=2097.3748, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=1675.0157, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(1000, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(0, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=3196.3011, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=1701.4989, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=1151.26, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(1000, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(0, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=5025.9262, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=4330.592, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=4151.432, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(1000, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_vg_P1_conv.png")
plt.clf()
# PSO convergence - path 2
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(1, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=4590.4769, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=3900.0763, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=3745.1477, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(2300, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(1, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=4327.5204, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=3082.7551, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=2773.9114, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(2300, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(1, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=4201.3305, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=2858.711, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=2636.9706, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(2300, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(1, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=4768.2483, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=3646.9265, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=3440.4205, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(2300, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_vg_P2_conv.png")
plt.clf()
# PSO convergence - path 3
fig, axs = plt.subplots(1, 4, figsize = (7, 7))
# Work 1
pltPSOconvergence(2, 1, vgPSOpathResDir, vgPSOstatResStr, ax = axs[0], trials = TRIALS)
l1 = axs[0].axhline(y=2447.6688, color='red', linestyle='--', label = "4-way")
l2 = axs[0].axhline(y=2058.698, color='green', linestyle='--', label = "8-way")
l3 = axs[0].axhline(y=1922.7959, color='blue', linestyle='--', label = "16-way")
axs[0].set_ylim(1900, 10000)
axs[0].set_title("water currents: 1", fontsize = 10)
# Work 2
pltPSOconvergence(2, 2, vgPSOpathResDir, vgPSOstatResStr, ax = axs[1], trials = TRIALS)
axs[1].axhline(y=2885.3459, color='red', linestyle='--', label = "4-way")
axs[1].axhline(y=2584.5256, color='green', linestyle='--', label = "8-way")
axs[1].axhline(y=2489.5882, color='blue', linestyle='--', label = "16-way")
axs[1].set_ylim(1900, 10000)
axs[1].set_title("water currents: 2", fontsize = 10)
axs[1].get_yaxis().set_visible(False)
# Work 3
pltPSOconvergence(2, 3, vgPSOpathResDir, vgPSOstatResStr, ax = axs[2], trials = TRIALS)
axs[2].axhline(y=2948.1269, color='red', linestyle='--', label = "4-way")
axs[2].axhline(y=2675.4716, color='green', linestyle='--', label = "8-way")
axs[2].axhline(y=2593.5196, color='blue', linestyle='--', label = "16-way")
axs[2].set_ylim(1900, 10000)
axs[2].set_title("water currents: 3", fontsize = 10)
axs[2].get_yaxis().set_visible(False)
# Work 4
pltPSOconvergence(2, 4, vgPSOpathResDir, vgPSOstatResStr, ax = axs[3], trials = TRIALS)
axs[3].axhline(y=2698, color='red', linestyle='--', label = "4-way")
axs[3].axhline(y=2687.1266, color='green', linestyle='--', label = "8-way")
axs[3].axhline(y=2109.6257, color='blue', linestyle='--', label = "16-way")
axs[3].set_ylim(1900, 10000)
axs[3].set_title("water currents: 4", fontsize = 10)
axs[3].get_yaxis().set_visible(False)
axs[0].legend(handles = [l1,l2,l3] , labels=['4-way', '8-way', '16-way'],loc='upper center', title = "Dijkstra cost",
                          bbox_to_anchor=(1, -0.04),fancybox=False, shadow=False, ncol=3)
plt.subplots_adjust(wspace = 0.1)
plt.savefig("test/visuals/pso_vg_P3_conv.png")
plt.clf()

# Plot region
m = init()
region(m)
plt.title("Boston Harbor")
plt.savefig(regionOut)
plt.clf()

# Plot region shapefile
poly()
plt.title("Boston Harbor - polygons")
plt.savefig(polyOut)
plt.clf()

# Plot Dijkstra results (Astar results are the same)
def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid", alpha = 1.0):
    path = [grid2world(p[0], p[1], transform, nrows) for p in \
            np.loadtxt(pathfile, delimiter = ",").astype("int")]
    m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle, alpha = alpha)

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
    graphtask(P2, C2)
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
    custom_labels = ["T-1, 4-way", "T-1, 8-way", "T-1, 16-way", "T-1, VG",
                     "T-2, 4-way", "T-2, 8-way", "T-2, 16-way", "T-2, VG",
                     "T-3, 4-way", "T-3, 8-way", "T-3, 16-way", "T-3, VG",
    ]
    legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, prop={'size':6}, handlelength = 5)
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

def pltPSO(work, pathResDir, pathResStr, trials = 1):
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    # Path 1
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(0, work, trial)
        pathplt(pathFile, transform, nrows, color = C1, linestyle = "solid", alpha = 0.75)
    graphtask(P1, C1)
    # Path 2
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(1, work, trial)
        pathplt(pathFile, transform, nrows, color = C2, linestyle = "solid", alpha = 0.75)
    graphtask(P2, C2)
    # Path 3
    for trial in np.array(range(trials)) + 1:
        pathFile = pathResDir + "/" + pathResStr.format(2, work, trial)
        pathplt(pathFile, transform, nrows, color = C3, linestyle = "solid", alpha = 0.75)
    graphtask(P3, C3)
    # Legend
    ax = plt.gca()
    custom_lines = [Line2D([0], [0], linestyle = "solid", color = C1, alpha = 1.0, lw = 1.5),
                    Line2D([0], [0], linestyle = "solid",  color = C2,  alpha = 1.0,  lw = 1.5),
                    Line2D([0], [0], linestyle = "solid",  color = C3,  alpha = 1.0,  lw = 1.5),
                    ]
    custom_labels = ["T-1 trials", "T-2 trials", "T-3 trials",
    ]
    legend = ax.legend(custom_lines, custom_labels, loc = "upper center", ncol = 3, prop={'size':6}, handlelength = 5)
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

# PSO, random, W0
m = init()
region(m)
pltPSO(0, rPSOpathResDir, rPSOpathResStr, trials = TRIALS)
plt.title("PSO results (water currents = None)")
plt.savefig("test/visuals/pso_rand_W0.png")
plt.clf()
# PSO, random, W1
m = init()
region(m)
pltPSO(1, rPSOpathResDir, rPSOpathResStr, trials = TRIALS)
plt.title("PSO results (water currents = 1)")
plt.savefig("test/visuals/pso_rand_W1.png")
plt.clf()
# PSO, random, W2
m = init()
region(m)
pltPSO(2, rPSOpathResDir, rPSOpathResStr, trials = TRIALS)
plt.title("PSO results (water currents = 2)")
plt.savefig("test/visuals/pso_rand_W2.png")
plt.clf()
# PSO, random, W3
m = init()
region(m)
pltPSO(3, rPSOpathResDir, rPSOpathResStr, trials = TRIALS)
plt.title("PSO results (water currents = 3)")
plt.savefig("test/visuals/pso_rand_W3.png")
plt.clf()
# PSO, random, W4
m = init()
region(m)
pltPSO(4, rPSOpathResDir, rPSOpathResStr, trials = TRIALS)
plt.title("PSO results (water currents = 4)")
plt.savefig("test/visuals/pso_rand_W4.png")
plt.clf()

# PSO, VG init, W0
m = init()
region(m)
pltPSO(0, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS)
plt.title("PSO results, VG-init (water currents = None)")
plt.savefig("test/visuals/pso_vg_W0.png")
plt.clf()
# PSO, VG init, W1
m = init()
region(m)
pltPSO(1, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS)
plt.title("PSO results, VG-init (water currents = 1)")
plt.savefig("test/visuals/pso_vg_W1.png")
plt.clf()
# PSO, VG init, W2
m = init()
region(m)
pltPSO(2, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS)
plt.title("PSO results, VG-init (water currents = 2)")
plt.savefig("test/visuals/pso_vg_W2.png")
plt.clf()
# PSO, VG init, W3
m = init()
region(m)
pltPSO(3, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS)
plt.title("PSO results, VG-init (water currents = 3)")
plt.savefig("test/visuals/pso_vg_W3.png")
plt.clf()
# PSO, VG init, W4
m = init()
region(m)
pltPSO(4, vgPSOpathResDir, vgPSOpathResStr, trials = TRIALS)
plt.title("PSO results, VG-init (water currents = 4)")
plt.savefig("test/visuals/pso_vg_W4.png")
plt.clf()


# Dijksta, graph, W0
m = init()
region(m)
pltdijkstra(0)
plt.title("Dijkstra results (water currents = None)")
plt.savefig("test/visuals/dijkstra_graph_W0.png")
plt.clf()
# Dijksta, graph,p W1
m = init()
region(m)
pltdijkstra(1)
plt.title("Dijkstra results (water currents = 1)")
plt.savefig("test/visuals/dijkstra_graph_W1.png")
plt.clf()
# Dijksta, graph, W2
m = init()
region(m)
pltdijkstra(2)
plt.title("Dijkstra results (water currents = 2)")
plt.savefig("test/visuals/dijkstra_graph_W2.png")
plt.clf()
# Dijksta, graph, W3
m = init()
region(m)
pltdijkstra(3)
plt.title("Dijkstra results (water currents = 3)")
plt.savefig("test/visuals/dijkstra_graph_W3.png")
plt.clf()
# Dijksta, graph, W4
m = init()
region(m)
pltdijkstra(4)
plt.title("Dijkstra results (water currents = 4)")
plt.savefig("test/visuals/dijkstra_graph_W4.png")
plt.clf()


# Plot visibility graphs
# P1
m = init()
region(m)
graphplt(visgraphFile_P1)
graphtask(P1, C1)
plt.title("Visibility graph (P1)")
plt.savefig(visgraphOut_P1)
plt.clf()
# P2
m = init()
region(m)
graphplt(visgraphFile_P2)
graphtask(P2, C2)
plt.title("Visibility graph (P2)")
plt.savefig(visgraphOut_P2)
plt.clf()
# P3
m = init()
region(m)
graphplt(visgraphFile_P3)
graphtask(P3, C3)
plt.title("Visibility graph (P3)")
plt.savefig(visgraphOut_P3)
plt.clf()

# Plot water currents
# W1 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1)
plt.title("Water Velocity (m/s)\nTime = +0h (W1)")
plt.savefig(currentsRasterOut_W1_0)
plt.clf()
# W1 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1, band = 2)
plt.title("Water Velocity (m/s)\nTime = +1h (W1)")
plt.savefig(currentsRasterOut_W1_1)
plt.clf()
# W1 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W1, currentsRasterFile_dir_W1, band = 3)
plt.title("Water Velocity (m/s)\nTime = +2h (W1)")
plt.savefig(currentsRasterOut_W1_2)
plt.clf()

# W2 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2)
plt.title("Water Velocity (m/s)\nTime = +0h (W2)")
plt.savefig(currentsRasterOut_W2_0)
plt.clf()
# W2 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2, band = 2)
plt.title("Water Velocity (m/s)\nTime = +1h (W2)")
plt.savefig(currentsRasterOut_W2_1)
plt.clf()
# W2 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W2, currentsRasterFile_dir_W2, band = 3)
plt.title("Water Velocity (m/s)\nTime = +2h (W2)")
plt.savefig(currentsRasterOut_W2_2)
plt.clf()

# W3 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3)
plt.title("Water Velocity (m/s)\nTime = +0h (W3)")
plt.savefig(currentsRasterOut_W3_0)
plt.clf()
# W3 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3, band = 2)
plt.title("Water Velocity (m/s)\nTime = +1h (W3)")
plt.savefig(currentsRasterOut_W3_1)
plt.clf()
# W3 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W3, currentsRasterFile_dir_W3, band = 3)
plt.title("Water Velocity (m/s)\nTime = +2h (W3)")
plt.savefig(currentsRasterOut_W3_2)
plt.clf()

# W4 - t0
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4)
plt.title("Water Velocity (m/s)\nTime = +0h (W4)")
plt.savefig(currentsRasterOut_W4_0)
plt.clf()
# W4 - t1
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4, band = 2)
plt.title("Water Velocity (m/s)\nTime = +1h (W4)")
plt.savefig(currentsRasterOut_W4_1)
plt.clf()
# W4 - t2
m = init()
region(m)
currents(currentsRasterFile_mag_W4, currentsRasterFile_dir_W4, band = 3)
plt.title("Water Velocity (m/s)\nTime = +2h (W4)")
plt.savefig(currentsRasterOut_W4_2)
plt.clf()










