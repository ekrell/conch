from osgeo import gdal, ogr, osr, gdalconst
from gdalconst import GA_ReadOnly
import numpy as np
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
import os, re
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image

def grid2world(row, col, transform, nrow):
    lat = transform[4] * col + transform[5] * row + transform[3]
    lon = transform[1] * col + transform[2] * row + transform[0]
    return (lat, lon)

#############
# Constants #
#############

# Fonts
rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Helvetica']})
#rc('text', usetex = True)

# Color maps
sea = (153.0/255.0, 188.0/255.0, 182.0/255.0)
land = (239.0/255.0, 209.0/255.0, 171.0/255.0)
c_vispnt = (56.0/255.0, 56.0/255.0, 56.0/255.0)
cm_landsea = LinearSegmentedColormap.from_list(
       "landsea", [sea, land], N = 2)

###########
# Options #
###########

# Water currents
currentsRasterFile_mag = "test/acc2020/waterMag.tif"
currentsRasterFile_dir = "test/acc2020/waterDir.tif"

# Full region
lower_left = {"lat" : 42.2647, "lon" : -70.9996}
upper_right = {"lat" : 42.3789, "lon" : -70.87151}
regionRasterFile = "test/acc2020/full.tif"

# Entities
entities = [
    {'label' : 'e1', 'color' : (200, 147, 199)},
    {'label' : 'e2', 'color' : (203, 100, 100)},
    {'label' : 'e3', 'color' : (229, 213, 140)}
]


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
    m.drawparallels(np.arange(-90.,120.,.05), labels = [1, 0, 0, 0], color = "gray")
    m.drawmeridians(np.arange(-180.,180.,.05), labels = [0, 0, 0, 1], color = "gray")

    return m

def region(m):
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
    m.pcolormesh(xx, yy, data, cmap = cm_landsea)

def rewardPlt(m, rfile):
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
    m.pcolormesh(xx, yy, data, cmap = cm_landsea)
    edata = np.loadtxt(rfile)
    edata[data != 0] = 0
    edata = np.ma.masked_array(edata, edata <= 0)
    m.pcolormesh(xx, yy, edata, cmap = "Purples", alpha = 0.75)


def entitiesPlt(efile, entities):
    m = init()
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
    m.pcolormesh(xx, yy, data, cmap = cm_landsea)
    eimg = np.asarray(Image.open(efile))[:, :, :3]
    emap = np.zeros((eimg.shape[0], eimg.shape[1]))
    for r in range(eimg.shape[0]):
        for c in range(eimg.shape[1]):
            for e in range(len(entities)):
                if eimg[r][c][0] == entities[e]["color"][0] and eimg[r][c][1] == entities[e]["color"][1] and eimg[r][c][2] ==  entities[e]["color"][2]:
                    emap[r][c] = e + 1
    eimg = eimg[:, :, 1]
    emap[data != 0] = 0
    emap = np.ma.masked_array(emap, emap <= 0)
    m.pcolormesh(xx, yy, emap, cmap = "RdYlBu")

def currents(band = 1):
    # Plot water currents
    ds_mag = gdal.Open(currentsRasterFile_mag, GA_ReadOnly)
    ds_dir = gdal.Open(currentsRasterFile_dir, GA_ReadOnly)
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
    m.quiver(xx_water[points], yy_water[points], data_u[points], data_v[points], alpha = 0.8, latlon = True, color = "white")

def plotPathsCompare(sx, sy, dx, dy, fileA, dA, wA, fileB, dB, wB, fileC, dC, wC, loc = "lower right"):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    pathplt(fileA, transform, nrows, color = c_pUNI, linestyle = l_pUNI)
    pathplt(fileB, transform, nrows, color = c_pVG, linestyle = l_pVG)
    pathplt(fileC, transform, nrows, color = c_pEVGB, linestyle = l_pEVGB)
    custom_lines = [Line2D([0], [0], linestyle = l_pUNI, color = c_pUNI, alpha = 1.0, lw = 2),
                    Line2D([0], [0], linestyle = l_pVG,  color = c_pVG,  alpha = 1.0,  lw = 2),
                    Line2D([0], [0], linestyle = l_pEVGB,  color = c_pEVGB,  alpha = 1.0,  lw = 2),
                    ]
    m.scatter(sx, sy, color = "magenta", marker = "o", s = 40)
    plt.text(sx - 0.001, sy - 0.005, "start", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    m.scatter(dx, dy, color = "red", marker = "x", s = 40)
    plt.text(dx - 0.001, dy - 0.005, "end", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    legend = ax.legend(custom_lines, ["A*-energy, dist = {}, work = {}".format(dA, wA),
                                      "PSO-best, dist = {}, work = {}".format(dB, wB),
                                      "Shortest path, dist = {}, work = {}".format(dC, wC)],
            loc = loc)
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)


def plotPathsMany(sx, sy, dx, dy, pathFiles):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    for p in pathFiles:
        pathplt(p, transform, nrows, color = (.5, .3, .1, .2), linestyle = l_pUNI)

def plotPathsReward(sx, sy, dx, dy, fileA, fileB, fileC, RWs = (0, 0, 0, 0, 0, 0), loc = "lower right"):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    pathplt(fileA, transform, nrows, color = c_pUNI, linestyle = l_pUNI)
    pathplt(fileB, transform, nrows, color = c_pVG, linestyle = l_pVG)
    pathplt(fileC, transform, nrows, color = c_pEVGB, linestyle = l_pEVGB)
    custom_lines = [Line2D([0], [0], linestyle = l_pUNI, color = c_pUNI, alpha = 1.0, lw = 2),
                    Line2D([0], [0], linestyle = l_pVG,  color = c_pVG,  alpha = 1.0,  lw = 2),
                    Line2D([0], [0], linestyle = l_pEVGB,  color = c_pEVGB,  alpha = 1.0,  lw = 2),
                    ]
    m.scatter(sx, sy, color = "magenta", marker = "o", s = 40)
    plt.text(sx - 0.001, sy - 0.005, "start", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    m.scatter(dx, dy, color = "red", marker = "x", s = 40)
    plt.text(dx - 0.001, dy - 0.005, "end", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    legend = ax.legend(custom_lines, ["PSO, reward = {r}, work = {w}".format(r = RWs[0], w = RWs[1]),
                                      "PSO, reward = {r}, work = {w}".format(r = RWs[2], w = RWs[3]),
                                      "Shortest, reward = {r}, work = {w}".format(r = RWs[4], w = RWs[5])],
            loc = loc)
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)


def convCurve(convergence, cmpval = None):
    fig, ax = plt.subplots()
    y = range(1, 250, 10)
    for c in convergence:
        ax.plot(y, c[:25])
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Fitness (work)")
    ax.set_xlabel("PSO generation")
    ax.set_title("Convergence of PSO")
    if cmpval is not None:
        ax.axhline(cmpval, color = 'red', lw = 2)


c_pUNI = "#6d2720"
l_pUNI = "solid"
c_pVG = "#406d20"
l_pVG = "dashed"
c_pEVGA = "#20666d"
l_pEVGA = "dotted"
c_pEVGB = "#4e206d"
l_pEVGB = (0, (3, 1, 1, 1))

def boxPlt(distances, cmpDist, durations, cmpDur, works, cmpWork, rewards = None):
    if rewards is None:
        fig, axs = plt.subplots(1, 3)
    else:
        fig, axs = plt.subplots(1, 4)
    axs[0].set_title("Distance (km)")
    axs[0].boxplot(distances)
    axs[0].axhline(cmpDist, color = 'red', lw = 2)
    axs[0].set_xticks([], [])
    axs[1].set_title("Duration (secs)")
    axs[1].boxplot(durations)
    axs[1].axhline(cmpDur, color = 'red', lw = 2)
    axs[1].set_xticks([], [])
    axs[2].set_title("Work")
    axs[2].boxplot(works)
    axs[2].axhline(cmpWork, color = 'red', lw = 2)
    axs[2].set_xticks([], [])
    if rewards is not None:
        axs[3].boxplot(rewards)
        axs[3].set_title("Reward")
        axs[3].set_xticks([], [])
    fig.tight_layout()

#-------------#
# Environment #
#-------------#

# Plot reward
m = init()
rewardPlt(m, "test/acc2020/reward.txt")
plt.title("Reward")
plt.savefig("test/acc2020/vis/reward.png")
plt.clf()

# Plot t1 entities
entitiesPlt("test/acc2020/e_t1.png", entities)
plt.title("Entities, update 1")
plt.savefig("test/acc2020/vis/e1.png")
plt.clf()

# Plot t2 entities
entitiesPlt("test/acc2020/e_t2.png", entities)
plt.title("Entities, update 2")
plt.savefig("test/acc2020/vis/e2.png")
plt.clf()

# Plot t3 entities
entitiesPlt("test/acc2020/e_t3.png", entities)
plt.title("Entities, update 3")
plt.savefig("test/acc2020/vis/e3.png")
plt.clf()

# Plot t4 entities
entitiesPlt("test/acc2020/e_t4.png", entities)
plt.title("Entities, update 4 (most recent)")
plt.savefig("test/acc2020/vis/e4.png")
plt.clf()

# Plot region raster
m = init()
region(m)
plt.title("Boston Harbor")
plt.savefig("test/acc2020/vis/full.png")
plt.clf()

# Plot currents t0
m = init()
region(m)
currents()
plt.title("Water Velocity (m/s)\nTime = 0 minutes      Model: NECOFS")
plt.savefig("test/acc2020/vis/full_water_1.png")
plt.clf()

# Plot currents t1
m = init()
region(m)
currents(band = 2)
plt.title("Water Velocity (m/s)\nTime = +60 minutes    Model: NECOFS")
plt.savefig("test/acc2020/vis/full_water_2.png")
plt.clf()


#----------------#
# FP-1: Reward 0 #
#----------------#

sy = 42.32343
sx = -70.99428
dy = 42.33600
dx = -70.88737

paths = [f for f in os.listdir('test/acc2020/') if re.match(r'FP1_PSO_EXP_-0.7.+.txt', f)]
paths = ['test/acc2020/' + p for p in paths]

distances = []
durations = []
works = []

convergence = []
for p in paths:
    with open(p.replace(".txt", ".out"), 'r') as f:
        data = f.readlines()
    distances.append(float([l for l in data if "Distance" in l][0].split()[1]))
    durations.append(float([l for l in data if "Duration" in l][0].split()[1]))
    works.append(float([l for l in data if "Cost" in l][0].split()[1]))

    on = False
    conv = []
    for l in data:
        if 'Exit condition' in l:
            on = False

        if on:
            c = l.split()[2]
            conv.append(float(c))

        if 'Fevals:' in l:
            on = True
    convergence.append(conv)

df = pd.DataFrame(
    {'distance' : distances,
    'duration' : durations,
    'work' : works,
     })

# Convergence curves of PSO runs
convCurve(convergence, 231.65)
plt.savefig("test/acc2020/vis/FP1_convergence.png")
plt.clf()


# Box plots of path distance, duration, reward
boxPlt(distances, 12.484, durations, 41.61, works, 231.65)
plt.savefig("test/acc2020/vis/FP1_box.png")
plt.clf()

# Plot work-min paths, comparing A* and PSO
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsCompare(sx, sy, dx, dy,
        "test/gsen6331/FP1-AD.txt", 12.48, 231.61,
        "test/acc2020/FP1_PSO_EXP_-0.6_4.0_3.0_4.txt", 9.74, 209.32,
        "test/gsen6331/FP1-BA.txt", 9.10, 392.14)
plt.title("Comparison of A* and PSO solutions")
plt.savefig("test/acc2020/vis/paths_FP1_compare.png")
plt.clf()

# Plot work-min paths, all PSO runs
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsMany(sx, sy, dx, dy, paths)
plt.title("PSO solution paths")
plt.savefig("test/acc2020/vis/paths_FP1_many.png")
plt.clf()

#------------------#
# FP-1: Reward 0.1 #
#------------------#

sy = 42.32343
sx = -70.99428
dy = 42.33600
dx = -70.88737

paths = [f for f in os.listdir('test/acc2020/') if re.match(r'FP1_PSO_RW_0.1.+.txt', f)]
paths = ['test/acc2020/' + p for p in paths]

distances = []
durations = []
works = []
rewards = []

convergence = []
for p in paths:
    with open(p.replace(".txt", ".out"), 'r') as f:
        data = f.readlines()
    distances.append(float([l for l in data if "Distance" in l][0].split()[1]))
    durations.append(float([l for l in data if "Duration" in l][0].split()[1]))
    works.append(float([l for l in data if "Cost" in l][0].split()[1]))
    rewards.append(float([l for l in data if "Reward" in l][0].split()[1]))

    on = False
    conv = []
    for l in data:
        if 'Exit condition' in l:
            on = False

        if on:
            c = l.split()[2]
            conv.append(float(c))

        if 'Fevals:' in l:
            on = True
    convergence.append(conv)

df = pd.DataFrame(
    {'distance' : distances,
     'duration' : durations,
     'work' : works,
     'reward' : rewards,
     })

# Convergence curves of PSO runs
convCurve(convergence)
plt.savefig("test/acc2020/vis/FP1_RW0.1_convergence.png")
plt.clf()

# Box plots of path distance, duration, reward
boxPlt(distances, 12.484, durations, 41.61, works, 231.65, rewards)
plt.savefig("test/acc2020/vis/FP1_RW0.1__box.png")
plt.clf()

# Plot work-min paths, comparing A* and PSO
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
rewardPlt(m, "test/acc2020/reward.txt")
currents()
plotPathsReward(sx, sy, dx, dy,
        "test/acc2020/FP1_PSO_RW_0.1-5.txt",
        "test/acc2020/FP1_PSO_RW_0.1-4.txt",
        "test/gsen6331/FP1-BA.txt",
        (6.12, 211.05, 5.44, 206.50, 4.96, 392.14))
plt.title("Best PSO paths when reward weight is 0.1x")
plt.savefig("test/acc2020/vis/paths_FP1_RW0.1.png")
plt.clf()

#------------------#
# FP-1: Reward 10  #
#------------------#

sy = 42.32343
sx = -70.99428
dy = 42.33600
dx = -70.88737

paths = [f for f in os.listdir('test/acc2020/') if re.match(r'FP1_PSO_RW_10.+.txt', f)]
paths = ['test/acc2020/' + p for p in paths]

distances = []
durations = []
works = []
rewards = []

convergence = []
for p in paths:
    with open(p.replace(".txt", ".out"), 'r') as f:
        data = f.readlines()
    distances.append(float([l for l in data if "Distance" in l][0].split()[1]))
    durations.append(float([l for l in data if "Duration" in l][0].split()[1]))
    works.append(float([l for l in data if "Cost" in l][0].split()[1]))
    rewards.append(float([l for l in data if "Reward" in l][0].split()[1]))

    on = False
    conv = []
    for l in data:
        if 'Exit condition' in l:
            on = False

        if on:
            c = l.split()[2]
            conv.append(float(c))

        if 'Fevals:' in l:
            on = True
    convergence.append(conv)

df = pd.DataFrame(
    {'distance' : distances,
     'duration' : durations,
     'work' : works,
     'reward' : rewards,
     })

# Convergence curves of PSO runs
convCurve(convergence)
plt.savefig("test/acc2020/vis/FP1_RW10_convergence.png")
plt.clf()

# Box plots of path distance, duration, reward
boxPlt(distances, 12.484, durations, 41.61, works, 231.65, rewards)
plt.savefig("test/acc2020/vis/FP1_RW10_box.png")
plt.clf()

# Plot work-min paths, comparing A* and PSO
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
rewardPlt(m, "test/acc2020/reward.txt")
currents()
plotPathsReward(sx, sy, dx, dy,
        "test/acc2020/FP1_PSO_RW_15-3.txt",
        "test/acc2020/FP1_PSO_RW_15-0.txt",
        "test/gsen6331/FP1-BA.txt",
        (10.80, 288.31, 7.09, 216.81, 4.96, 392.14))
plt.title("Best PSO paths when reward weight is 10x")
plt.savefig("test/acc2020/vis/paths_FP1_RW10.png")
plt.clf()


#------#
# FP-2 #
#------#

sy = 42.33283
sx = -70.97322
dy = 42.27184
dx = -70.903406
paths = [f for f in os.listdir('test/acc2020/') if re.match(r'FP1_PSO_FP2.+.txt', f)]
paths = ['test/acc2020/' + p for p in paths]

distances = []
durations = []
works = []

convergence = []
for p in paths:
    with open(p.replace(".txt", ".out"), 'r') as f:
        data = f.readlines()
    distances.append(float([l for l in data if "Distance" in l][0].split()[1]))
    durations.append(float([l for l in data if "Duration" in l][0].split()[1]))
    works.append(float([l for l in data if "Cost" in l][0].split()[1]))

    on = False
    conv = []
    for l in data:
        if 'Exit condition' in l:
            on = False

        if on:
            c = l.split()[2]
            conv.append(float(c))

        if 'Fevals:' in l:
            on = True
    convergence.append(conv)

df = pd.DataFrame(
    {'distance' : distances,
    'duration' : durations,
    'work' : works,
     })

# Convergence curves of PSO runs
convCurve(convergence, 298.57)
plt.savefig("test/acc2020/vis/FP2_convergence.png")
plt.clf()

# Box plots of path distance, duration, reward
boxPlt(distances, 10.825, durations, 36.08, works, 298.57)
plt.savefig("test/acc2020/vis/FP2_box.png")
plt.clf()

# Plot work-min paths, comparing A* and PSO
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsCompare(sx, sy, dx, dy,
        "test/gsen6331/FP2-AD.txt", 10.83, 298.57,
        "test/acc2020/FP1_PSO_FP2_50-17.txt", 9.62, 254.65,
        "test/gsen6331/FP2-BA.txt", 9.34, 267.56,
        loc = "upper right")
plt.title("Comparison of A* and PSO solutions")
plt.savefig("test/acc2020/vis/paths_FP2_compare.png")
plt.clf()

# Plot work-min paths, all PSO runs
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsMany(sx, sy, dx, dy, paths)
plt.title("PSO solution paths")
plt.savefig("test/acc2020/vis/paths_FP2_many.png")
plt.clf()


#------#
# FP-3 #
#------#

sy = -70.874744
sx = 42.300212
dy = -70.9945635
dx = 42.326108
paths = [f for f in os.listdir('test/acc2020/') if re.match(r'FP3_PSO_EXP.+.txt', f)]
paths = ['test/acc2020/' + p for p in paths]

distances = []
durations = []
works = []

convergence = []
for p in paths:
    with open(p.replace(".txt", ".out"), 'r') as f:
        data = f.readlines()
    distances.append(float([l for l in data if "Distance" in l][0].split()[1]))
    durations.append(float([l for l in data if "Duration" in l][0].split()[1]))
    works.append(float([l for l in data if "Cost" in l][0].split()[1]))

    on = False
    conv = []
    for l in data:
        if 'Exit condition' in l:
            on = False
        if on:
            c = l.split()[2]
            conv.append(float(c))
        if 'Fevals:' in l:
            on = True
    convergence.append(conv)

df = pd.DataFrame(
    {'distance' : distances,
    'duration' : durations,
    'work' : works,
     })

# Convergence curves of PSO runs
convCurve(convergence, 481.17)
plt.savefig("test/acc2020/vis/FP3_convergence.png")
plt.clf()


# Box plots of path distance, duration, reward
boxPlt(distances, 15.825, durations, 51.11, works, 481.17)
plt.savefig("test/acc2020/vis/FP3_box.png")
plt.clf()

# Plot work-min paths, comparing A* and PSO
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsCompare(sx, sy, dx, dy,
        "test/gsen6331/FP3-AD.txt", 15.33, 481.17,
        "test/acc2020/FP3_PSO_EXP_50-17.txt", 11.95, 600.23,
        "test/gsen6331/FP3-BA.txt", 11.00, 530.41)
plt.title("Comparison of A* and PSO solutions")
plt.savefig("test/acc2020/vis/paths_FP3_compare.png")
plt.clf()

# Plot work-min paths, all PSO runs
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsMany(sx, sy, dx, dy, paths)
plt.title("PSO solution paths")
plt.savefig("test/acc2020/vis/paths_FP3_many.png")
plt.clf()
