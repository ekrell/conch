from osgeo import gdal, ogr, osr, gdalconst
from gdalconst import GA_ReadOnly
from shapely.geometry import shape
import shapefile
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
currentsRasterFile_mag = "test/gsen6331/waterMag.tif"
currentsRasterFile_dir = "test/gsen6331/waterDir.tif"

# Full region
lower_left = {"lat" : 42.2647, "lon" : -70.9996}
upper_right = {"lat" : 42.3789, "lon" : -70.87151}
regionRasterFile = "test/gsen6331/full.tif"
regionShapeFile = "test/gsen6331/full.shp"

unigraphFile = "test/gsen6331/full_uni.pickle"
visgraphFile = "test/gsen6331/full_vg_fp1.pickle"
evisgraphFile_A = "test/gsen6331/full_evg-a_fp1.pickle"
evisgraphFile_B = "test/gsen6331/full_evg-b_fp1.pickle"

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
    m.quiver(xx_water[points], yy_water[points], data_u[points], data_v[points], alpha = 0.6, latlon = True, color = "white")

def poly():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = init()
    m.readshapefile(regionShapeFile.replace(".shp", ""), "BH")
    patches = []
    for info, shape in zip(m.BH_info, m.BH):
        patches.append(Polygon(np.array(shape), True))
    ax.add_collection(PatchCollection(patches, facecolor = land, edgecolor = 'k', linewidths=1, zorder = 2))

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



def plotPathsStatic(sx, sy, dx, dy, fileA, fileB):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    pathplt(fileA, transform, nrows, color = c_pUNI, linestyle = l_pUNI)
    pathplt(fileB, transform, nrows, color = c_pVG, linestyle = l_pVG)
    custom_lines = [Line2D([0], [0], linestyle = l_pUNI, color = c_pUNI, alpha = 1.0, lw = 2),
                    Line2D([0], [0], linestyle = l_pVG,  color = c_pVG,  alpha = 1.0,  lw = 2)]
    m.scatter(sx, sy, color = "magenta", marker = "o", s = 40)
    plt.text(sx - 0.015, sy - 0.005, "start", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    m.scatter(dx, dy, color = "red", marker = "x", s = 40)
    plt.text(dx - 0.001, dy - 0.005, "end", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    legend = ax.legend(custom_lines, ["8-way uniform", "visibility graph"], loc = "lower left")
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

def plotPathsDynamic(sx, sy, dx, dy, fileA, fileB, fileC, fileD):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    pathplt(fileA, transform, nrows, color = c_pUNI, linestyle = l_pUNI)
    pathplt(fileB, transform, nrows, color = c_pVG, linestyle = l_pVG)
    pathplt(fileC, transform, nrows, color = c_pEVGA, linestyle = l_pEVGA)
    pathplt(fileD, transform, nrows, color = c_pEVGB, linestyle = l_pEVGB)
    custom_lines = [Line2D([0], [0], linestyle = l_pUNI, color = c_pUNI, alpha = 1.0, lw = 2),
                    Line2D([0], [0], linestyle = l_pVG,  color = c_pVG,  alpha = 1.0,  lw = 2),
                    Line2D([0], [0], linestyle = l_pEVGA,  color = c_pEVGA, alpha = 1.0,  lw = 2),
                    Line2D([0], [0], linestyle = l_pEVGB,  color = c_pEVGB, alpha = 1.0,  lw = 2)
                    ]
    m.scatter(sx, sy, color = "magenta", marker = "o", s = 40)
    plt.text(sx - 0.001, sy - 0.005, "start", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    m.scatter(dx, dy, color = "red", marker = "x", s = 40)
    plt.text(dx - 0.001, dy - 0.005, "end", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    legend = ax.legend(custom_lines, ["8-way uniform", "visibility graph", "extended VG A", "extended VG B"],
            loc = "lower right")
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

def plotPathsDynamicNoEVG(sx, sy, dx, dy, fileA, fileB):
    def pathplt(pathfile, transform, nrows, color = "white", linestyle = "solid"):
        path = [grid2world(p[0], p[1], transform, nrows) for p in \
                np.loadtxt(pathfile, delimiter = ",").astype("int")]
        m.plot(list(zip(*path))[1], list(zip(*path))[0], color = color, linestyle = linestyle)
    ds = gdal.Open(regionRasterFile, GA_ReadOnly)
    transform = ds.GetGeoTransform()
    nrows = ds.GetRasterBand(1).ReadAsArray().shape[0]
    pathplt(fileA, transform, nrows, color = c_pUNI, linestyle = l_pUNI)
    pathplt(fileB, transform, nrows, color = c_pVG, linestyle = l_pVG)
    custom_lines = [Line2D([0], [0], linestyle = l_pUNI, color = c_pUNI, alpha = 1.0, lw = 2),
                    Line2D([0], [0], linestyle = l_pVG,  color = c_pVG,  alpha = 1.0,  lw = 2),
                    ]
    m.scatter(sx, sy, color = "magenta", marker = "o", s = 40)
    plt.text(sx - 0.001, sy - 0.005, "start", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    m.scatter(dx, dy, color = "red", marker = "x", s = 40)
    plt.text(dx - 0.001, dy - 0.005, "end", fontweight = "bold", color = "salmon",
            bbox = dict(facecolor = "gray", alpha = 0.6, linewidth = 0.0))
    legend = ax.legend(custom_lines, ["8-way uniform", "visibility graph"],
            loc = "lower right")
    legend.get_frame().set_alpha(6.0)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_linewidth(0.0)

c_pUNI = "#6d2720"
l_pUNI = "solid"
c_pVG = "#406d20"
l_pVG = "dashed"
c_pEVGA = "#20666d"
l_pEVGA = "dotted"
c_pEVGB = "#4e206d"
l_pEVGB = (0, (3, 1, 1, 1))

# Graph stats
uni = pickle.load(open(unigraphFile, 'rb'))
vg = pickle.load(open(visgraphFile, 'rb'))
evg_a = pickle.load(open(evisgraphFile_A, 'rb'))
evg_b = pickle.load(open(evisgraphFile_B, 'rb'))

uni_edges = set()
for v in uni:
    for w in uni[v]:
        if (v, w) not in uni_edges and (w, v) not in uni_edges:
            uni_edges.add((v, w))
vg_edges = set()
for v in vg:
    for w in vg[v]:
        if (v, w) not in vg_edges and (w, v) not in vg_edges:
            vg_edges.add((v, w))
evg_a_edges = set()
for v in evg_a:
    for w in evg_a[v]:
        if (v, w) not in evg_a_edges and (w, v) not in evg_a_edges:
            evg_a_edges.add((v, w))
evg_b_edges = set()
for v in evg_b:
    for w in evg_b[v]:
        if (v, w) not in evg_b_edges and (w, v) not in evg_b_edges:
            evg_b_edges.add((v, w))

#print("UNI    nodes: {}, edges: {}".format(len(uni),    len(uni_edges)))
print("VG     nodes: {}, edges: {}".format(len(vg),     len(vg_edges)))
print("EVG-A  nodes: {}, edges: {}".format(len(evg_a),  len(evg_a_edges)))
print("EVG-B  nodes: {}, edges: {}".format(len(evg_b),  len(evg_b_edges)))



# Plot paths (FP-1 - no currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
plotPathsStatic(-70.99428, 42.32343, -70.88737, 42.33600,
        "test/gsen6331/FP1-AA.txt", "test/gsen6331/FP1-BA.txt")
plt.title("Solution paths\nno water currents")
plt.savefig("test/gsen6331/vis/paths_FP1_static.png")
plt.clf()

# Plot paths (FP-1 - currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamic(-70.99428, 42.32343, -70.88737, 42.33600,
        "test/gsen6331/FP1-AD.txt", "test/gsen6331/FP1-BD.txt",
        "test/gsen6331/FP1-CD.txt", "test/gsen6331/FP1-DD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP1_dynamic.png")
plt.clf()

# Plot paths (FP-1 - currents, no EVG)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamicNoEVG(-70.99428, 42.32343, -70.88737, 42.33600,
        "test/gsen6331/FP1-AD.txt", "test/gsen6331/FP1-BD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP1_dynamic_noevg.png")
plt.clf()

# Plot paths (FP-2 - no currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
plotPathsStatic(-70.97322, 42.33283, -70.903406, 42.27183,
        "test/gsen6331/FP2-AA.txt", "test/gsen6331/FP2-BA.txt")
plt.title("Solution paths\nno water currents")
plt.savefig("test/gsen6331/vis/paths_FP2_static.png")
plt.clf()

# Plot paths (FP-2 - currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamic(-70.97322, 42.33283, -70.903406, 42.27184,
        "test/gsen6331/FP2-AD.txt", "test/gsen6331/FP2-BD.txt",
        "test/gsen6331/FP2-CD.txt", "test/gsen6331/FP2-DD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP2_dynamic.png")
plt.clf()

# Plot paths (FP-2 - currents, no EVG)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamicNoEVG(-70.97322, 42.33283, -70.903406, 42.27184,
        "test/gsen6331/FP2-AD.txt", "test/gsen6331/FP2-BD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP2_dynamic_noevg.png")
plt.clf()


# Plot paths (FP-3 - no currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
plotPathsStatic(-70.874744, 42.300212, -70.9945635, 42.326108,
        "test/gsen6331/FP3-AA.txt", "test/gsen6331/FP3-BA.txt")
plt.title("Solution paths\nno water currents")
plt.savefig("test/gsen6331/vis/paths_FP3_static.png")
plt.clf()

# Plot paths (FP-3 - currents)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamic(-70.874744, 42.300212, -70.9945635, 42.326108,
        "test/gsen6331/FP3-AD.txt", "test/gsen6331/FP3-BD.txt",
        "test/gsen6331/FP3-CD.txt", "test/gsen6331/FP3-DD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP3_dynamic.png")
plt.clf()

# Plot paths (FP-3 - currents, no EVG)
fig = plt.figure()
ax = fig.add_subplot(111)
m = init()
region(m)
currents()
plotPathsDynamicNoEVG(-70.874744, 42.300212, -70.9945635, 42.326108,
        "test/gsen6331/FP3-AD.txt", "test/gsen6331/FP3-BD.txt")
plt.title("Solution paths\nwater currents (t = 0 shown)")
plt.savefig("test/gsen6331/vis/paths_FP3_dynamic_noevg.png")
plt.clf()

# Plot uniform graph
#m = init()
#region(m)
#graphplt(CAREFUL - WILL CRASH COMPUTER)
#plt.title("8-way uniform graph")
#plt.savefig("test/gsen6331/vis/full_uni.png")
#plt.clf()


# Plot visibility graph
m = init()
region(m)
graphplt(visgraphFile)
plt.title("Visibility graph")
plt.savefig("test/gsen6331/vis/full_vg.png")
plt.clf()

# Plot extended visibility graph A
m = init()
region(m)
graphplt(evisgraphFile_A)
plt.title("Extended visibility graph - A")
plt.savefig("test/gsen6331/vis/full_evg-A.png")
plt.clf()

# Plot extended visibility graph A
m = init()
region(m)
graphplt(evisgraphFile_B)
plt.title("Extended visibility graph - B")
plt.savefig("test/gsen6331/vis/full_evg-B.png")
plt.clf()

# Plot region raster
m = init()
region(m)
plt.title("Boston Harbor")
plt.savefig("test/gsen6331/vis/full.png")
plt.clf()

# Plot region shapefile
poly()
plt.title("Boston Harbor - polygons")
plt.savefig("test/gsen6331/vis/full_poly.png")
plt.clf()

# Plot currents t0
m = init()
region(m)
currents()
plt.title("Water Velocity (m/s)\nTime = 0 minutes      Model: NECOFS")
plt.savefig("test/gsen6331/vis/full_water_1.png")
plt.clf()

# Plot currents t1
m = init()
region(m)
currents(band = 2)
plt.title("Water Velocity (m/s)\nTime = +60 minutes    Model: NECOFS")
plt.savefig("test/gsen6331/vis/full_water_2.png")
plt.clf()


