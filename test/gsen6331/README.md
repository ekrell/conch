## GSEN 6331 class project
## Spring 2020
## Experimental results

### Path planning in dynamic marine environments

- Marine environment has dynamic water currents (varies over space and time)
- Path planning minimizes energy consumption
- Compare Astar, Dijkstra using uniform grids, visibility graphs, and extended visibility graphs
- Extended visibility graphs add additional nodes/edges to a visibility graph for additional choices for energy efficiency

### Study region



### Data descriptions

- **Currents:** dynamic water currents over entire study region
	- Velocity magnitude (m/s): `waterMag.tif`
	- Velocity direction (m/s): `waterDir.tif`
	- Dimensions (pixels): 1000 x 1093
	- Bands:
		- Each represents a single time interval
		- 1 hour between intervals
		- 4 bands
	- Source: The Northeast Coastal Ocean Forecast System (http://fvcom.smast.umassd.edu/necofs/)

- **Full:** entire region 
	- Binary occupancy grid: `full.tif`
	- (1 = occupied/land, 0 = free/water)
	- Dimensions (pixels): 1000 x 1093
	- Source: NOAA World Vector Shorelines (https://shoreline.noaa.gov/data/datasheets/wvs.html)

- **Mini:** subregion of Full
	- Binary occupancy grid: `mini.tif`
	- (1 = occupied/land, 0 = free/water)
	- Dimensions (pixels): 439 x 511
	- Source: NOAA World Vector Shorelines (https://shoreline.noaa.gov/data/datasheets/wvs.html)

### Graph generation

The uniform graph, visibility graph, and extended visibility graph are made only once for each region

#### Uniform graphs

- Full
	- Command: `python3 planners/gplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -m test/gsen6331/full_uni.png`
	- Mean creation time (5 trials): 10.538 seconds
	- Number of nodes: 751111
	- Number of edges: 
	- Files:
		- Graph: `full_uni.pickle` 
		- Plot: `full_uni.png`

![alt text](full_uni.png "full uni plot")

- Mini
	- Command: `python3 planners/gplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -m test/gsen6331/mini_uni.png`
	- Mean creation time (5 trials): 2.688 seconds
	- Number of nodes: 201203
	- Number of edges: 
	- Files:
		- Graph: `mini_uni.pickle`
		- Plot: `mini_uni.png`

![alt text](mini_uni.png "mini uni plot")

#### Visibility graphs

- Full
	- Command: `python3 planners/vgplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg.graph -s test/gsen6331/full.shp -m test/gsen6331/full_poly.png -v test/gsen6331/full_vg.png -n 4 --build`
	- Mean creation time (1 thread, 5 trials): 20.009 seconds
	- Mean creation time (4 threads, 5 trials): 10.235 seconds
	- Number of nodes: 
	- Number of edges: 1037
	- Files:
		- Graph: `full_vg.graph` 
		- Plot: `full_vg.png`

![alt text](full_vg.png "full vg plot")


- Full
	- Command: `python3 planners/vgplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg.graph -s test/gsen6331/mini.shp -m test/gsen6331/mini_poly.png -v test/gsen6331/mini_vg.png -n 4 --build`
	- Mean creation time (1 thread, 5 trials): 1.371 seconds
	- Mean creation time (4 threads, 5 trials): 0.780 seconds
	- Number of nodes: 
	- Number of edges: 140
	- Files:
		- Graph: `mini_vg.graph` 
		- Plot: `mini_vg.png`

![alt text](mini_vg.png "mini vg plot")

#### Extended visibility graphs


### Path planning

- Label: **full-static-a**
	- Region: full
	- Currents? disabled
	- Mission:
		- Start: 
		- Goal: 

	


