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

### Experiments

- Label: **full-static-a**
	- Region: full
	- Currents? disabled
	- Mission:
		- Start: 
		- Goal: 
	


