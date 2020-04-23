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
	- Resolution: xxxxx
	- Source: NOAA World Vector Shorelines (https://shoreline.noaa.gov/data/datasheets/wvs.html)

- **Mini:** subregion of Full
	- Binary occupancy grid: `mini.tif`
	- (1 = occupied/land, 0 = free/water)
	- Dimensions (pixels): 439 x 511
	- Resolution: xxxx
	- Source: NOAA World Vector Shorelines (https://shoreline.noaa.gov/data/datasheets/wvs.html)

### Graph generation

The uniform graph, visibility graph, and extended visibility graph are made only once for each region

#### Uniform graphs

- Full
	- Mean creation time (5 trials): 10.538 seconds
	- Number of nodes: 751111
	- Number of edges: 
	- Files:
		- Graph: `full_uni.pickle` 
		- Plot: `full_uni.png`

	python3 planners/gplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -m test/gsen6331/full_uni.png

![alt text](full_uni.png "full uni plot")

- Mini
	- Mean creation time (5 trials): 2.688 seconds
	- Number of nodes: 201203
	- Number of edges: 
	- Files:
		- Graph: `mini_uni.pickle`
		- Plot: `mini_uni.png`

	python3 planners/gplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -m test/gsen6331/mini_uni.png

![alt text](mini_uni.png "mini uni plot")

#### Visibility graphs

- Full
	- Mean creation time (1 thread, 5 trials): 20.009 seconds
	- Mean creation time (4 threads, 5 trials): 10.235 seconds
	- Number of nodes: 
	- Number of edges: 1037
	- Files:
		- Graph: `full_vg.graph` 
		- Plot: `full_vg.png`

	python3 planners/vgplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg.graph -s test/gsen6331/full.shp -m test/gsen6331/full_poly.png -v test/gsen6331/full_vg.png -n 4 --build

![alt text](full_vg.png "full vg plot")


- Mini
	- Mean creation time (1 thread, 5 trials): 1.371 seconds
	- Mean creation time (4 threads, 5 trials): 0.780 seconds
	- Number of nodes: 
	- Number of edges: 140
	- Files:
		- Graph: `mini_vg.graph` 
		- Plot: `mini_vg.png`

	python3 planners/vgplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg.graph -s test/gsen6331/mini.shp -m test/gsen6331/mini_poly.png -v test/gsen6331/mini_vg.png -n 4 --build

![alt text](mini_vg.png "mini vg plot")

#### Extended visibility graphs

- Full (A, less added nodes)
	- Params: (ydiff: 5, xdiff: 5, radius:20, threshold: 2)
	- Base visibility graph: `full_vg.graph`
	- Mean _additional_ creation time:
	- Number of nodes:
	- Number of edges:
	- File:
		- Graph: full_evg-a.graph
		- Plot: full_evg-a.png

	python3 planners/vg2evg.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -e test/gsen6331/full_evg-a.graph -m test/gsen6331/full_evg-a.png

![alt text](full_evg-a.png "full evg a plot")

- Full (B, more added nodes)
	- Params: (ydiff: 2.5, xdiff: 2.5, radius:30, threshold: 2)
	- Base visibility graph: `full_vg.graph`
	- Mean _additional_ creation time:
	- Number of nodes:
	- Number of edges:
	- File:
		- Graph: full_evg-b.graph
		- Plot: full_evg-b.png

	python3 planners/vg2evg.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -e test/gsen6331/full_evg-a.graph -m test/gsen6331/full_evg-a.png

![alt text](full_evg-b.png "full evg b plot")

- Mini (A, less added nodes)
	- Params: (ydiff: 10, xdiff: 10, radius: 15, threshold: 2)
	- Base visibility graph: `mini_vg.graph`
	- Mean _additional_ creation time:
	- Number of nodes:
	- Number of edges:
	- File:
		- Graph: mini_evg-a.graph
		- Plot: mini_evg-a.png

	python3 planners/vg2evg.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -e test/gsen6331/mini_evg-a.graph -m test/gsen6331/mini_evg-a.png

![alt text](mini_evg-a.png "mini evg a plot")

- Mini (B, more added nodes)
	- Params: (ydiff: 7, xdiff: 7, radius:20, threshold: 2)
	- Base visibility graph: `mini_vg.graph`
	- Mean _additional_ creation time:
	- Number of nodes:
	- Number of edges:
	- File:
		- Graph: mini_evg-b.graph
		- Plot: mini_evg-b.png

	python3 planners/vg2evg.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -e test/gsen6331/mini_evg-a.graph -m test/gsen6331/mini_evg-a.png

![alt text](mini_evg-b.png "mini evg b plot")


### Path planning missions

- Mission: FP1
	- Region: Full
	- Start: (42.32343 N, -70.99428 E)
	- Goal:  (42.33600 N, -70.88737 E)
	- Boat speed: 0.5 meters/second

- Mission: FP2
	- Region: Full
	- Start: (42.33238 N, -70.97322 E)
	- Goal:  (42.27183 N, -70.90341 E)
	- Boat speed: 0.5 meters/second

- Mission: FP3
	- Region: Full
	- Start: (42.36221 N, -70.95617 E)
	- Goal:  (42.35282 N, -70.97952 E)
	- Boat speed: 0.5 meters/second

- Mission: MP1
	- Region: Mini
	- Start: (
	- Goal:  (
	- Boat speed: 0.5 meters/second

- Mission: MP2
	- Region: Mini
	- Start: (
	- Goal:  (
	- Boat speed: 0.5 meters/second

- Mission: MP3
	- Region: Mini
	- Start: (
	- Goal:  (
	- Boat speed: 0.5 meters/second


### Path planning experiments

- Experiment: FP1-AA
	- Setup:
		- Mission: FP1
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 19.5863 seconds
		- Distance: 9.5239 km
		- Duration: 31.7463 min
		- Cost (currents): N/A

	python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-AA.png

![alt text](FP1-AA.png "FP1-AA path plot")

- Experiment: FP1-AB
	- Setup:
		- Mission: FP1
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 21.6373 seconds
		- Distance: 9.5239 km
		- Duration: 31.7463 min
		- Cost (currents): N/A

	python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-AB.png

![alt text](FP1-AB.png "FP1-AB path plot")


- Experiment: FP1-AC
	- Setup:
		- Mission: FP1
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 205.7763 seconds
		- Distance: 13.1283 km
		- Duration: 43.7610 min
		- Cost (currents): 236.8804

	python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP1-AC.png "FP1-AC path plot")

- Experiment: FP1-AD
	- Setup:
		- Mission: FP1
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 201.2631 seconds
		- Distance: 13.1283 km
		- Duration: 43.7610 min
		- Cost (currents): 236.7048

	python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP1-AD.png "FP1-AD path plot")














