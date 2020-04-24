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
	- Start: (42.29000 N, -70.92000 E)
	- Goal:  (42.30000 N, -70.96000 E)
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



- Experiment: FP2-AA
	- Setup:
		- Mission: FP2
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 20.7169 seconds
		- Distance: 9.5884 km
		- Duration: 31.9612 min
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-AA.png

![alt text](FP2-AA.png "FP2-AA path plot")



- Experiment: FP2-AB
	- Setup:
		- Mission: FP2
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 22.9612 seconds
		- Distance: 9.5884 km
		- Duration: 31.9612 min
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341 --solver a* --speed 0.5 -m test/gsen6331/FP2-AB.png

![alt text](FP2-AB.png "FP2-AB path plot")



- Experiment: FP2-AC
	- Setup:
		- Mission: FP2
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 267.6161 seconds
		- Distance: 10.0603 km
		- Duration: 33.5343 min
		- Cost (currents): 281.1850

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341 --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP2-AC.png "FP2-AC path plot")



- Experiment: FP2-AD
	- Setup:
		- Mission: FP2
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 264.5657
		- Distance: 10.0603 km
		- Duration: 33.5343 min
		- Cost (currents): 281.0374

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341  --solver a* --speed 0.5 -m test/gsen6331/FP2-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP2-AD.png "FP2-AD path plot")



- Experiment: FP3-AA
	- Setup:
		- Mission: FP3
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 
		- Distance:
		- Duration: 
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-AA.png

![alt text](FP3-AA.png "FP3-AA path plot")



- Experiment: FP3-AB
	- Setup:
		- Mission: FP3
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 6.2303 seconds
		- Distance: 4.8091 km
		- Duration: 16.0303 min
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP3-AB.png

![alt text](FP3-AB.png "FP3-AB path plot")



- Experiment: FP3-AC
	- Setup:
		- Mission: FP3
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 100.8932 seconds
		- Distance: 4.8164 km
		- Duration: 16.0546 min
		- Cost (currents): 177.8319

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP3-AC.png "FP3-AC path plot")



- Experiment: FP3-AD
	- Setup:
		- Mission: FP3
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 101.9508 seconds
		- Distance: 4.8164 km
		- Duration: 16.0546 min
		- Cost (currents): 177.8213

		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver a* --speed 0.5 -m test/gsen6331/FP3-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](FP3-AD.png "FP3-AD path plot")



- Experiment: MP1-AA
	- Setup:
		- Mission: MP1
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 4.0744 seconds
		- Distance: 4.5610 km
		- Duration: 15.2033 min
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-AA.png

![alt text](MP1-AA.png "MP1-AA path plot")



- Experiment: MP1-AB
	- Setup:
		- Mission: MP1
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 4.3459 seconds
		- Distance: 4.5610 km
		- Duration: 15.2033 min
		- Cost (currents): N/A

		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96 --solver a* --speed 0.5 -m test/gsen6331/MP1-AB.png

![alt text](MP1-AB.png "MP1-AB path plot")



- Experiment: MP1-AC
	- Setup:
		- Mission: MP1
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 34.3144 seconds
		- Distance: 5.2350 km
		- Duration: 17.4499 min
		- Cost (currents): 113.7643

		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96 --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](MP1-AC.png "MP1-AC path plot")



- Experiment: MP1-AD
	- Setup:
		- Mission: MP1
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 40.9089 seconds
		- Distance: 5.2350 km
		- Duration: 17.4499 min
		- Cost (currents): 113.7755

		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

![alt text](MP1-AD.png "MP1-AD path plot")


		
- Experiment: FP1-BA
	- Setup:
		- Mission: FP1
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 0.011194 seconds
		- Distance: 11.3801 km
		- Duration: 37.9336 min
		- Cost (currents): N/A

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-BA.png


![alt text](FP1-BA.png "FP1-BA path plot")



- Experiment: FP1-BB
	- Setup:
		- Mission: FP1
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 0.014285 seconds
		- Distance: 11.3801 km
		- Duration: 37.9336 min
		- Cost (currents): N/A

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-BB.png



![alt text](FP1-BB.png "FP1-BB path plot")



- Experiment: FP1-BC
	- Setup:
		- Mission: FP1
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 1.743125 seconds
		- Distance: 13.0744 km
		- Duration: 43.5814 min
		- Cost (currents): 395.6258

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


![alt text](FP1-BC.png "FP1-BC path plot")



- Experiment: FP1-BD
	- Setup:
		- Mission: FP1
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 1.85576 seconds
		- Distance: 13.0744 km
		- Duration: 43.5814 min
		- Cost (currents): 395.3967

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


![alt text](FP1-BD.png "FP1-BD path plot")



- Experiment: FP2-BA
	- Setup:
		- Mission: FP2
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 0.0095270 seconds
		- Distance: 12.5727 km
		- Duration: 41.9089 min
		- Cost (currents): N/A

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-BA.png




![alt text](FP2-BA.png "FP2-BA path plot")



- Experiment: FP2-BB
	- Setup:
		- Mission: FP2
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time: 0.01472187 seconds
		- Distance: 12.5727 km
		- Duration: 41.9089 min
		- Cost (currents): N/A

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-BB.png


![alt text](FP2-BB.png "FP2-BB path plot")



- Experiment: FP2-BC
	- Setup:
		- Mission: FP2
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 2.454721 seconds
		- Distance: Distance: 12.7219 km
		- Duration: Duration: 42.4063 min
		- Cost (currents): 552.5717

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


![alt text](FP2-BC.png "FP2-BC path plot")



- Experiment: FP2-BD
	- Setup:
		- Mission: FP2
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time: 2.57866
		- Distance: 12.7219 km
		- Duration: 42.4063 min
		- Cost (currents): 552.5395

		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


![alt text](FP2-BD.png "FP2-BD path plot")



- Experiment: FP3-BA
	- Setup:
		- Mission: FP3
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents): N/A


![alt text](FP3-BA.png "FP3-BA path plot")



- Experiment: FP3-BB
	- Setup:
		- Mission: FP3
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents): N/A


![alt text](FP3-BB.png "FP3-BB path plot")



- Experiment: FP3-BC
	- Setup:
		- Mission: FP3
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents):


![alt text](FP3-BC.png "FP3-BC path plot")



- Experiment: FP3-BD
	- Setup:
		- Mission: FP3
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents):


![alt text](FP3-BD.png "FP3-BD path plot")



- Experiment: MP1-BA
	- Setup:
		- Mission: MP1
		- Solver: Dijkstra
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents): N/A


![alt text](MP1-BA.png "MP1-BA path plot")



- Experiment: MP1-BB
	- Setup:
		- Mission: MP1
		- Solver: Astar
		- Graph: uniform
		- Currents? disabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents): N/A


![alt text](MP1-BB.png "MP1-BB path plot")



- Experiment: MP1-BC
	- Setup:
		- Mission: MP1
		- Solver: Dijkstra (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents):


![alt text](MP1-BC.png "MP1-BC path plot")



- Experiment: MP1-BD
	- Setup:
		- Mission: MP1
		- Solver: Astar (current-aware)
		- Graph: uniform
		- Currents? enabled
	- Results:
		- Planning time:
		- Distance:
		- Duration:
		- Cost (currents):


![alt text](MP1-BD.png "MP1-BD path plot")






