# conch
Path planning for unmanned surface vehicles subject to environmental forces. 

# Overview

This repo contains multiple tools for unmanned surface vehicle path planning. 
The vehicle is assumed to be acted on by weather forces such as wind and water currents. 
Planning algorithms herein use a work-based cost function to minimize the energy expenditure. 
Other planning objectives may be incorporated by individual tools. 

Some of the algorithms include a maximization of the path's reward. The reward refers to a grid of values across the search domain.
For example, the reward might come from a map of seagrass meadows if it is desirable for the robot's path to intersect it for data collection. 

Tools are classified as either _builders_ or _solvers_, both being part of the planning process. For example, the builder `vgplanner_build.py` can be used to convert a raster map to visibility graph. The, the solver `gplanner_solve.py` can run either Dijkstra or A* to solve a path on that graph. Other solvers, such as `metaplanner.py` take a raster directly and do not have an associated builder. 

## Builders

- `gplanner_build.py`: Converts a raster occupancy grid to fully-connected grid
- `vgplanner_build.py`: Converts a raster occupancy grid to visibility graph
- `vg2evg.py`: Creates an extended visibility graph from a given visibility graph
- `vg2g.py`: Converts a graph in [TaipanRex's pyvisgraph](https://github.com/TaipanRex/pyvisgraph) format to a simple Python dictionary

## Planners

- `gridplanner.py`: Uses Dijkstra or A* to plan on an ascii map. Water currents may be provided, but it is not geospatial-aware and will use Euclidean distance.
- `rasterplanner.py`: Uses Dijkstra or A* to plan on a GeoTiff. Planning incorporates work minimization.
- `metaplanner.py`: Metaheuristic planning on a raster map that incorporates work minimization and reward maximization. User's choice of metaheuristic algorithm; default is particle swarm optimization. 
- `gplanner_solve.py`: Graph-based planning with either Dijkstra or A* (input graph is a Python dictionary).

# Related publications

- [Autonomous Water Surface Vehicle Metaheuristic Mission Planning using Self-generated Goals and Environmental Forecasts](https://www.researchgate.net/publication/340066053_Autonomous_Water_Surface_Vehicle_Metaheuristic_Mission_Planning_using_Self-generated_Goals_and_Environmental_Forecasts)
    - Evan Krell, Scott A. King, Luis Rodolfo Garcia Carrillo
    - American Control Conference. March 2020
    - DOI: 10.13140/RG.2.2.30318.56640

# Quick start tutorials

### Installation

    # Clone the repo
    git clone https://github.com/ekrell/conch.git
    
    # Get dependencies
    sudo apt update
    sudo apt install gdal-bin
    sudo apt install python3-gdal
    pip3 install haversine numpy matplotlib dill pyvisgraph pandas pygmo rasterio shapely geopandas pyshp fiona

### Particle Swarm Optimization + Visibility Graphs

PSO is used to generate a solution for a path planning problem.
Given water current forecasts, PSO will optimize the path based on energy efficiency. 
The VG is used to quickly generate a set of initial candidate solution paths to use as the PSO initial population.

        # Setup variables
        INDIR=test/inputs/
        OUTDIR=test/outputs/

        # Generate Visibility Graph (VG)
        python3 planners/vgplanner_build.py \
            -r $INDIR/full.tif \                 # Input occupancy grid raster (GeoTiff)
            -g $OUTDIR/sample_vg.graph \         # Output VG
            -s $OUTDIR/sample_full.shp \         # Output geospatial shapefile of polygons
            -m $OUTDIR/sample_poly.png \         # Output figure of polygons
            -v $OUTDIR/sample_vg.png \           # Output figure of VG
            -n 4 \                               # Number of CPU workers
            --build                              # Actually build the graph (otherwise, just print info)

        # Add (start, goal) points to VG & convert to Python dictionary
        python3 planners/vg2g.py \
            -r $INDIR/full.tif \                 # Input occupancy grid raster (GeoTiff)
            -v $OUTDIR/sample_vg.graph \         # Input VG
            -o $OUTDIR/sample_vg.pickle \        # Output VG as pickled Python dictionary
            --sy 42.32343 \                      # Start y coord (latitude)
            --sx -70.99428 \                     # Start x coord (longitude)
            --dy 42.33600 \                      # Goal y coord (latitude)
            --dx -70.88737                       # Goal x coord (longitude)

        # Generate initial population using VG
        python3 planners/getNpaths.py \
            --region $INDIR/full.tif \           # Input occupancy grid raster (GeoTiff)
            --n_paths 100 \                      # Number of paths to generate
            --graph $OUTDIR/sample_vg.pickle \   # Input VG
            --paths $OUTDIR/sample_initpop.txt \ # Output paths for initial population
            --sy 42.32343 \                      # Start y coord (latitude)
            --sx -70.99428 \                     # Start x coord (longitude)
            --dy 42.33600 \                      # Goal y coord (latitude)
            --dx -70.88737                       # Goal x coord (longitude) 
            --map $OUTDIR/sample_initpop.png     # Output figure of initial population

        # Solve 
        python3 planners/metaplanner.py \
            -r $INDIR/full.tif \                     # Input occupancy grid raster (GeoTiff)
            --currents_mag $INDIR/20170503_magwater.tiff \  # Water currents magnitudes raster (GeoTiff)
            --currents_dir $INDIR/20170503_dirwater.tiff \  # Water currents directions raster (GeoTiff)
            --sy 42.32343 \                          # Start y coord (latitude)
            --sx -70.99428 \                         # Start x coord (longitude)
            --dy 42.33600 \                          # Goal y coord (latitude)
            --dx -70.88737 \                         # Goal x coord (longitude)
            --speed 0.5 \                            # Constant target boat speed
            --generations 500 \                      # PSO generations
            --pool_size 100 \                        # PSO pool size
            --num_waypoints 5 \                      # Number of waypoints in PSO solution
            --init_pop $OUTDIR/sample_initpop.txt \  # Input initial population paths
            --map $OUTDIR/sample_pso_path.png \      # Output figure of solution path
            --path $OUTDIR/sample_pso_path.txt \     # Output path waypoints
            > $OUTDIR/sample_pso_stats.out           # Output path information 


### Particle Swarm Optimization

PSO is used to generate a solution for a path planning problem.
Given water current forecasts, PSO will optimize the path based on energy efficiency. 
The PSO initial population is randomly generated. 

**Not expected to perform as well as when using VG (above).**

        # Setup variables
        INDIR=test/inputs/
        OUTDIR=test/outputs/

        # Solve 
        python3 planners/metaplanner.py \
            -r $INDIR/full.tif \                  # Input occupancy grid raster (GeoTiff)
            --currents_mag $INDIR/20170503_magwater.tiff \  # Water currents magnitudes raster (GeoTiff)
            --currents_dir $INDIR/20170503_dirwater.tiff \  # Water currents directions raster (GeoTiff)
            --sy 42.32343 \                       # Start y coord (latitude)
            --sx -70.99428 \                      # Start x coord (longitude)
            --dy 42.33600 \                       # Goal y coord (latitude)
            --dx -70.88737 \                      # Goal x coord (longitude)
            --speed 0.5 \                         # Constant target boat speed
            --generations 500 \                   # PSO generations
            --pool_size 100 \                     # PSO pool size
            --num_waypoints 5 \                   # Number of waypoints in PSO solution
            --map $OUTDIR/sample_pso_path_2.png \   # Output figure of solution path
            --path $OUTDIR/sample_pso_path_2.txt \  # Output path waypoints
            > $OUTDIR/sample_pso_stats_2_out.txt        # Output path information 


### Dijkstra on Uniform Graphs or Visibility Graphs

Dijkstra is used to generate a solution for a path planning problem.
Given water current forecasts, Dijkstra will optimize the path based on energy efficiency.

#### Using a Uniform Graph

In this example, using an 8-way neighborhood.

        # Setup variables
        INDIR=test/inputs/
        OUTDIR=test/outputs/

        # Generate Uniform Graph from raster
        python3 planners/gplanner_build.py \
            --region $INDIR/full.tif \             # Input occupancy grid raster (GeoTiff)
            --graph $OUTDIR/sample_uni_8.pickle \  # Output pickled graph
            --nhood_type 8                         # Number of neighbors (accepts: 4, 8, 16)

        # Solve
        python3 planners/graphplanner.py \
            --region $INDIR/full.tif \                           # Input occupancy grid raster (GeoTiff) 
            --graph $OUTDIR/sample_uni_8.pickle \                # Input 8-way neighborhood Uniform Graph
            --speed 0.5 \                                        # Constant target boat speed
            --sy 42.32343 \                                      # Start y coord (latitude)
            --sx -70.99428 \                                     # Start x coord (longitude)
            --dy 42.33600 \                                      # Goal y coord (latitude)
            --dx -70.88737 \                                     # Goal x coord (longitude)
            --solver dijkstra \                                  # Select solver (dijkstra, A*) 
            --map $OUTDIR/sample_uni_8_path.png \                # Output figure of solution path
            --trace  $OUTDIR/sample_dijkstra_uni_8_trace.png \   # Output figure of visited nodes
            --path  $OUTDIR/sample_dijkstra_uni_8_path.txt \     # Output path waypoints
            > $OUTDIR/sample_dijkstra_uni_8_stats_out.txt        # Output path information 


#### Using a Visibility Graph

        # Setup variables
        INDIR=test/inputs/
        OUTDIR=test/outputs/

        # Generate Visibility Graph (VG)
        python3 planners/vgplanner_build.py \
            -r $INDIR/full.tif \                 # Input occupancy grid raster (GeoTiff)
            -g $OUTDIR/sample_vg.graph \         # Output VG
            -s $OUTDIR/sample_full.shp \         # Output geospatial shapefile of polygons
            -m $OUTDIR/sample_poly.png \         # Output figure of polygons
            -v $OUTDIR/sample_vg.png \           # Output figure of VG
            -n 4 \                               # Number of CPU workers
            --build                              # Actually build the graph (otherwise, just print info)

        # Add (start, goal) points to VG & convert to Python dictionary
        python3 planners/vg2g.py \
            -r $INDIR/full.tif \                 # Input occupancy grid raster (GeoTiff)
            -v $OUTDIR/sample_vg.graph \         # Input VG
            -o $OUTDIR/sample_vg.pickle \        # Output VG as pickled Python dictionary
            --sy 42.32343 \                      # Start y coord (latitude)
            --sx -70.99428 \                     # Start x coord (longitude)
            --dy 42.33600 \                      # Goal y coord (latitude)
            --dx -70.88737                       # Goal x coord (longitude)

        # Solve
        python3 planners/graphplanner.py \
            --region $INDIR/full.tif \                      # Input occupancy grid raster (GeoTiff) 
            --currents_mag $INDIR/20170503_magwater.tiff \  # Water currents magnitudes raster (GeoTiff)
            --currents_dir $INDIR/20170503_dirwater.tiff \  # Water currents directions raster (GeoTiff)
            --graph $OUTDIR/sample_vg.pickle \              # Input VG
            --speed 0.5 \                                   # Constant target boat speed
            --sy 42.32343 \                                 # Start y coord (latitude)
            --sx -70.99428 \                                # Start x coord (longitude)
            --dy 42.33600 \                                 # Goal y coord (latitude)
            --dx -70.88737 \                                # Goal x coord (longitude)
            --solver dijkstra \                             # Select solver (dijkstra, A*) 
            --map $OUTDIR/sample_dijkstra_vg_path.png \     # Output figure of solution path
            > $OUTDIR/sample_dijkstra_vg_stats_out.txt      # Output path information 


# Repo organization

- planners: main executables (builders and planners)
- tools: minor utilities for data format conversions, etc
- test: example input files, scripts with experimental runs, and outputs
    - inputs: input data (i.e. GeoTiff rasters of the map and water current forecasts)
    - outputs: anything produced by the tools included in this repo
    - scripts: various scripts for setting up runs

# Tool documentation

**Detailed documentation on the parameters of each tool coming soon.**






