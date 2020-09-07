# conch
Path planning for unmanned surface vehicles subject to environmental forces. 

# Notice

Because of several large files, `test/` is now in `.gitignore`, which includes input files and experiment runs. 
Soon I will host the data separately, and include a `wget` step to pull the data after cloning the repo. 
Maybe I will keep the input files in the repo and only store outputs separately, since most users won't need those

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

- `metaplanner.py`: Metaheuristic planning on a raster map that incorporates work minimization and reward maximization. User's choice of metaheuristic algorithm; default is particle swarm optimization. 
- `gplanner_solve.py`: Graph-based planning with either Dijkstra or A* (input graph is a Python dictionary).

# Related publications

- [Autonomous Water Surface Vehicle Metaheuristic Mission Planning using Self-generated Goals and Environmental Forecasts](https://www.researchgate.net/publication/340066053_Autonomous_Water_Surface_Vehicle_Metaheuristic_Mission_Planning_using_Self-generated_Goals_and_Environmental_Forecasts)
    - Evan Krell, Scott A. King, Luis Rodolfo Garcia Carrillo
    - American Control Conference. March 2020
    - DOI: 10.13140/RG.2.2.30318.56640

# Quick start tutorials

### Metaheuristic path planning with `metaplanner.py`

Particle swarm optimization will be used to generate a path.

### Dijkstra or A* on uniform grids or visibility graphs

### Dijkstra or A* on extended visibility graphs

An extended visibility graph (EVG) is a visibility graph with additional nodes added. A visibility graph is guaranteed to include the optimal shortest-distance path, but not necessarily the most energy-efficient. So, additional nodes offer more planning opportunities. 

        # Generate visibility graph
        python3 planners/vgplanner_build.py -r test/full.tif -g vg.graph -s full.shp -m poly.png -v vg.png -n 4 --build

        # Extend visibility graph
        python3 planners/vg2evg.py -r test/full.tif -v vg.graph -e evg.graph -m evg.png --xdiff 10 --ydiff 10 --radius 10 --threshold 2

        # Convert to python dictionary
        python3 planners/vg2g.py -r test/full.tif -v evg.graph -o evg-ex.pickle \
            --sy 42.32343 --sx -70.99428 --dy 42.33600 --dx -70.88737

        # Solve path
        python3 planners/gplanner_solve.py -r test/full.tif -g evg-ex.pickle --currents_mag test/waterMag.tif --currents_dir test/waterDir.tif \
            --sy 42.32343 --sx -70.99428 --dy 42.33600 --dx -70.88737 --speed 0.5 --solver a* \
            -m evg-exsolve.png -p  evg-exsolve.txt > evg-exsolve.out



# Repo organization

- planners: main executables (builders and planners)
- tools: minor utilities for data format conversions, etc
- test: example input files, scripts with experimental runs, and outputs
    - scripts: various scripts for setting up runs
    - acc2020: scripts and outputs for ACC 2020 publication (DOI: 10.13140/RG.2.2.30318.56640)
    - gsen6331: scripts and outputs for TAMUCC course GSEN 6331

# Tool documentation






