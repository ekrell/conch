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

- `metaplanner.py`: Metaheuristic planning on a raster map that incorporates work minimization and reward maximization. User's choice of metaheuristic algorithm; default is particle swarm optimization. 
- `gplanner_solve.py`: Graph-based planning with either Dijkstra or A* (input graph is a Python dictionary).

# Related publications

- [Autonomous Water Surface Vehicle Metaheuristic Mission Planning using Self-generated Goals and Environmental Forecasts](https://www.researchgate.net/publication/340066053_Autonomous_Water_Surface_Vehicle_Metaheuristic_Mission_Planning_using_Self-generated_Goals_and_Environmental_Forecasts)
    - Evan Krell, Scott A. King, Luis Rodolfo Garcia Carrillo
    - American Control Conference. March 2020
    - DOI: 10.13140/RG.2.2.30318.56640

# Quick start tutorials

### Metaheuristic path planning with `metaplanner.py`

### Dijkstra or A* on uniform grids or visibility graphs

### Dijkstra or A* on extended visibility graphs

# Repo organization

- planners: main executables (builders and planners)
- tools: minor utilities for data format conversions, etc
- test: example input files, scripts with experimental runs, and outputs
    - acc2020: scripts and outputs for ACC 2020 publication (DOI: 10.13140/RG.2.2.30318.56640)
    - gsen6331: scripts and outputs for TAMUCC course GSEN 6331

# Tool documentation






