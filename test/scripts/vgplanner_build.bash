#!/bin/bash

# Purpose: build all visibility graphs
#   that any other scripts may need to use

python3 planners/vgplanner_build.py \
    --region test/inputs/full.tif \
    --graph  test/outputs/visgraph_build/visgraph.pickle \
    --shape  test/outputs/visgraph_build/visgraph.shp \
    --map    test/outputs/visgraph_build/visgraph-poly.png \
    --vgmap  test/outputs/visgraph_build/visgraph.png \
    --build \
    > test/outputs/visgraph_build/visgraph.out
