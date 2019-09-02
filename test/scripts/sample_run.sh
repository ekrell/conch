#!/bin/bash
# A sample run of the 'goto' planner.
# Run this from 'conch' root.

python planners/goto.py \
    --start_lat              42.287 \
    --start_lon              -70.92 \
    --target_lat             42.300 \
    --target_lon             -70.97 \
    --speed                  100    \
    --region_file            test/data/sample_region.tif          \
    --magnitude_force_file   test/data/sample_force_magnitude.tif \
    --direction_force_file   test/data/sample_force_direction.tif \
    --obstacle_flag          1 \
    --num_waypoints          5 \
    --generations            100

