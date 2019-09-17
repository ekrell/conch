#!/bin/bash
# A sample run of the 'goto' planner.
# Run this from 'conch' root.

python planners/goto.py \
    --start_lat               42.3249 \
    --start_lon              -70.9318 \
    --target_lat              42.2792 \
    --target_lon             -70.9250 \
    --speed                  10       \
    --region_file            test/data/sample_region.tif         \
    --magnitude_force_file   test/data/cdc_magnitude_certain.tif \
    --direction_force_file   test/data/cdc_direction_certain.tif \
    --obstacle_flag          1 \
    --num_waypoints          5 \
    --generations            500 \
    --figure_out             test/data/sample_goto.png

