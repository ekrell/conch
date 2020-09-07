#!/bin/bash

FUJIN_PATH_FILE=/home/ekrell/Documents/Work/repos/conch/waypoints.txt

python planners/goto.py \
    --cached \
    --rowcol \
    --path_file $FUJIN_PATH_FILE \
    --start_lat               42.3249 \
    --start_lon              -70.9318 \
    --target_lat              42.2792 \
    --target_lon             -70.9250 \
    --speed                  10       \
    --region_file            test/data/sample_region.tif         \
    --magnitude_force_file   test/data/cdc_magnitude_uncertain.tif \
    --direction_force_file   test/data/cdc_direction_uncertain.tif \
    --obstacle_flag          1 \
    --num_waypoints          5 \
    --generations            500

