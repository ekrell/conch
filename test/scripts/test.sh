#!/bin/bash
# Run this from 'conch' root

START_LAT=42.3249
START_LON=-70.9318
TARGET_LAT=42.2792
TARGET_LON=-70.9250
SPEED=100
NUM_WAYPOINTS=5
GENERATIONS=1000
CHUNKS=1

python planners/goto.py \
    --start_lat              $START_LAT  \
    --start_lon              $START_LON  \
    --target_lat             $TARGET_LAT \
    --target_lon             $TARGET_LON \
    --speed                  $SPEED      \
    --region_file            test/data/sample_region.tif         \
    --magnitude_force_file   test/data/sample_force_magnitude.tif \
    --direction_force_file   test/data/sample_force_direction.tif \
    --obstacle_flag          1 \
    --num_waypoints          $NUM_WAYPOINTS  \
    --generations            $GENERATIONS \
    --figure_out             test/data/test.png \
    --table_out              test/data/test.csv \
    --pickle_out             test/data/test.pickle""
