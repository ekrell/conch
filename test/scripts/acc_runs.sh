#!/bin/bash
# Run this from 'conch' root.

START_LAT=42.3249
START_LON=-70.9318
TARGET_LAT=42.2792
TARGET_LON=-70.9250
SPEED=100
NUM_WAYPOINTS=10
GENERATIONS=500
CHUNKS=1

CMD_PREFIX="python planners/goto.py \
    --start_lat              $START_LAT  \
    --start_lon              $START_LON  \
    --target_lat             $TARGET_LAT \
    --target_lon             $TARGET_LON \
    --speed                  $SPEED      \
    --region_file            test/data/sample_region.tif         \
    --magnitude_force_file   test/data/cdc_magnitude_certain.tif \
    --direction_force_file   test/data/cdc_direction_certain.tif \
    --obstacle_flag          1 \
    --num_waypoints          $NUM_WAYPOINTS  \
    --generations            $GENERATIONS \
    --figure_out             test/data/alg-{1}-{2}-{3}-{4}.png \
    --table_out              test/data/alg-{1}-{2}-{3}-{4}.csv \
    --pickle_out             test/data/alg-{1}-{2}-{3}-{4}.pickle"

# Repetitions for each experiment, since stochastic
REPS=$(seq 0 1 10)

# Differential Evolution
CMD_DE="$CMD_PREFIX \
    --hyperparams de,{1},{2}"
CMD_DE="${CMD_DE//alg/de}"
LS_f=$(seq 0.0 0.1 1.0)
LS_cr=$(seq 0.0 0.1 1.0)
CMD_DE_PARA="parallel -j $CHUNKS $CMD_DE ::: $LS_f ::: $LS_cr ::: $REPS"

# Simple Genetic Algorithm
CMD_SGA="$CMD_PREFIX \
    --hyperparams sga,{1},{2}"
CMD_SGA="${CMD_SGA//alg/sga}"
LS_r=$(seq 0.0 0.1 1.0)
LS_m=$(seq 0.0 0.1 1.0)
CMD_SGA_PARA="parallel -j $CHUNKS $CMD_SGA ::: $LS_r ::: $LS_m ::: $REPS"

# Particle Swarm Optimization
CMD_PSO="$CMD_PREFIX \
    --hyperparams pso,{1},{2},{3}"
CMD_PSO="${CMD_PSO//alg/pso}"
LS_omega=$(seq 0.0 0.1 1.0)
LS_eta1=$(seq 0.0 0.4 4.0)
LS_eta2=$(seq 0.0 0.4 4.0)
CMD_PSO_PARA="parallel -j $CHUNKS $CMD_PSO ::: $LS_omega ::: $LS_eta1 \
                                           ::: $LS_eta2  ::: $REPS"

# Artificial Bee Colony
CMD_ABC="$CMD_PREFIX \
    --hyperparams abc,{1}"
CMD_ABC="${CMD_ABC//alg/abc}"
LS_tries=$(seq 0 5 50)
CMD_ABC_PARA="parallel -j $CHUNKS $CMD_ABC ::: $LS_tries ::: $REPS"


# Execute
$CMD_DE_PARA  &
$CMD_SGA_PARA &
$CMD_PSO_PARA &
$CMD_ABC_PARA &
wait






