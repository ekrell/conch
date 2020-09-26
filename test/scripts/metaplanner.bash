#!/bin/bash

CHUNKS=$1

PREFIX="/usr/bin/python3 planners/metaplanner.py -r test/inputs/full.tif"
DIR="test/outputs/metaplanner/"

# Start-goal path scenarios
PATH_1="--sy 42.32343 --sx -70.99428 --dy 42.33600 --dx -70.88737"
PATH_2="--sy 42.33283 --sx -70.97322 --dy 42.27184 --dx -70.903406"
PATH_3="--sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952"

# Weather options
WORK_0="" # No currents
WORK_1="--currents_mag test/inputs/20170503_magwater.tiff --currents_dir test/inputs/20170503_dirwater.tiff"
WORK_2="--currents_mag test/inputs/20170801_magwater.tiff --currents_dir test/inputs/20170801_dirwater.tiff"
WORK_3="--currents_mag test/inputs/20191001_magwater.tiff --currents_dir test/inputs/20191001_dirwater.tiff"
WORK_4="--currents_mag test/inputs/20200831_magwater.tiff --currents_dir test/inputs/20200831_dirwater.tiff"

PSO_CMD="$PREFIX $PATH_{1} $WORK_{2} --speed {3} --generations {4} --pool_size {5} --num_waypoints {6} \
    --map $DIR/PSO_P{1}_W{2}_S{3}_G{4}_I{5}_N{6}__T{7}.png \
    --path $DIR/PSO_P{1}_W{2}_S{3}_G{4}_I{5}_N{6}__T{7}.txt \
    > $DIR/PSO_P{1}_W{2}_S{3}_G{4}_I{5}_N{6}__T{7}.out"


paths=("$PATH_1" "$PATH_2" "$PATH_3")
works=("$WORK_0" "$WORK_1" "$WORK_2" "$WORK_3" "$WORK_4")
PATHS="0 1 2"
WORKS="0 1 2 3 4"
SPEEDS="0.5 5"
GENS="500"
POOLS="50"
WAYPOINTS="5"
TRIALS="1 2 3"

rm $DIR/cmds.txt
for p in $PATHS; do
    for w in $WORKS; do
        for speed in $SPEEDS; do
            for gen in $GENS; do
                for pool in $POOLS; do
                    for way in $WAYPOINTS; do
                        for trial in $TRIALS; do
echo "$PREFIX ${paths[$p]} ${works[$w]} --speed $speed --generations $gen --pool_size $pool --num_waypoints $way \
    --map $DIR/PSO_P$p""_W$w""_S$speed""_G$gen""_I$pool""_N$way""__T$trial"".png \
    --path $DIR/PSO_P$p""_W$w""_S$speed""_G$gen""_I$pool""_N$way""__T$trial"".txt \
    > $DIR/PSO_P$p""_W$w""_S$speed""_G$gen""_I$pool""_N$way""__T$trial"".out" >> $DIR/cmds.txt
                        done
                    done
                done
            done
        done
    done
done

cat $DIR/cmds.txt | parallel -j 5
