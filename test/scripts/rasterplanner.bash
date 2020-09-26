#!/bin/bash


runastar_static () {
    $PREFIX $DMEAS $P --solver a* $S -n 4 -m $DIR""/map-astar-4-$LABEL.png --trace $DIR""/trace-astar-4-$LABEL.png --path $DIR""/path-astar-4-$LABEL.txt > $DIR""/out-astar-4-$LABEL.txt
    $PREFIX $DMEAS $P --solver a* $S -n 8 -m $DIR""/map-astar-8-$LABEL.png --trace $DIR""/trace-astar-8-$LABEL.png --path $DIR""/path-astar-8-$LABEL.txt > $DIR""/out-astar-8-$LABEL.txt
    $PREFIX $DMEAS $P --solver a* $S -n 16 -m $DIR""/map-astar-16-$LABEL.png --trace $DIR""/trace-astar-16-$LABEL.png --path $DIR""/path-astar-16-$LABEL.txt > $DIR""/out-astar-16-$LABEL.txt
}

runall () {
    $PREFIX $DMEAS $P $WORK --solver dijkstra $S -n 4 -m $DIR""/map-dijkstra-4-$LABEL.png --trace $DIR""/trace-dijkstra-4-$LABEL.png --path $DIR""/path-dijkstra-4-$LABEL.txt > $DIR""/out-dijkstra-4-$LABEL.txt
    $PREFIX $DMEAS $P $WORK --solver dijkstra $S -n 8 -m $DIR""/map-dijkstra-8-$LABEL.png --trace $DIR""/trace-dijkstra-8-$LABEL.png --path $DIR""/path-dijkstra-8-$LABEL.txt > $DIR""/out-dijkstra-8-$LABEL.txt
    $PREFIX $DMEAS $P $WORK --solver dijkstra $S -n 16 -m $DIR""/map-dijkstra-16-$LABEL.png --trace $DIR""/trace-dijkstra-16-$LABEL.png --path $DIR""/path-dijkstra-16-$LABEL.txt > $DIR""/out-dijkstra-16-$LABEL.txt
}

SPEED=$1
PREFIX="/usr/bin/python3 planners/rasterplanner.py -r test/inputs/full.tif"
DIR="test/outputs/rasterplanner/haversine_s$SPEED"
S="--speed $SPEED"
DMEAS="--dist_measure haversine"
PATH_1="--sy 42.32343 --sx -70.99428 --dy 42.33600 --dx -70.88737"
PATH_2="--sy 42.33283 --sx -70.97322 --dy 42.27184 --dx -70.903406"
PATH_3="--sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952"
WORK_1="--currents_mag test/inputs/20170503_magwater.tiff --currents_dir test/inputs/20170503_dirwater.tiff"
WORK_2="--currents_mag test/inputs/20170801_magwater.tiff --currents_dir test/inputs/20170801_dirwater.tiff"
WORK_3="--currents_mag test/inputs/20191001_magwater.tiff --currents_dir test/inputs/20191001_dirwater.tiff"
WORK_4="--currents_mag test/inputs/20200831_magwater.tiff --currents_dir test/inputs/20200831_dirwater.tiff"


LABEL="P1W0"
P=$PATH_1
WORK=""
runastar_static
runall

LABEL="P1W1"
P=$PATH_1
WORK=$WORK_1
runall

LABEL="P1W2"
P=$PATH_1
WORK=$WORK_2
runall

LABEL="P1W3"
P=$PATH_1
WORK=$WORK_3
runall

LABEL="P1W4"
P=$PATH_1
WORK=$WORK_4
runall

LABEL="P2W0"
P=$PATH_2
WORK=""
runastar_static
runall

LABEL="P2W1"
P=$PATH_2
WORK=$WORK_1
runall

LABEL="P2W2"
P=$PATH_2
WORK=$WORK_2
runall

LABEL="P2W3"
P=$PATH_2
WORK=$WORK_3
runall

LABEL="P2W4"
P=$PATH_2
WORK=$WORK_4
runall

LABEL="P3W0"
P=$PATH_3
WORK=""
runastar_static
runall

LABEL="P3W1"
P=$PATH_3
WORK=$WORK_1
runall

LABEL="P3W2"
P=$PATH_3
WORK=$WORK_2
runall

LABEL="P3W3"
P=$PATH_3
WORK=$WORK_3
runall

LABEL="P3W4"
P=$PATH_3
WORK=$WORK_4
runall

