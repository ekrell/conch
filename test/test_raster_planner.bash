PREFIX="python3 planners/rasterplanner.py -r test/full.tif"
DIR="test/rasterplanner/"

# FP-1
FP1="--sy 42.32343 --sx -70.99428 --dy 42.33600 --dx -70.88737"
$PREFIX $FP1 --solver a* -n 4 -m $DIR""/map-astar-4-FP1.png --trace $DIR""/trace-astar-4-FP1.png --path $DIR""/path-astar-4-FP1.txt > $DIR""/out-astar-4-FP1.txt
$PREFIX $FP1 --solver a* -n 8 -m $DIR""/map-astar-8-FP1.png --trace $DIR""/trace-astar-8-FP1.png --path $DIR""/path-astar-8-FP1.txt > $DIR""/out-astar-8-FP1.txt
$PREFIX $FP1 --solver a* -n 16 -m $DIR""/map-astar-16-FP1.png --trace $DIR""/trace-astar-16-FP1.png --path $DIR""/path-astar-16-FP1.txt > $DIR""/out-astar-16-FP1.txt
$PREFIX $FP1 --solver dijkstra -n 4 -m $DIR""/map-dijkstra-4-FP1.png --trace $DIR""/trace-dijkstra-4-FP1.png --path $DIR""/path-dijkstra-4-FP1.txt > $DIR""/out-dijkstra-4-FP1.txt
$PREFIX $FP1 --solver dijkstra -n 8 -m $DIR""/map-dijkstra-8-FP1.png --trace $DIR""/trace-dijkstra-8-FP1.png --path $DIR""/path-dijkstra-8-FP1.txt > $DIR""/out-dijkstra-8-FP1.txt
$PREFIX $FP1 --solver dijkstra -n 16 -m $DIR""/map-dijkstra-16-FP1.png --trace $DIR""/trace-dijkstra-16-FP1.png --path $DIR""/path-dijkstra-16-FP1.txt > $DIR""/out-dijkstra-16-FP1.txt

# FP-2
FP2="--sy 42.33283 --sx -70.97322 --dy 42.27184 --dx -70.903406"
$PREFIX $FP2 --solver a* -n 4 -m $DIR""/map-astar-4-FP2.png --trace $DIR""/trace-astar-4-FP2.png --path $DIR""/path-astar-4-FP2.txt > $DIR""/out-astar-4-FP2.txt
$PREFIX $FP2 --solver a* -n 8 -m $DIR""/map-astar-8-FP2.png --trace $DIR""/trace-astar-8-FP2.png --path $DIR""/path-astar-8-FP2.txt > $DIR""/out-astar-8-FP2.txt
$PREFIX $FP2 --solver a* -n 16 -m $DIR""/map-astar-16-FP2.png --trace $DIR""/trace-astar-16-FP2.png --path $DIR""/path-astar-16-FP2.txt > $DIR""/out-astar-16-FP2.txt
$PREFIX $FP2 --solver dijkstra -n 4 -m $DIR""/map-dijkstra-4-FP2.png --trace $DIR""/trace-dijkstra-4-FP2.png --path $DIR""/path-dijkstra-4-FP2.txt > $DIR""/out-dijkstra-4-FP2.txt
$PREFIX $FP2 --solver dijkstra -n 8 -m $DIR""/map-dijkstra-8-FP2.png --trace $DIR""/trace-dijkstra-8-FP2.png --path $DIR""/path-dijkstra-8-FP2.txt > $DIR""/out-dijkstra-8-FP2.txt
$PREFIX $FP2 --solver dijkstra -n 16 -m $DIR""/map-dijkstra-16-FP2.png --trace $DIR""/trace-dijkstra-16-FP2.png --path $DIR""/path-dijkstra-16-FP2.txt > $DIR""/out-dijkstra-16-FP2.txt

