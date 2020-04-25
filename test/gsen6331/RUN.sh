python3 planners/gplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -m test/gsen6331/full_uni.png

python3 planners/gplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -m test/gsen6331/mini_uni.png

python3 planners/vgplanner_build.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg.graph -s test/gsen6331/full.shp -m test/gsen6331/full_poly.png -v test/gsen6331/full_vg.png -n 4 --build

python3 planners/vgplanner_build.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg.graph -s test/gsen6331/mini.shp -m test/gsen6331/mini_poly.png -v test/gsen6331/mini_vg.png -n 4 --build

python3 planners/vg2evg.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -e test/gsen6331/full_evg-a.graph -m test/gsen6331/full_evg-a.png

python3 planners/vg2evg.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -e test/gsen6331/full_evg-a.graph -m test/gsen6331/full_evg-a.png

python3 planners/vg2evg.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -e test/gsen6331/mini_evg-a.graph -m test/gsen6331/mini_evg-a.png

python3 planners/vg2evg.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -e test/gsen6331/mini_evg-a.graph -m test/gsen6331/mini_evg-a.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-AA.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-AB.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-AA.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341 --solver a* --speed 0.5 -m test/gsen6331/FP2-AB.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341 --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.90341  --solver a* --speed 0.5 -m test/gsen6331/FP2-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-AA.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP3-AB.png

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_uni.pickle -u None --sy 42.36221 --sx -70.95617 --dy 42.35282 --dx -70.97952 --solver a* --speed 0.5 -m test/gsen6331/FP3-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-AA.png

python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96 --solver a* --speed 0.5 -m test/gsen6331/MP1-AB.png

python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96 --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-AC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_uni.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-AD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-BA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-BB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-BA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-BB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp3.pickle -u None --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-BA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp3.pickle -u None --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP3-BB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp3.pickle  --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_vg.graph -o test/gsen6331/full_vg_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_vg_fp3.pickle  --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP2-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -o test/gsen6331/mini_vg_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg_mp1.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-BA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -o test/gsen6331/mini_vg_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg_mp1.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-BB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -o test/gsen6331/mini_vg_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg_mp1.pickle  --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-BC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_vg.graph -o test/gsen6331/mini_vg_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_vg_mp1.pickle  --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-BD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-CA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-CB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver dijkstra --speed 0.5 -m test/gsen6331/FP1-CC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp1.pickle --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp1.pickle -u None --sy 42.32343  --sx -70.99428  --dy 42.33600  --dx -70.88737  --solver a* --speed 0.5 -m test/gsen6331/FP1-CD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-CA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-CB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver dijkstra --speed 0.5 -m test/gsen6331/FP2-CC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp2.pickle --sy 42.33283 --sx -70.97322  --dy 42.27183  --dx -70.903406
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp2.pickle -u None --sy 42.33283  --sx -70.97322  --dy 42.27183  --dx -70.903406  --solver a* --speed 0.5 -m test/gsen6331/FP2-CD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp3.pickle -u None --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-CA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp3.pickle -u None --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP3-CB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp3.pickle  --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver dijkstra --speed 0.5 -m test/gsen6331/FP3-CC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/full.tif -v test/gsen6331/full_evg-a.graph -o test/gsen6331/full_evg-a_fp3.pickle --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/full.tif -g test/gsen6331/full_evg-a_fp3.pickle  --sy 42.362209 --sx -70.956174  --dy 42.35282  --dx -70.97952  --solver a* --speed 0.5 -m test/gsen6331/FP2-CD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_evg-a.graph -o test/gsen6331/mini_evg-a_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_evg-a_mp1.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-CA.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_evg-a.graph -o test/gsen6331/mini_evg-a_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_evg-a_mp1.pickle -u None --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-CB.png


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_evg-a.graph -o test/gsen6331/mini_evg-a_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_evg-a_mp1.pickle  --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver dijkstra --speed 0.5 -m test/gsen6331/MP1-CC.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif


		# Convert visibility graph format to standard format
		python3 planners/vg2g.py -r test/gsen6331/mini.tif -v test/gsen6331/mini_evg-a.graph -o test/gsen6331/mini_evg-a_mp1.pickle --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96
		# Solve
		python3 planners/gplanner_solve.py -r test/gsen6331/mini.tif -g test/gsen6331/mini_evg-a_mp1.pickle  --sy 42.29 --sx -70.92  --dy 42.30  --dx -70.96  --solver a* --speed 0.5 -m test/gsen6331/MP1-CD.png --currents_mag test/gsen6331/waterMag.tif --currents_dir test/gsen6331/waterDir.tif

