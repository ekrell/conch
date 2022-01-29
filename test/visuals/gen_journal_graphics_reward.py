#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import subprocess

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 7})

# Files
fileTemplate = "PSO_P{}_W{}_S0.5_G500_I100_N5_R{}__T{}.out"

# Experiments
paths = [0, 1, 2]
weathers = [1, 2, 3, 4]
rewardWeights = np.array([500, 1000, 2000, ]) #5000])
rewardNames = [" 1", "1.5", " 2", "3.5"]
trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dataDir1 = "test/outputs/metaplanner_reward"
dataDir2 = "test/outputs/metaplanner_initPop_reward"
outfile = "test/visuals/rewardWeight_combo.pdf"
outfileLegend = "test/visuals/rewardWeightLegend_combo.pdf"

dpi = 300

# Get all rewards, costs from planning output files
rewardData1 = np.zeros((len(paths), len(weathers), len(rewardWeights), len(trials)))
costData1   = np.zeros((len(paths), len(weathers), len(rewardWeights), len(trials)))

rewardData2 = np.zeros((len(paths), len(weathers), len(rewardWeights), len(trials)))
costData2   = np.zeros((len(paths), len(weathers), len(rewardWeights), len(trials)))


for p in range(len(paths)):
    path = paths[p]
    for w in range(len(weathers)):
        weather = weathers[w]
        for r in range(len(rewardWeights)):
            reward = rewardWeights[r]
            for t in range(len(trials)):
                trial = t + 1

                f1 = fileTemplate.format(path, weather, reward, trial)
                f1 = dataDir1 + "/" + f1
                rewardValue1 = float(subprocess.run(["grep", "Reward", f1], stdout=subprocess.PIPE).stdout.decode('utf-8').replace(" ", "").split(":")[1])
                costValue1   = float(subprocess.run(["grep", "Cost",   f1], stdout=subprocess.PIPE).stdout.decode('utf-8').replace(" ", "").split(":")[1])
                rewardData1[p, w, r, t] = rewardValue1
                costData1[p, w, r, t]   = costValue1

                f2 = fileTemplate.format(path, weather, reward, trial)
                f2 = dataDir2 + "/" + f2
                rewardValue2 = float(subprocess.run(["grep", "Reward", f2], stdout=subprocess.PIPE).stdout.decode('utf-8').replace(" ", "").split(":")[1])
                costValue2   = float(subprocess.run(["grep", "Cost",   f2], stdout=subprocess.PIPE).stdout.decode('utf-8').replace(" ", "").split(":")[1])
                rewardData2[p, w, r, t] = rewardValue2
                costData2[p, w, r, t]   = costValue2


cScale = np.array((rewardWeights - np.min(rewardWeights)) / (np.max(rewardWeights) - np.min(rewardWeights)))
colorScale = minmax_scale(cScale, feature_range=(0.5, 0.9))
sizeScale = np.array((rewardWeights - np.min(rewardWeights)) / (np.max(rewardWeights) - np.min(rewardWeights)))
sizeScale = minmax_scale(sizeScale, feature_range=(10, 40))
sizeScale = np.repeat(sizeScale, 10)
hueScale = minmax_scale(cScale, feature_range=(150, 225))
reds = [np.array((hueScale[c]/255., 100./255., 100./255., 0.5)) for c in range(len(colorScale)) for i in range(len(trials))]
greens = [np.array((100./255., hueScale[c]/255., 100./255., 0.5)) for c in range(len(colorScale)) for i in range(len(trials))]
blues = [np.array((100./255., 100./255., hueScale[c]/255., 0.5)) for c in range(len(colorScale)) for i in range(len(trials))]

colors = np.repeat(np.array(range(len(rewardWeights))), 10)

# Plot
fig, ax = plt.subplots(1, 4, figsize=(6,4))

for w in range(len(weathers)):
    ax[w].scatter(costData1[0,w,:,:].flatten(), rewardData1[0,w,:,:].flatten(), c=reds, marker="o", s=sizeScale, label="$\mathcal{T}_1$, ${PSO}_{R}$", edgecolor="k")
    ax[w].scatter(costData1[1,w,:,:].flatten(), rewardData1[1,w,:,:].flatten(), c=greens, marker="o", s=sizeScale, label="$\mathcal{T}_2$, ${PSO}_{R}$", edgecolor="k")
    ax[w].scatter(costData1[2,w,:,:].flatten(), rewardData1[2,w,:,:].flatten(), c=blues, marker="o", s=sizeScale, label="$\mathcal{T}_3$, ${PSO}_{R}$", edgecolor="k")

    ax[w].scatter(costData2[0,w,:,:].flatten(), rewardData2[0,w,:,:].flatten(), c=reds, marker="D", s=sizeScale, label="$\mathcal{T}_1$, ${PSO}_{VG}$", edgecolor="k")
    ax[w].scatter(costData2[1,w,:,:].flatten(), rewardData2[1,w,:,:].flatten(), c=greens, marker="D", s=sizeScale, label="$\mathcal{T}_2$, ${PSO}_{VG}$", edgecolor="k")
    ax[w].scatter(costData2[2,w,:,:].flatten(), rewardData2[2,w,:,:].flatten(), c=blues, marker="D", s=sizeScale, label="$\mathcal{T}_3$, ${PSO}_{VG}$", edgecolor="k")

    ax[w].set_title("Weather $\mathcal{{W}}_{}$".format(w+1))
    #ax[w].set_ylim(0, 4.25)
    ax[w].set_xlabel("Solution cost")
    ax[w].set_ylabel("Solution reward")

    if w > 0:
        ax[w].set_ylabel(None)

plt.tight_layout()
plt.savefig(outfile, dpi=dpi)

# Custom legend
cScale = np.array((rewardWeights - np.min(rewardWeights)) / (np.max(rewardWeights) - np.min(rewardWeights)))
colorScale = minmax_scale(cScale, feature_range=(0.5, 0.9))
sizeScale = np.array((rewardWeights - np.min(rewardWeights)) / (np.max(rewardWeights) - np.min(rewardWeights)))
sizeScale = minmax_scale(sizeScale, feature_range=(10, 80))
sizeScale = sizeScale * 1.5
hueScale = minmax_scale(cScale, feature_range=(150, 225))
reds = [np.array((hueScale[c]/255., 100./255., 100./255., 0.5)) for c in range(len(colorScale))]
greens = [np.array((100./255., hueScale[c]/255., 100./255., 0.5)) for c in range(len(colorScale))]
blues = [np.array((100./255., 100./255., hueScale[c]/255., 0.5)) for c in range(len(colorScale))]

fig, ax = plt.subplots(1, 2, figsize=(4, 1.2))
# PSO_R
x = [0, 0.5, 1, ]# 1.5]
y = [0, 0, 0, ]# 0]
ax[0].scatter(x, y, s=sizeScale, c=reds, marker="o", edgecolor="k")
y = [0.0025, 0.0025, 0.0025]#, 0.0025]
ax[0].scatter(x, y, s=sizeScale, c=greens, marker="o", edgecolor="k")
y = [0.005, 0.005, 0.005]#, 0.005]
ax[0].scatter(x, y, s=sizeScale, c=blues, marker="o", edgecolor="k")
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_xlim(-0.1, 1.1)
ax[0].set_ylim(-0.001, 0.006)
xset = -0.2
yset = -0.0008
ax[0].text(xset, yset + 0, "$\mathcal{T}_1$")
ax[0].text(xset, yset + 0.0025 , "$\mathcal{T}_2$")
ax[0].text(xset, yset + 0.005 , "$\mathcal{T}_3$")
xset = -0.05
ax[0].text(xset + -.1, 0.00625, "$W_{R} = 1x$")
ax[0].text(xset + 0.5, 0.00625, "$1.5x$")
ax[0].text(xset + 1, 0.00625, "$2x$")
#ax[0].text(xset + 1.5, 0.00625, "$3.5x$")
ax[0].set_xlabel("${PSO}_{R}$")
# PSO_VG
sizeScale = sizeScale * 0.7
x = [0, 0.5, 1,]# 1.5]
y = [0, 0, 0,]# 0]
ax[1].scatter(x, y, s=sizeScale, c=reds, marker="D", edgecolors='k')
y = [0.0025, 0.0025, 0.0025,]# 0.0025]
ax[1].scatter(x, y, s=sizeScale, c=greens, marker="D", edgecolor="k")
y = [0.005, 0.005, 0.005,]# 0.005]
ax[1].scatter(x, y, s=sizeScale, c=blues, marker="D", edgecolor="k")
ax[1].set_yticks([])
ax[1].set_xticks([])
ax[1].set_xlim(-0.1, 1.1)
ax[1].set_ylim(-0.001, 0.006)
xset = -0.2
yset = -0.0008
ax[1].text(xset, yset + 0, "$\mathcal{T}_1$")
ax[1].text(xset, yset + 0.0025 , "$\mathcal{T}_2$")
ax[1].text(xset, yset + 0.005 , "$\mathcal{T}_3$")
xset = -0.05
ax[1].text(xset + -.1, 0.00625, "$W_{R} = 1x$")
ax[1].text(xset + 0.5, 0.00625, "$1.5x$")
ax[1].text(xset + 1, 0.00625, "$2x$")
#ax[1].text(xset + 1.5, 0.00625, "$3.5x$")
ax[1].set_xlabel("${PSO}_{VG}$")

plt.tight_layout()
plt.savefig(outfileLegend, dpi=dpi)
