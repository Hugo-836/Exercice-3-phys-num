import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import glob
import re
import math
from scipy.signal import find_peaks
from scipy.special import ellipk


# ============================================================
# USER SETTINGS
# ============================================================

folder = r"/Users/pablitoooooo/Desktop/Physique Num/Exo 3/Exercice-3-phys-num/problème/Scan_nsteps_Trajectory_8.64e+04"

plot_layout = {
    "theta_time": True,
    "phase_space": False,
    "energy": True,
    "real_space": False,
    "power": True,
    "energy_balance": True
}

# ============================================================
# Output folder
# ============================================================

fig_dir = os.path.join(folder, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ============================================================
# Scan files
# ============================================================

files = sorted(glob.glob(os.path.join(folder, "*.txt")))

datasets = []
param_values = []
param_name = None

for f in files:

    name = os.path.basename(f)      # remove path
    name = name[:-4]                # remove ".txt"

    parts = name.split("_")

    param_name = parts[-2]          # scanned parameter
    value = float(parts[-1])        # parameter value

    data = np.loadtxt(f)

    datasets.append(data)
    param_values.append(value)

print(f"Found {len(datasets)} datasets.")

# Sort datasets
order = np.argsort(param_values)
param_values = np.array(param_values)[order]
datasets = [datasets[i] for i in order]

#--------PLOT--------#

for i,data in enumerate(datasets):
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    vx = data[:,3]
    vy = data[:,4]
    plt.figure()
    plt.scatter(x,y) 
    plt.xlabel(r"x [m]", fontsize=20)
    plt.ylabel(r"y [m]", fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,f"Trajectory: {param_name} = {param_values[i]}.png"), dpi=300)