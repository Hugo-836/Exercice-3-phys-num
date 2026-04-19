import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import glob
import re
import math
from scipy.interpolate import CubicSpline


# ============================================================
# USER SETTINGS
# ============================================================

folder = r"/Users/pablitoooooo/Desktop/Physique Num/Exo 3/Exercice-3-phys-num/problème/Scan_nsteps_Trajectory_8.64e+04"

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

#---------PLOT

R_T = 6378.1e3 
Perigee = R_T + 10000

perigee_altitudes = []
perigee_velocities = []

for i, data in enumerate(datasets):
    
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    vx = data[:,3]
    vy = data[:,4]

    r = np.sqrt(x**2 + y**2)

    #Perigee
    idx_p = np.argmin(r)
    r_p = r[idx_p]
    t_perigee = t[idx_p]    

    h_p = r_p - R_T

    Interpolation_Vx = CubicSpline(t, vx)
    Interpolation_Vy = CubicSpline(t, vy)
    vx_p = Interpolation_Vx(t_perigee)
    vy_p = Interpolation_Vy(t_perigee)

    v_p = np.sqrt(vx_p**2 + vy_p**2)

    perigee_altitudes.append(h_p)
    perigee_velocities.append(v_p)

    Interpolation_x = CubicSpline(t, x)
    Interpolation_y = CubicSpline(t, y)
    x_p = Interpolation_x(t_perigee)
    y_p = Interpolation_y(t_perigee)

    plt.figure(figsize=(6,6))
    plt.plot(x, y, label="Trajectory")
    plt.scatter(x_p, y_p, color='red', label="Perigee", zorder=5)

    #Terre
    theta = np.linspace(0, 2*np.pi, 500)
    plt.fill(R_T*np.cos(theta), R_T*np.sin(theta), color='blue', alpha=0.3, label="Earth")
    zoom = 5 * R_T
    plt.xlim(-zoom, zoom)
    plt.ylim(-zoom, zoom)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Number of steps = {param_values[i]}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"Trajectory_{i}.png"), dpi=300)
    plt.close()

# ---- Print results ----
for i in range(len(datasets)):
    print(f"Simulation {i}:")
    print(f"  Perigee altitude = {perigee_altitudes[i]/1000:.2f} km")
    print(f"  Perigee velocity = {perigee_velocities[i]/1000:.2f} km/s")

# ============================================================Velocity MAX

plt.figure(figsize=(6,6))
plt.loglog(172800/param_values, perigee_velocities, marker='o', label="Numerical values")
plt.axhline(11.121e3, color='red', linestyle='--', label="Analytical value")
plt.xlabel(r"Time steps d$t$")
plt.ylabel(r"Perigee velocity $v_{\text{max}}$ [m/s]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Perigee_velocity.png"), dpi=300)

# ============================================================Altitude MIN

plt.figure(figsize=(6,6))
plt.loglog(172800/param_values, perigee_altitudes, marker='o', label="Numerical values")
plt.axhline(10000, color='red', linestyle='--', label="Analytical value")
plt.xlabel(r"Time steps d$t$")
plt.ylabel(r"Perigee altitude $h_{\text{min}}$ [m]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Perigee_altitude.png"), dpi=300)


# ============================================================Error on Nf
vp = np.array(perigee_velocities)
error = np.abs(vp - 11121) / 11121

plt.figure()
plt.loglog(172800/param_values, error, 'r+-', label="numerical")
plt.loglog(172800/param_values, (172800/param_values)**4/1e6, 'k-.', label="O(dt^4)")
plt.xlabel(r"d$\overline{t}$")
plt.ylabel("Relative error on Nf")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"Nf_error.png"), dpi=300)


dt_values = 172800/param_values
vp = np.array(perigee_velocities)
errors = np.abs(vp - 11121)

print(dt_values)
# Compute order of convergence
orders = []
for i in range(len(errors)-1):
    p = np.log(errors[i]/errors[i+1]) / np.log(dt_values[i]/dt_values[i+1])
    orders.append(p)

# ---- Print results ----
for i, p in enumerate(orders):
    print(f"Order between dt={dt_values[i]} and dt={dt_values[i+1]}: p = {p:.3f}")

# ---- Log-log plot ----
plt.figure()
plt.loglog(dt_values, errors, 'o-', label="Error")

# Reference slope (example: order 2)
plt.loglog(dt_values, errors[0]*(dt_values/dt_values[0])**4, '--', label=r"Slope $\mathcal{O}(dt^4)$")

plt.xlabel("Time step Δt")
plt.ylabel("Error")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"Conv.png"), dpi=300)
