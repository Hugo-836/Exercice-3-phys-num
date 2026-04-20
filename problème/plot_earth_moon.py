"""
plot_earth_moon.py — Tous les graphes pour la section 3.4 (système Terre-Lune)
  3.4(a) : orbites dans le référentiel barycentrique
  3.4(b) : conservation de l'énergie mécanique
  3.4(c) : conservation du moment cinétique
  3.4(d) : conservation de la distance Terre-Lune
  3.4(e) : vérification de la période orbitale

Usage : python3 plot_earth_moon.py
Génère les figures dans 'figures_earth_moon/'
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE   = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures_earth_moon")
os.makedirs(FIGDIR, exist_ok=True)

# ── Constantes physiques ──────────────────────────────────────────────────────
G   = 6.674e-11
mT  = 5.972e24
mL  = 7.342e22
M   = mT + mL
d   = 3.84400e8           # distance Terre-Lune [m]
Omega = math.sqrt(G*M/d**3)
T_orb = 2*math.pi/Omega   # période orbitale [s]

# ── Colonnes du fichier de sortie (N=2, drag=false) ──────────────────────────
# t  x1 y1  x2 y2  vx1 vy1  vx2 vy2  E  px py  dt
COL_T  = 0
COL_X1, COL_Y1   = 1, 2
COL_X2, COL_Y2   = 3, 4
COL_VX1, COL_VY1 = 5, 6
COL_VX2, COL_VY2 = 7, 8
COL_E   = 9
COL_PX, COL_PY = 10, 11
COL_DT  = 12

# ── Chargement ───────────────────────────────────────────────────────────────
fp = os.path.join(HERE, "output_earth_moon.out")
d_ = np.loadtxt(fp)

t   = d_[:, COL_T]
x1  = d_[:, COL_X1];  y1  = d_[:, COL_Y1]
x2  = d_[:, COL_X2];  y2  = d_[:, COL_Y2]
vx1 = d_[:, COL_VX1]; vy1 = d_[:, COL_VY1]
vx2 = d_[:, COL_VX2]; vy2 = d_[:, COL_VY2]
E   = d_[:, COL_E]
dt  = d_[:, COL_DT]

# Quantités dérivées
r12  = np.sqrt((x2-x1)**2 + (y2-y1)**2)   # distance Terre-Lune

# Moment cinétique (L_z = m*(x*vy - y*vx))
Lz = mT*(x1*vy1 - y1*vx1) + mL*(x2*vy2 - y2*vx2)

# Centre de masse (doit rester fixe)
xcm = (mT*x1 + mL*x2) / M
ycm = (mT*y1 + mL*y2) / M

# Valeurs initiales de référence
E0  = E[0]
Lz0 = Lz[0]
r0  = r12[0]
n_periods = t / T_orb

print(f"=== Section 3.4 — Système Terre-Lune ===")
print(f"Période analytique T = {T_orb:.6e} s ({T_orb/86400:.4f} jours)")
print(f"tf = {t[-1]:.6e} s  ({t[-1]/T_orb:.4f} périodes)")
print(f"E₀ = {E0:.6e} J")
print(f"ΔE/E₀ max = {np.abs((E-E0)/E0).max():.2e}")
print(f"ΔLz/Lz₀ max = {np.abs((Lz-Lz0)/Lz0).max():.2e}")
print(f"d₀ = {r0/1e3:.1f} km  (référence = {d/1e3:.1f} km)")
print(f"Δd/d₀ max = {np.abs((r12-r0)/r0).max():.2e}")
print(f"Δx_COM max = {np.abs(xcm).max():.2e} m")
print(f"Δy_COM max = {np.abs(ycm).max():.2e} m")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Orbites dans le référentiel barycentrique
# ─────────────────────────────────────────────────────────────────────────────
def fig_orbits():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # — Vue globale —
    ax = axes[0]
    ax.plot(x1/1e6, y1/1e6, color="tab:blue", lw=1.2, label="Terre")
    ax.plot(x2/1e6, y2/1e6, color="tab:orange", lw=0.8, label="Lune")
    ax.scatter([0], [0], s=80, color="gray", zorder=5, label="Barycentre")
    ax.scatter([x1[0]/1e6], [y1[0]/1e6], s=50, marker="o", color="tab:blue",
               edgecolors="k", zorder=5)
    ax.scatter([x2[0]/1e6], [y2[0]/1e6], s=30, marker="o", color="tab:orange",
               edgecolors="k", zorder=5)
    # Cercles théoriques
    theta = np.linspace(0, 2*np.pi, 500)
    r_T = mL/M * d; r_L = mT/M * d
    ax.plot(r_T/1e6*np.cos(theta), r_T/1e6*np.sin(theta),
            "b--", lw=0.6, alpha=0.5, label=f"Cercle théorique Terre (r={r_T/1e3:.0f} km)")
    ax.plot(r_L/1e6*np.cos(theta), r_L/1e6*np.sin(theta),
            "r--", lw=0.6, alpha=0.5, label=f"Cercle théorique Lune (r={r_L/1e3:.0f} km)")
    ax.set_xlabel("x [10⁶ m]")
    ax.set_ylabel("y [10⁶ m]")
    ax.set_title("Orbites dans le référentiel barycentrique\n(10 périodes)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # — Trajectoire de la Terre (zoom, doit être un cercle) —
    ax = axes[1]
    ax.plot(x1/1e3, y1/1e3, color="tab:blue", lw=1.5)
    ax.scatter([0], [0], s=80, color="gray", zorder=5, label="Barycentre")
    ax.plot(r_T/1e3*np.cos(theta), r_T/1e3*np.sin(theta),
            "r--", lw=1, label=f"Cercle parfait (r={r_T/1e3:.1f} km)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Orbite de la Terre (zoom)\ndoit être un cercle parfait")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Section 3.4 — Orbites Terre-Lune (référentiel barycentrique)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_orbits.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Conservations (énergie, moment cinétique, distance, COM)
# ─────────────────────────────────────────────────────────────────────────────
def fig_conservation():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # — Énergie relative —
    ax = axes[0, 0]
    rel_E = (E - E0) / np.abs(E0)
    ax.semilogy(n_periods, np.abs(rel_E) + 1e-20, color="tab:blue", lw=1)
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("|ΔE / E₀|")
    ax.set_title(f"Conservation de l'énergie\nΔE/E₀ max = {np.abs(rel_E).max():.2e}")
    ax.grid(True, which="both", alpha=0.4)

    # — Moment cinétique relatif —
    ax = axes[0, 1]
    rel_L = (Lz - Lz0) / np.abs(Lz0)
    ax.semilogy(n_periods, np.abs(rel_L) + 1e-20, color="tab:green", lw=1)
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("|ΔLz / Lz₀|")
    ax.set_title(f"Conservation du moment cinétique\nΔLz/Lz₀ max = {np.abs(rel_L).max():.2e}")
    ax.grid(True, which="both", alpha=0.4)

    # — Distance Terre-Lune —
    ax = axes[1, 0]
    rel_r = (r12 - r0) / r0
    ax.plot(n_periods, rel_r * 1e6, color="tab:red", lw=1)
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("Δd / d₀  [×10⁻⁶]")
    ax.set_title(f"Variation de la distance Terre-Lune\nΔd/d₀ max = {np.abs(rel_r).max():.2e}")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.grid(True, alpha=0.4)

    # — Position du centre de masse —
    ax = axes[1, 1]
    ax.plot(n_periods, xcm / 1e3, color="tab:purple", lw=1, label="x_COM")
    ax.plot(n_periods, ycm / 1e3, color="tab:brown", lw=1, ls="--", label="y_COM")
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("Position COM [km]")
    ax.set_title(f"Dérive du centre de masse\n|COM| max = {max(np.abs(xcm).max(), np.abs(ycm).max()):.1f} m")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    fig.suptitle("Section 3.4 — Lois de conservation (ε = 10⁻¹⁰)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_conservation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Vitesses et distance en fonction du temps
# ─────────────────────────────────────────────────────────────────────────────
def fig_dynamics():
    v1 = np.sqrt(vx1**2 + vy1**2)   # vitesse Terre
    v2 = np.sqrt(vx2**2 + vy2**2)   # vitesse Lune

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(n_periods, v1, color="tab:blue", lw=1, label="Terre")
    ax.plot(n_periods, v2, color="tab:orange", lw=1, label="Lune")
    ax.axhline(Omega*mL/M*d, color="tab:blue", ls="--", lw=0.8, alpha=0.7,
               label=f"v_T théor = {Omega*mL/M*d:.2f} m/s")
    ax.axhline(Omega*mT/M*d, color="tab:orange", ls="--", lw=0.8, alpha=0.7,
               label=f"v_L théor = {Omega*mT/M*d:.0f} m/s")
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("|v| [m/s]")
    ax.set_title("Vitesses Terre et Lune")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(n_periods, r12/1e3, color="tab:red", lw=1)
    ax.axhline(d/1e3, color="gray", ls="--", lw=1, label=f"d théorique = {d/1e3:.0f} km")
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("Distance Terre-Lune [km]")
    ax.set_title("Distance Terre-Lune vs temps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    ax = axes[2]
    ax.plot(n_periods, dt/3600, color="tab:cyan", lw=0.8)
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("Δt [h]")
    ax.set_title("Pas de temps adaptatif")
    ax.grid(True, alpha=0.4)
    ax.text(0.05, 0.92, f"<Δt> = {dt.mean()/3600:.2f} h\nΔt_max = {dt.max()/3600:.2f} h",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    fig.suptitle("Section 3.4 — Dynamique du système Terre-Lune", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_dynamics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Vérification de la période
# ─────────────────────────────────────────────────────────────────────────────
def fig_period():
    # La position angulaire de la Lune dans le référentiel barycentrique
    theta_L = np.unwrap(np.arctan2(y2, x2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(n_periods, np.degrees(theta_L), color="tab:orange", lw=1)
    # Droite analytique: theta = Omega * t (rad)
    theta_analytic = np.degrees(Omega * t)
    ax.plot(n_periods, theta_analytic % 360 - 360 + np.degrees(theta_L[0]),
            "r--", lw=0.8, alpha=0.7)
    # Ligne droite complète
    ax.plot(n_periods, np.degrees(Omega * t + np.arctan2(y2[0], x2[0])),
            "r--", lw=1, label=f"θ = Ωt + θ₀  (Ω = {Omega:.4e} rad/s)")
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("θ_Lune [°]  (angle déroulé)")
    ax.set_title("Angle orbital de la Lune vs temps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    ax = axes[1]
    theta_err = np.degrees(theta_L) - np.degrees(Omega * t + np.arctan2(y2[0], x2[0]))
    ax.plot(n_periods, theta_err, color="tab:purple", lw=1)
    ax.set_xlabel("Nombre de périodes")
    ax.set_ylabel("Erreur angulaire [°]")
    ax.set_title(f"Erreur angulaire par rapport à la rotation uniforme\n"
                 f"Erreur max = {np.abs(theta_err).max():.2e}°")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.grid(True, alpha=0.4)

    fig.suptitle("Section 3.4 — Vérification de la rotation uniforme", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_period.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Section 3.4 — Génération des figures ===\n")

    print("Figure 1 — Orbites barycentrique")
    fig_orbits()

    print("Figure 2 — Lois de conservation")
    fig_conservation()

    print("Figure 3 — Dynamique")
    fig_dynamics()

    print("Figure 4 — Vérification de la période")
    fig_period()

    print(f"\nToutes les figures dans : {FIGDIR}/")

if __name__ == "__main__":
    main()
