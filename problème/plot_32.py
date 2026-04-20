"""
plot_32.py — Tous les graphes pour la section 3.2
  3.2(a) : résultat analytique (imprimé et figure)
  3.2(b) : étude de convergence dt fixe (hmin, vmax vs nsteps)
  3.2(c) : pas adaptatif (dt vs t, dt vs r, hmin vs epsilon, comparaison)

Usage : python3 plot_32.py
Génère les figures dans le dossier 'figures_32/'
"""

import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(HERE, "figures_32")
os.makedirs(FIGDIR, exist_ok=True)

# ── Constantes physiques ──────────────────────────────────────────────────────
G   = 6.674e-11
mT  = 5.972e24
RT  = 6378100.0      # m
r0  = 314159000.0    # m
v0  = 1200.0         # m/s
h_p = 10000.0        # m (périgée cible)
mu  = G * mT

# ── Helpers de lecture (format N=2) ──────────────────────────────────────────
# Colonnes : t  x1 y1  x2 y2  vx1 vy1  vx2 vy2  E  px py  dt
COL_T   = 0
COL_X1, COL_Y1 = 1, 2
COL_X2, COL_Y2 = 3, 4
COL_VX1, COL_VY1 = 5, 6
COL_VX2, COL_VY2 = 7, 8
COL_E  = 9
COL_DT = 12

def load(fpath):
    d = np.loadtxt(fpath)
    if d.ndim == 1: d = d.reshape(1,-1)
    return d

def radius_from_earth(d):
    """Distance Artemis-Terre [m]."""
    return np.sqrt((d[:,COL_X2]-d[:,COL_X1])**2 + (d[:,COL_Y2]-d[:,COL_Y1])**2)

def speed_rel(d):
    """Vitesse relative Artemis/Terre [m/s]."""
    return np.sqrt((d[:,COL_VX2]-d[:,COL_VX1])**2 + (d[:,COL_VY2]-d[:,COL_VY1])**2)

def hmin_vmax(d):
    r   = radius_from_earth(d)
    spd = speed_rel(d)
    return float(r.min()) - RT, float(spd.max())

def nsteps_from_name(fp):
    return float(os.path.basename(fp).split("_")[-1].replace(".txt",""))

def eps_from_name(fp):
    return float(os.path.basename(fp).split("_")[-1].replace(".txt",""))

# ─────────────────────────────────────────────────────────────────────────────
# 3.2(a) — Calcul analytique
# ─────────────────────────────────────────────────────────────────────────────
def analytic_3_2a():
    rp    = RT + h_p
    Espec = 0.5*v0**2 - mu/r0
    a     = -mu/(2*Espec)
    ecc   = 1.0 - rp/a
    L     = np.sqrt(mu*a*(1-ecc**2))
    vt    = L/r0
    vr    = -np.sqrt(v0**2 - vt**2)
    vmax  = np.sqrt(mu*(2/rp - 1/a))
    T_orb = 2*np.pi*np.sqrt(a**3/mu)

    print("=" * 55)
    print("  3.2(a) — Résultat analytique")
    print("=" * 55)
    print(f"  r0         = {r0/1e3:.3f} km")
    print(f"  |v0|       = {v0:.1f} m/s")
    print(f"  h_perigee  = {h_p/1e3:.2f} km")
    print(f"  rp         = {rp/1e3:.4f} km")
    print(f"  a          = {a/1e6:.4f} × 10^6 m")
    print(f"  e          = {ecc:.6f}")
    print(f"  T_orb      = {T_orb/86400:.3f} jours")
    print(f"  vr(0)      = {vr:.4f} m/s  (radial, vers la Terre)")
    print(f"  vt(0)      = {vt:.4f} m/s  (tangentiel)")
    print(f"  vmax       = {vmax:.4f} m/s  (au périgée)")
    print("=" * 55)
    return {"a": a, "ecc": ecc, "vr": vr, "vt": vt, "vmax": vmax,
            "rp": rp, "T_orb": T_orb}

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Trajectoire de référence (nsteps=32000)
# ─────────────────────────────────────────────────────────────────────────────
def fig_trajectory(ana):
    fp = os.path.join(HERE, "Scan_32_fixed", "fixed_nsteps_32000.txt")
    d  = load(fp)
    dx = (d[:,COL_X2]-d[:,COL_X1]) / 1e3   # km
    dy = (d[:,COL_Y2]-d[:,COL_Y1]) / 1e3
    r  = np.sqrt(dx**2+dy**2)
    t  = d[:,COL_T] / 3600                  # heures

    fig = plt.figure(figsize=(13,5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # — Orbite dans l'espace —
    ax = fig.add_subplot(gs[0])
    theta = np.linspace(0, 2*np.pi, 500)
    ax.fill(RT/1e3*np.cos(theta), RT/1e3*np.sin(theta),
            color="#1f77b4", alpha=0.25, label="Terre")
    ax.plot(dx, dy, color="tab:orange", lw=0.8, label="Artemis II")
    ax.scatter([0],[0], color="#1f77b4", s=80, zorder=5)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Trajectoire dans l'espace")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # — Distance à la Terre —
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, r, color="tab:blue", lw=0.8)
    ax2.axhline(RT/1e3, color="tab:orange", ls="--", lw=1, label=f"Surface RT={RT/1e3:.0f} km")
    ax2.set_xlabel("t [h]")
    ax2.set_ylabel("Distance [km]")
    ax2.set_title("Distance Artemis—Terre")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    # — Vitesse —
    spd = speed_rel(d) / 1e3
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, spd, color="tab:red", lw=0.8)
    ax3.axhline(ana["vmax"]/1e3, color="tab:green", ls="--", lw=1,
                label=f"vmax analytique = {ana['vmax']/1e3:.3f} km/s")
    ax3.set_xlabel("t [h]")
    ax3.set_ylabel("|v| [km/s]")
    ax3.set_title("Vitesse d'Artemis")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)

    out = os.path.join(FIGDIR, "fig1_trajectoire.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Convergence dt fixe : hmin et vmax vs nsteps
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence_fixed(ana):
    files = sorted(glob.glob(os.path.join(HERE, "Scan_32_fixed", "*.txt")))
    ns, hs, vs = [], [], []
    for fp in files:
        d = load(fp)
        h_, v_ = hmin_vmax(d)
        ns.append(nsteps_from_name(fp))
        hs.append(h_)
        vs.append(v_)

    idx = np.argsort(ns)
    ns  = np.array(ns)[idx]
    hs  = np.array(hs)[idx]
    vs  = np.array(vs)[idx]

    # Valeur de référence = nsteps maximal
    h_ref = hs[-1]
    v_ref = vs[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(ns, np.abs(hs - h_ref) + 1, marker="o", color="tab:blue",
              label="|hmin − hmin_ref|")
    ax.axhline(np.abs(h_ref - h_p), color="tab:red", ls="--",
               label=f"|hmin_ref − analytique| = {abs(h_ref-h_p):.1f} m")
    # Pente théorique dt^4 (RK4) → nsteps^-4
    ns_fit = np.array([ns[0], ns[-1]])
    ax.loglog(ns_fit, 1e6*(ns_fit/ns[0])**(-4), "k--", alpha=0.5, label="pente ∝ nsteps⁻⁴")
    ax.set_xlabel("nsteps")
    ax.set_ylabel("|hmin − hmin_ref| [m]")
    ax.set_title("Convergence de hmin (dt fixe)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)

    ax = axes[1]
    ax.loglog(ns, np.abs(vs - v_ref) + 1e-3, marker="o", color="tab:orange",
              label="|vmax − vmax_ref|")
    ax.axhline(np.abs(v_ref - ana["vmax"]), color="tab:red", ls="--",
               label=f"|vmax_ref − analytique| = {abs(v_ref-ana['vmax']):.3f} m/s")
    ax.loglog(ns_fit, 1e3*(ns_fit/ns[0])**(-4), "k--", alpha=0.5, label="pente ∝ nsteps⁻⁴")
    ax.set_xlabel("nsteps")
    ax.set_ylabel("|vmax − vmax_ref| [m/s]")
    ax.set_title("Convergence de vmax (dt fixe)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)

    fig.suptitle("Section 3.2(b) — Convergence schéma à pas fixe", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_convergence_fixed.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # Imprimer tableau
    print(f"\n  {'nsteps':>8}  {'dt [s]':>10}  {'hmin [km]':>12}  {'vmax [km/s]':>12}")
    for n, h_, v_ in zip(ns, hs, vs):
        print(f"  {int(n):>8}  {172800/n:>10.2f}  {(h_+RT)/1e3:>12.4f}  {v_/1e3:>12.6f}")
    print(f"\n  Analytique : hmin = {h_p/1e3:.4f} km   vmax = {ana['vmax']/1e3:.6f} km/s")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Pas de temps adaptatif : dt vs t et dt vs distance
# ─────────────────────────────────────────────────────────────────────────────
def fig_adaptive_dt(ana):
    fp = os.path.join(HERE, "Scan_32_adaptive", "adaptive_eps_1.0e-08.txt")
    d  = load(fp)
    t  = d[:,COL_T] / 3600
    r  = radius_from_earth(d) / 1e3   # km
    dt = d[:,COL_DT]                  # s

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # — dt vs temps —
    ax = axes[0]
    ax.semilogy(t, dt, lw=0.7, color="tab:green")
    ax.set_xlabel("t [h]")
    ax.set_ylabel("Δt [s]")
    ax.set_title("Pas de temps adaptatif vs temps")
    ax.grid(True, which="both", alpha=0.4)

    # — Distance vs temps (comparaison) —
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(t, r, color="tab:blue", lw=0.8, label="distance [km]")
    ax2_twin.semilogy(t, dt, color="tab:green", lw=0.8, alpha=0.6, label="Δt [s]")
    ax2.axhline(RT/1e3, color="tab:orange", ls="--", lw=1)
    ax2.set_xlabel("t [h]")
    ax2.set_ylabel("Distance Terre [km]", color="tab:blue")
    ax2_twin.set_ylabel("Δt [s]", color="tab:green")
    ax2.set_title("Distance et Δt vs temps")
    ax2.grid(True, alpha=0.3)

    # — dt vs distance (échelle log-log) —
    ax3 = axes[2]
    sc = ax3.scatter(r, dt, c=t, s=4, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax3, label="t [h]")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Distance Artemis—Terre [km]")
    ax3.set_ylabel("Δt [s]")
    ax3.set_title("Δt adaptatif vs distance à la Terre")
    ax3.grid(True, which="both", alpha=0.4)

    fig.suptitle(f"Section 3.2(c) — Pas adaptatif (ε = 1e-8,  Nsteps = {len(d)-1})",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_adaptive_dt.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    print(f"     Δt min = {dt.min():.3f} s  Δt max = {dt.max():.0f} s")
    print(f"     Nsteps adaptatif = {len(d)-1}  (ε=1e-8)")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Convergence adaptatif : hmin et vmax vs epsilon
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence_adaptive(ana):
    files = sorted(glob.glob(os.path.join(HERE, "Scan_32_adaptive", "*.txt")))
    eps_l, hs_l, vs_l, ns_l = [], [], [], []
    for fp in files:
        d = load(fp)
        h_, v_ = hmin_vmax(d)
        eps_l.append(eps_from_name(fp))
        hs_l.append(h_)
        vs_l.append(v_)
        ns_l.append(len(d)-1)

    idx  = np.argsort(eps_l)
    eps  = np.array(eps_l)[idx]
    hs   = np.array(hs_l)[idx]
    vs   = np.array(vs_l)[idx]
    ns   = np.array(ns_l)[idx]

    h_ref = hs[np.argmin(eps)]   # valeur la plus précise
    v_ref = vs[np.argmin(eps)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.loglog(eps, np.abs(hs - h_ref) + 1e-3, marker="o", color="tab:blue")
    ax.axhline(np.abs(h_ref - h_p), color="tab:red", ls="--",
               label=f"|hmin_ref − analytique| ≈ {abs(h_ref-h_p):.1f} m")
    ax.set_xlabel("ε (tolérance)")
    ax.set_ylabel("|hmin − hmin_ref| [m]")
    ax.set_title("Convergence de hmin (adaptatif)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)
    ax.invert_xaxis()

    ax = axes[1]
    ax.loglog(eps, np.abs(vs - v_ref) + 1e-6, marker="o", color="tab:orange")
    ax.axhline(np.abs(v_ref - ana["vmax"]), color="tab:red", ls="--",
               label=f"|vmax_ref − analytique| ≈ {abs(v_ref-ana['vmax']):.3f} m/s")
    ax.set_xlabel("ε (tolérance)")
    ax.set_ylabel("|vmax − vmax_ref| [m/s]")
    ax.set_title("Convergence de vmax (adaptatif)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)
    ax.invert_xaxis()

    ax = axes[2]
    ax.loglog(eps, ns, marker="s", color="tab:green")
    ax.set_xlabel("ε (tolérance)")
    ax.set_ylabel("Nombre de pas de temps")
    ax.set_title("Nsteps adaptatif vs ε")
    ax.grid(True, which="both", alpha=0.4)
    ax.invert_xaxis()

    fig.suptitle("Section 3.2(c) — Convergence schéma adaptatif", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_convergence_adaptive.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    print(f"\n  {'ε':>10}  {'hmin [km]':>12}  {'vmax [km/s]':>12}  {'Nsteps':>8}")
    for e, h_, v_, n in zip(eps, hs, vs, ns):
        print(f"  {e:>10.1e}  {(h_+RT)/1e3:>12.5f}  {v_/1e3:>12.7f}  {n:>8d}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Comparaison fixe vs adaptatif : précision vs coût en pas de temps
# ─────────────────────────────────────────────────────────────────────────────
def fig_comparison(ana):
    files_f = sorted(glob.glob(os.path.join(HERE, "Scan_32_fixed", "*.txt")))
    files_a = sorted(glob.glob(os.path.join(HERE, "Scan_32_adaptive", "*.txt")))

    # Valeur de référence (adaptatif le plus fin)
    hs_a, vs_a, ns_a, eps_a = [], [], [], []
    for fp in files_a:
        d = load(fp); h_, v_ = hmin_vmax(d)
        eps_a.append(eps_from_name(fp)); hs_a.append(h_); vs_a.append(v_); ns_a.append(len(d)-1)
    idx = np.argsort(eps_a)
    eps_a = np.array(eps_a)[idx]; hs_a = np.array(hs_a)[idx]
    vs_a  = np.array(vs_a)[idx];  ns_a  = np.array(ns_a)[idx]
    h_ref = hs_a[-1]; v_ref = vs_a[-1]

    hs_f, vs_f, ns_f = [], [], []
    for fp in files_f:
        d = load(fp); h_, v_ = hmin_vmax(d)
        hs_f.append(h_); vs_f.append(v_); ns_f.append(nsteps_from_name(fp))
    idx = np.argsort(ns_f)
    ns_f = np.array(ns_f)[idx]; hs_f = np.array(hs_f)[idx]; vs_f = np.array(vs_f)[idx]

    err_h_f = np.abs(hs_f - h_ref) + 1
    err_h_a = np.abs(hs_a - h_ref) + 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ns_f, err_h_f, "o-", color="tab:blue", label="dt fixe")
    ax.loglog(ns_a, err_h_a, "s-", color="tab:orange", label="dt adaptatif (ε)")
    ax.set_xlabel("Nombre de pas de temps")
    ax.set_ylabel("|hmin − hmin_ref| [m]")
    ax.set_title("Section 3.2(c) — Comparaison : précision vs coût")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)

    out = os.path.join(FIGDIR, "fig5_comparaison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Conservation de l'énergie (adaptatif ε=1e-8)
# ─────────────────────────────────────────────────────────────────────────────
def fig_energy():
    fp = os.path.join(HERE, "Scan_32_adaptive", "adaptive_eps_1.0e-08.txt")
    d  = load(fp)
    t  = d[:,COL_T] / 3600
    E  = d[:,COL_E]
    r  = radius_from_earth(d) / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    rel_err = np.abs((E - E[0]) / E[0])
    ax.semilogy(t, rel_err + 1e-18, color="tab:red", lw=0.8)
    ax.set_xlabel("t [h]")
    ax.set_ylabel("|ΔE / E₀|")
    ax.set_title("Conservation de l'énergie mécanique (ε=1e-8)")
    ax.grid(True, which="both", alpha=0.4)
    ax.text(0.05, 0.92, f"erreur max = {rel_err.max():.2e}",
            transform=ax.transAxes, fontsize=9)

    # Énergie mécanique totale vs distance
    ax2 = axes[1]
    ax2.plot(r, E/1e9, lw=0.5, color="tab:purple", alpha=0.8)
    ax2.set_xlabel("Distance Artemis—Terre [km]")
    ax2.set_ylabel("Énergie mécanique [GJ]")
    ax2.set_title("Énergie vs distance (doit être constante)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig6_energie.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    print(f"     Erreur relative énergie max = {rel_err.max():.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Section 3.2 — Génération des figures ===\n")

    ana = analytic_3_2a()
    print()

    print("Figure 1 — Trajectoire de référence")
    fig_trajectory(ana)

    print("\nFigure 2 — Convergence dt fixe")
    fig_convergence_fixed(ana)

    print("\nFigure 3 — Évolution du pas adaptatif")
    fig_adaptive_dt(ana)

    print("\nFigure 4 — Convergence adaptatif")
    fig_convergence_adaptive(ana)

    print("\nFigure 5 — Comparaison fixe vs adaptatif")
    fig_comparison(ana)

    print("\nFigure 6 — Conservation de l'énergie")
    fig_energy()

    print(f"\nToutes les figures dans : {FIGDIR}/")

if __name__ == "__main__":
    main()
