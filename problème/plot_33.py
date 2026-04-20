"""
plot_33.py — Tous les graphes pour la section 3.3 (atmosphère terrestre)
  3.3(a) : trajectoire avec traînée, accélération max, puissance max, convergence
  3.3(b) : scan de la direction initiale pour minimiser l'accélération max

Usage : python3 plot_33.py
Génère les figures dans le dossier 'figures_33/'
"""

import glob, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE   = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures_33")
os.makedirs(FIGDIR, exist_ok=True)

# ── Constantes physiques ──────────────────────────────────────────────────────
G   = 6.674e-11
mT  = 5.972e24
RT  = 6378100.0
r0  = 314159000.0
v0  = 1200.0
h_p = 10000.0
mu  = G * mT
rho0     = 1.2
lambda_a = 7238.2

# ── Format des colonnes N=2, drag=true ───────────────────────────────────────
# t  x1 y1  x2 y2  vx1 vy1  vx2 vy2  E  px py  dt  a_drag  P_drag
COL_T  = 0
COL_X1, COL_Y1 = 1, 2
COL_X2, COL_Y2 = 3, 4
COL_VX2, COL_VY2 = 7, 8
COL_E  = 9
COL_DT = 12
COL_ADRAG = 13
COL_PDRAG = 14

def load(fpath):
    d = np.loadtxt(fpath)
    if d.ndim == 1: d = d.reshape(1, -1)
    return d

def dist_earth(d):
    return np.sqrt((d[:,COL_X2]-d[:,COL_X1])**2 + (d[:,COL_Y2]-d[:,COL_Y1])**2)

def speed(d):
    return np.sqrt((d[:,COL_VX2]-d[:,COL_X1])**2 + (d[:,COL_VY2]-d[:,COL_Y1])**2)

def speed_rel(d):
    return np.sqrt((d[:,COL_VX2]-d[:,1+4])**2 + (d[:,COL_VY2]-d[:,1+5])**2)

def speed_art(d):
    vx = d[:,COL_VX2] - d[:,5]
    vy = d[:,COL_VY2] - d[:,6]
    return np.sqrt(vx**2 + vy**2)

def analytic_v0():
    rp = RT + h_p
    Espec = 0.5*v0**2 - mu/r0
    a = -mu/(2*Espec)
    ecc = 1 - rp/a
    L = math.sqrt(mu*a*(1-ecc**2))
    vt = L/r0
    vr = -math.sqrt(v0**2 - vt**2)
    theta = math.atan2(vt, vr)
    return vr, vt, theta

def eps_from_name(fp):
    return float(os.path.basename(fp).split("_")[-1].replace(".txt",""))

def angle_from_name(fp):
    s = os.path.basename(fp).replace("dir_","").replace(".txt","")
    return float(s)

def _safe_angle(fp):
    try:
        return angle_from_name(fp)
    except ValueError:
        return math.nan

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Vue d'ensemble de la rentrée atmosphérique (CI de 3.2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_overview():
    fp = os.path.join(HERE, "Scan_33_convergence", "drag_eps_1.0e-08.txt")
    d  = load(fp)
    t  = d[:,COL_T] / 3600
    r  = dist_earth(d) / 1e3    # km
    alt = r - RT/1e3            # altitude km

    has_drag = d.shape[1] >= 15
    a_drag = d[:,COL_ADRAG] if has_drag else np.zeros(len(t))
    P_drag = d[:,COL_PDRAG] if has_drag else np.zeros(len(t))

    spd = speed_art(d) / 1e3   # km/s

    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # ── Trajectoire ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    dx = (d[:,COL_X2] - d[:,COL_X1]) / 1e3
    dy = (d[:,COL_Y2] - d[:,COL_Y1]) / 1e3
    # Colorer par altitude pour voir l'entrée atm
    sc = ax.scatter(dx, dy, c=alt, s=2, cmap="RdYlGn_r",
                    vmin=0, vmax=1000, zorder=2)
    plt.colorbar(sc, ax=ax, label="Altitude [km]")
    theta = np.linspace(0, 2*np.pi, 500)
    ax.fill(RT/1e3*np.cos(theta), RT/1e3*np.sin(theta),
            color="#1f77b4", alpha=0.25)
    # Cercle à 100 km (limite atm indicative)
    ax.plot(*(((RT+100000)/1e3)*np.array([np.cos(theta),np.sin(theta)])),
            "g--", lw=0.7, alpha=0.7, label="alt=100 km")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title("Trajectoire (couleur = altitude)")
    ax.axis("equal"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Altitude vs temps ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, alt, color="tab:blue", lw=0.8)
    ax.axhline(0, color="tab:orange", ls="--", lw=1, label="Surface Terre")
    ax.axhline(10, color="tab:green", ls=":", lw=1, label="h=10 km (cible)")
    ax.set_xlabel("t [h]"); ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude d'Artemis vs temps")
    ax.set_ylim(-500, None)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Vitesse vs temps ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, spd, color="tab:red", lw=0.8)
    ax.set_xlabel("t [h]"); ax.set_ylabel("|v| [km/s]")
    ax.set_title("Vitesse d'Artemis")
    ax.grid(True, alpha=0.3)

    # ── Accélération de traînée vs altitude ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    mask = alt < 500    # Zoom sur l'entrée atmosphérique
    ax.plot(alt[mask], a_drag[mask] / 9.81, color="tab:purple", lw=0.8)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Accélération de traînée [g]")
    ax.set_title("Accélération vs altitude (entrée atm)")
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    ax.grid(True, alpha=0.3)
    amax_g = a_drag.max() / 9.81
    ax.text(0.97, 0.95, f"amax = {amax_g:.2f} g", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    # ── Puissance de traînée vs temps ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, np.abs(P_drag) / 1e6, color="tab:brown", lw=0.8)
    ax.set_xlabel("t [h]"); ax.set_ylabel("|P_drag| [MW]")
    ax.set_title("Puissance de la force de traînée")
    ax.grid(True, alpha=0.3)
    Pmax = np.abs(P_drag).max()
    ax.text(0.97, 0.95, f"|P|max = {Pmax/1e6:.2f} MW", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    # ── Densité de l'atmosphère ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    h_arr = np.linspace(0, 200, 500)   # km
    rho_arr = rho0 * np.exp(-h_arr * 1e3 / lambda_a)
    ax.semilogy(h_arr, rho_arr, color="tab:cyan", lw=1.2)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("ρ [kg/m³]")
    ax.set_title("Profil de densité atmosphérique")
    ax.axvline(h_p/1e3, color="tab:green", ls=":", label=f"h={h_p/1e3:.0f} km")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Section 3.3(a) — Rentrée atmosphérique (ε = 1×10⁻⁸)", fontsize=12)
    out = os.path.join(FIGDIR, "fig1_overview.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    amax   = a_drag.max()
    hmin   = float(dist_earth(d).min()) - RT
    Pmax_v = np.abs(P_drag).max()
    print(f"     amax   = {amax:.3f} m/s²  = {amax/9.81:.3f} g")
    print(f"     |P|max = {Pmax_v/1e6:.3f} MW")
    print(f"     hmin   = {hmin:.1f} m  ({hmin/1e3:.4f} km)")
    return amax, Pmax_v

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Zoom sur la phase d'entrée atmosphérique
# ─────────────────────────────────────────────────────────────────────────────
def fig_atm_entry():
    fp = os.path.join(HERE, "Scan_33_convergence", "drag_eps_1.0e-08.txt")
    d  = load(fp)
    t  = d[:,COL_T]
    r  = dist_earth(d)
    alt = (r - RT) / 1e3   # km

    a_drag = d[:,COL_ADRAG]
    P_drag = d[:,COL_PDRAG]
    spd    = speed_art(d) / 1e3

    # Garder uniquement la phase atmosphérique (alt < 200 km)
    mask = alt < 200
    t_m  = t[mask] / 3600
    alt_m = alt[mask]
    a_m  = a_drag[mask]
    P_m  = P_drag[mask]
    s_m  = spd[mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0,0]
    ax.plot(t_m, alt_m, color="tab:blue", lw=1)
    ax.set_xlabel("t [h]"); ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude lors de l'entrée atmosphérique")
    ax.axhline(0, color="tab:orange", ls="--", lw=1)
    ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(t_m, s_m, color="tab:red", lw=1)
    ax.set_xlabel("t [h]"); ax.set_ylabel("|v| [km/s]")
    ax.set_title("Vitesse lors de l'entrée")
    ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax2 = ax.twinx()
    ax.plot(t_m, a_m / 9.81, color="tab:purple", lw=1, label="a_drag [g]")
    ax2.plot(t_m, alt_m, color="tab:blue", lw=0.7, alpha=0.5, label="alt [km]")
    ax.set_xlabel("t [h]"); ax.set_ylabel("Accélération [g]", color="tab:purple")
    ax2.set_ylabel("Altitude [km]", color="tab:blue")
    ax.set_title("Accélération de traînée et altitude")
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.92, f"amax = {a_m.max()/9.81:.2f} g",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    ax = axes[1,1]
    ax2 = ax.twinx()
    ax.plot(t_m, np.abs(P_m)/1e6, color="tab:brown", lw=1, label="|P| [MW]")
    ax2.plot(t_m, alt_m, color="tab:blue", lw=0.7, alpha=0.5)
    ax.set_xlabel("t [h]"); ax.set_ylabel("|P_drag| [MW]", color="tab:brown")
    ax2.set_ylabel("Altitude [km]", color="tab:blue")
    ax.set_title("Puissance de traînée")
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.92, f"|P|max = {np.abs(P_m).max()/1e6:.2f} MW",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    fig.suptitle("Section 3.3(a) — Zoom sur la phase atmosphérique", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_atm_entry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Convergence : amax et Pmax vs epsilon
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence():
    files = sorted(glob.glob(os.path.join(HERE, "Scan_33_convergence", "*.txt")))
    eps_l, amax_l, Pmax_l, hmin_l, nsteps_l = [], [], [], [], []
    for fp in files:
        d = load(fp)
        if d.shape[1] < 15: continue
        eps_l.append(eps_from_name(fp))
        amax_l.append(d[:,COL_ADRAG].max())
        Pmax_l.append(np.abs(d[:,COL_PDRAG]).max())
        hmin_l.append(float(dist_earth(d).min()) - RT)
        nsteps_l.append(len(d)-1)

    idx  = np.argsort(eps_l)
    eps  = np.array(eps_l)[idx]
    amax = np.array(amax_l)[idx]
    Pmax = np.array(Pmax_l)[idx]
    hmin = np.array(hmin_l)[idx]
    ns   = np.array(nsteps_l)[idx]

    ref_a = amax[np.argmin(eps)]
    ref_P = Pmax[np.argmin(eps)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0,0]
    ax.loglog(eps, np.abs(amax - ref_a) + 1e-4, "o-", color="tab:purple")
    ax.set_xlabel("ε"); ax.set_ylabel("|amax − amax_ref| [m/s²]")
    ax.set_title("Convergence de l'accélération maximale")
    ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.4)

    ax = axes[0,1]
    ax.loglog(eps, np.abs(Pmax - ref_P) + 1, "o-", color="tab:brown")
    ax.set_xlabel("ε"); ax.set_ylabel("|Pmax − Pmax_ref| [W]")
    ax.set_title("Convergence de la puissance maximale")
    ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.4)

    ax = axes[1,0]
    ax.semilogx(eps, amax / 9.81, "o-", color="tab:purple")
    ax.set_xlabel("ε"); ax.set_ylabel("amax [g]")
    ax.set_title("Accélération maximale vs ε")
    ax.invert_xaxis(); ax.grid(True, alpha=0.4)

    ax = axes[1,1]
    ax.loglog(eps, ns, "s-", color="tab:green")
    ax.set_xlabel("ε"); ax.set_ylabel("Nombre de pas")
    ax.set_title("Coût (Nsteps) vs ε")
    ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.4)

    fig.suptitle("Section 3.3(a) — Étude de convergence (schéma adaptatif)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    print(f"\n  {'ε':>10}  {'amax [g]':>10}  {'|P|max [MW]':>12}  {'hmin [km]':>10}  {'Nsteps':>8}")
    for e, a, p, h_, n in zip(eps, amax, Pmax, hmin, ns):
        print(f"  {e:>10.1e}  {a/9.81:>10.4f}  {p/1e6:>12.4f}  {h_/1e3:>10.4f}  {n:>8d}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Scan de direction : amax et hmin vs angle
# ─────────────────────────────────────────────────────────────────────────────
def fig_direction_scan():
    files = sorted(glob.glob(os.path.join(HERE, "Scan_33_direction", "*.txt")))
    angles, amax_l, hmin_l, lands_l = [], [], [], []

    for fp in files:
        try:
            ang = angle_from_name(fp)
        except ValueError:
            continue
        d = load(fp)
        r   = dist_earth(d)
        hmin = float(r.min()) - RT
        lands = hmin < 0    # collision = atterrissage

        amax = d[:,COL_ADRAG].max() if d.shape[1] >= 15 else 0.0
        angles.append(ang); amax_l.append(amax); hmin_l.append(hmin)
        lands_l.append(lands)

    idx = np.argsort(angles)
    angles = np.array(angles)[idx]
    amax   = np.array(amax_l)[idx]
    hmin   = np.array(hmin_l)[idx]
    lands  = np.array(lands_l)[idx]

    # Trouver l'angle optimal (minimise amax parmi ceux qui atterrissent)
    mask_land = lands
    if mask_land.any():
        i_opt = np.argmin(amax[mask_land])
        ang_opt = angles[mask_land][i_opt]
        amax_opt = amax[mask_land][i_opt]
    else:
        ang_opt = 0.0; amax_opt = amax.min()

    # Séparer la fenêtre de rentrée rasante (aerobraking) des rentrées directes
    aerobraking = mask_land & (angles < -0.01)
    direct      = mask_land & (angles >= -0.01)
    flyby       = ~mask_land

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # — amax vs angle (zoom sur la fenêtre critique) —
    ax = axes[0]
    # Trajectoires sans atterrissage proches de 0
    ax.plot(angles[flyby & (angles > -0.06)], amax[flyby & (angles > -0.06)]/9.81,
            "o", color="tab:blue", ms=5, label="Survol (pas d'atterrissage)")
    ax.plot(angles[aerobraking], amax[aerobraking]/9.81, "s", color="tab:orange",
            ms=6, label="Aerobraking multi-passage")
    ax.plot(angles[direct], amax[direct]/9.81, "^", color="tab:red",
            ms=5, label="Rentrée directe")
    ax.axvline(ang_opt, color="tab:green", ls="--", lw=1.5,
               label=f"Optimal Δα={ang_opt:+.3f}°\n({amax_opt/9.81:.1f} g)")
    ax.axvline(0, color="gray", ls=":", lw=1, label="Référence 3.2(a)\n(22.8 g)")
    ax.set_xlabel("Δα [°]  (rotation autour de la direction analytique)")
    ax.set_ylabel("amax [g]")
    ax.set_title("Accélération maximale vs direction\n(zoom fenêtre critique)")
    ax.set_xlim(-0.06, 0.06)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # — amax vs angle (vue large) —
    ax = axes[1]
    ax.plot(angles[flyby], amax[flyby]/9.81, "o", color="tab:blue", ms=3,
            label="Survol")
    ax.plot(angles[mask_land], amax[mask_land]/9.81, "s", color="tab:red",
            ms=4, label="Atterrissage")
    ax.axvline(ang_opt, color="tab:green", ls="--", lw=1.5,
               label=f"Optimal Δα={ang_opt:+.3f}°")
    ax.set_xlabel("Δα [°]")
    ax.set_ylabel("amax [g]")
    ax.set_title("Accélération maximale vs direction\n(vue globale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # — hmin vs angle (zoom) —
    ax = axes[2]
    ax.plot(angles[flyby & (angles > -0.06)],
            hmin[flyby & (angles > -0.06)]/1e3,
            "o", color="tab:blue", ms=5, label="Survol")
    ax.plot(angles[mask_land & (angles > -0.06)],
            hmin[mask_land & (angles > -0.06)]/1e3,
            "s", color="tab:red", ms=5, label="Atterrissage")
    ax.axhline(0, color="tab:orange", ls="--", lw=1.5, label="Surface Terre")
    ax.axvline(ang_opt, color="tab:green", ls="--", lw=1.5, label=f"Optimal")
    ax.axvline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Δα [°]")
    ax.set_ylabel("hmin [km]")
    ax.set_title("Altitude minimale vs direction\n(zoom fenêtre critique)")
    ax.set_xlim(-0.06, 0.06)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle("Section 3.3(b) — Scan de la direction initiale", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_direction_scan.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    print(f"     Angle optimal : Δα = {ang_opt:+.3f}°,  amax = {amax_opt/9.81:.3f} g")
    print(f"     Réduction par rapport à la référence : {(1 - amax_opt/amax[angles==0.0][0])*100:.1f}%"
          if (angles==0.0).any() else "")
    return ang_opt

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Trajectoire et profils pour l'angle optimal (3.3b)
# ─────────────────────────────────────────────────────────────────────────────
def fig_optimal(ang_opt):
    # Trouver le fichier le plus proche de ang_opt
    files = sorted(glob.glob(os.path.join(HERE, "Scan_33_direction", "*.txt")))
    valid = [f for f in files if not math.isnan(_safe_angle(f))]
    best  = min(valid, key=lambda f: abs(_safe_angle(f) - ang_opt))
    d_opt = load(best)

    # Référence sans optimisation (Δα=0)
    ref_f = min(valid, key=lambda f: abs(_safe_angle(f) - 0.0))
    d_ref = load(ref_f)

    def extract(d):
        t   = d[:,COL_T] / 3600
        r   = dist_earth(d) / 1e3
        alt = r - RT/1e3
        a   = d[:,COL_ADRAG] if d.shape[1] >= 15 else np.zeros(len(t))
        return t, alt, a

    t_opt, alt_opt, a_opt = extract(d_opt)
    t_ref, alt_ref, a_ref = extract(d_ref)

    ang_opt_actual = _safe_angle(best)
    ang_ref_actual = _safe_angle(ref_f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    theta = np.linspace(0, 2*np.pi, 500)

    # — Trajectoires superposées (vue globale) —
    ax = axes[0, 0]
    dx_opt = (d_opt[:,COL_X2]-d_opt[:,COL_X1])/1e3
    dy_opt = (d_opt[:,COL_Y2]-d_opt[:,COL_Y1])/1e3
    dx_ref = (d_ref[:,COL_X2]-d_ref[:,COL_X1])/1e3
    dy_ref = (d_ref[:,COL_Y2]-d_ref[:,COL_Y1])/1e3
    ax.fill(RT/1e3*np.cos(theta), RT/1e3*np.sin(theta),
            color="#1f77b4", alpha=0.25)
    ax.plot(dx_ref, dy_ref, color="tab:blue", lw=0.7, alpha=0.8,
            label=f"Référence (Δα=0°, {a_ref.max()/9.81:.1f} g)")
    ax.plot(dx_opt, dy_opt, color="tab:green", lw=1,
            label=f"Optimal (Δα={ang_opt_actual:+.3f}°, {a_opt.max()/9.81:.1f} g)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title("Trajectoires comparées (vue globale)")
    ax.axis("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # — Altitude vs temps —
    ax = axes[0, 1]
    ax.plot(t_ref, alt_ref, color="tab:blue", lw=0.8, alpha=0.8,
            label=f"Référence (amax={a_ref.max()/9.81:.1f} g)")
    ax.plot(t_opt, alt_opt, color="tab:green", lw=1,
            label=f"Optimal (amax={a_opt.max()/9.81:.1f} g)")
    ax.axhline(0, color="tab:orange", ls="--", lw=1, label="Surface")
    ax.axhline(100, color="gray", ls=":", lw=0.7, label="100 km")
    ax.set_xlabel("t [h]"); ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude vs temps")
    ax.set_ylim(-500, None); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # — Zoom entrée atmosphérique (alt < 600 km) —
    ax = axes[1, 0]
    mask_r = alt_ref < 600
    mask_o = alt_opt < 600
    ax.plot(t_ref[mask_r], alt_ref[mask_r], color="tab:blue", lw=1,
            label="Référence (rentrée directe)")
    ax.plot(t_opt[mask_o], alt_opt[mask_o], color="tab:green", lw=1,
            label=f"Optimal (aerobraking + rentrée)")
    ax.axhline(0, color="tab:orange", ls="--", lw=1)
    ax.axhline(100, color="gray", ls=":", lw=0.7, label="100 km (Kármán)")
    ax.set_xlabel("t [h]"); ax.set_ylabel("Altitude [km]")
    ax.set_title("Zoom : phase atmosphérique (alt < 600 km)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # — Accélération de traînée vs altitude —
    ax = axes[1, 1]
    mask_r = alt_ref < 300
    mask_o = alt_opt < 300
    ax.plot(alt_ref[mask_r], a_ref[mask_r]/9.81, color="tab:blue", lw=1,
            label=f"Référence (Δα=0°) — {a_ref.max()/9.81:.1f} g")
    ax.plot(alt_opt[mask_o], a_opt[mask_o]/9.81, color="tab:green", lw=1,
            label=f"Optimal (Δα={ang_opt_actual:+.3f}°) — {a_opt.max()/9.81:.1f} g")
    ax.set_xlabel("Altitude [km]"); ax.set_ylabel("Accélération de traînée [g]")
    ax.set_title("Accélération vs altitude lors de l'entrée")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    reduction = (1 - a_opt.max()/a_ref.max()) * 100
    fig.suptitle(
        f"Section 3.3(b) — Comparaison référence vs direction optimale\n"
        f"Réduction de amax : {reduction:.0f}%  "
        f"({a_ref.max()/9.81:.1f} g → {a_opt.max()/9.81:.1f} g)",
        fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig5_optimal.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Section 3.3 — Génération des figures ===\n")

    print("Figure 1 — Vue d'ensemble de la rentrée")
    amax, Pmax = fig_overview()

    print("\nFigure 2 — Zoom sur la phase atmosphérique")
    fig_atm_entry()

    print("\nFigure 3 — Étude de convergence")
    fig_convergence()

    print("\nFigure 4 — Scan de la direction initiale")
    ang_opt = fig_direction_scan()

    print("\nFigure 5 — Trajectoire optimale vs référence")
    fig_optimal(ang_opt)

    print(f"\nToutes les figures dans : {FIGDIR}/")

    print("\n--- Résumé 3.3(a) ---")
    print(f"  CI de 3.2(a)  →  amax = {amax/9.81:.3f} g  |P|max = {Pmax/1e6:.3f} MW")

if __name__ == "__main__":
    main()
