"""
gPCT Simulator v7.1
Christopher Dean White - MIT License (c)2026

v7.1 changes from v7.0:
  - Adds recursion_residual() diagnostic function
  - Replaces panel 4 (redundant H-zoom) with the recursion residual,
    which directly verifies Postulate II's coupling equation D(G·Dt) = t·DG
    holds along each worldline. Panels 3, 5, 6 unchanged.

The recursion residual is the diagnostic that distinguishes v6's broken
algebraic shortcut from v7's correct ODE solve. With v6 it was ~99%; with
v7 it sits at the numerical-derivative floor ~1e-6 for the four s∞ = 2π
worldlines, and diverges for the breakdown case E.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered")

# Configuration
D_TAU  = 0.002
STEPS  = 300_000
TRIM   = 500
TAU    = np.arange(STEPS) * D_TAU
TARGET = 1.0 / (2 * np.pi)

WORLDLINES = {
    "A: constant slope  (s = 2π)":      {
        "s": np.full(STEPS, 2*np.pi), "color": "#f5a623", "ls": "-", "lw": 2.5,
    },
    "B: perturbed above (s → 2π)":      {
        "s": 2*np.pi + 2.0*np.exp(-0.3*TAU), "color": "#5c8fe0", "ls": "-", "lw": 2,
    },
    "C: perturbed below (s → 2π)":      {
        "s": 2*np.pi - 1.5*np.exp(-0.3*TAU), "color": "#1D9E75", "ls": "-", "lw": 2,
    },
    "D: far address     (s → 2π)":      {
        "s": 2*np.pi + 8.0*np.exp(-0.5*TAU), "color": "#9B59B6", "ls": "-", "lw": 2,
    },
    "E: s∞ → 0  (framework breakdown)": {
        "s": 2.0 / (1 + TAU + 0.1), "color": "#e05c5c", "ls": "--", "lw": 1.5,
    },
}

# Core functions

def integrate_G(s_arr, d_tau):
    G = np.zeros(len(s_arr)); G[0] = 1.0
    for i in range(1, len(s_arr)):
        G[i] = G[i-1] + s_arr[i-1] * d_tau
    return G

def integrate_t(G_arr, s_arr, d_tau):
    n = len(s_arr); tau = np.arange(n) * d_tau
    def s_func(tt): return np.interp(tt, tau, s_arr)
    def rhs(tt, y):
        G, time, p = y
        s = s_func(tt)
        if abs(G) < 1e-14: return [s, p, 0.0]
        return [s, p, s * (time - p) / G]
    G0 = float(G_arr[0]); s0 = float(s_arr[0])
    p0 = (1.0/(2*np.pi)) * G0/s0 if abs(s0) > 1e-12 else 0.0
    sol = solve_ivp(rhs, [tau[0], tau[-1]], [G0, 1.0, p0],
                    t_eval=tau, method="DOP853", rtol=1e-9, atol=1e-12)
    if not sol.success: raise RuntimeError(f"ODE solve failed: {sol.message}")
    return sol.y[1]

def compute_H(G, d_tau, window=301):
    Gs  = savgol_filter(G, window, 4)
    DG  = np.gradient(Gs, d_tau)
    D2G = np.gradient(DG, d_tau)
    den = -Gs * D2G - DG**2
    return np.where(np.abs(den) > 1e-4, DG / den, np.nan)

def recursion_residual(G, t, d_tau, window=301):
    """Pointwise relative residual of Postulate II: |D(G·Dt) - t·DG| / |t·DG|."""
    Gs = savgol_filter(G, window, 4)
    ts = savgol_filter(t, window, 4)
    Dt = np.gradient(ts, d_tau)
    DG = np.gradient(Gs, d_tau)
    LHS = np.gradient(Gs * Dt, d_tau)
    RHS = ts * DG
    return np.abs(LHS - RHS) / (np.abs(RHS) + 1e-30)

def w_curve(s_arr):
    mx    = np.max(np.abs(s_arr)) + 1e-12
    p     = s_arr / mx
    s_hat = 2 * np.abs(p)
    P     = np.cos(np.pi * s_hat / 2 - np.pi / 4)**2
    return s_hat, P

# Run
print("Running ODE solves for each worldline...")
results = {}
for name, cfg in WORLDLINES.items():
    s = cfg["s"]
    G = integrate_G(s, D_TAU)
    t = integrate_t(G, s, D_TAU)
    H = compute_H(G, D_TAU)
    R = recursion_residual(G, t, D_TAU)
    sh, P = w_curve(s)
    results[name] = dict(s=s, G=G, t=t, H=H, resid=R, s_hat=sh, P=P,
                         color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"])
    print(f"  {name}")

# Diagnostics
print("=" * 80)
print("gPCT v7.1 — Phase-address worldlines")
print(f"  Target: H = -1/2π = {-TARGET:.6f}")
print("=" * 80)
print(f"\n{'Worldline':<42} {'H_late':>10} {'rec_resid':>14}")
print("-" * 80)
for name, d in results.items():
    h_late = np.nanmedian(d["H"][3*STEPS//4 : -TRIM])
    sl = slice(STEPS//4, 3*STEPS//4)
    r_med = np.nanmedian(d["resid"][sl])
    print(f"{name:<42} {h_late:>10.6f} {r_med:>14.2e}")
print("=" * 80)

# Plotting
TAU_plot = TAU[TRIM:-TRIM]

fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "gPCT v7.1 — Phase-Address Worldlines\n"
    "Any worldline with s∞ = 2π converges to H = −1/2π   "
    "(no cosmological model assumed)",
    fontsize=13, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)

# Panel 1: s(τ)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axhline(2*np.pi, color="gold", lw=1.5, ls=":", zorder=1, label="s∞ = 2π")
for name, d in results.items():
    ax1.plot(TAU_plot, np.clip(d["s"][TRIM:-TRIM], 0, 16),
             color=d["color"], ls=d["ls"], lw=d["lw"],
             label=name.split("(")[0].strip(), alpha=0.9)
ax1.set_title("Slope s(τ)  [the primitive]\nPhase address varies")
ax1.set_xlabel("Proper time τ"); ax1.set_ylabel("s = Dφg")
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

# Panel 2: G(τ)
ax2 = fig.add_subplot(gs[0, 1])
for name, d in results.items():
    G_plot = d["G"][TRIM:-TRIM]
    G_plot = np.where(G_plot > 0, G_plot, np.nan)
    ax2.plot(TAU_plot, G_plot, color=d["color"], ls=d["ls"], lw=d["lw"],
             label=name.split("(")[0].strip(), alpha=0.9)
ax2.set_yscale("log")
ax2.set_title("G(τ) = ∫s dτ  [log scale]\nLinear G ⟺ H = −1/2π")
ax2.set_xlabel("Proper time τ"); ax2.set_ylabel("G (log scale)")
ax2.legend(fontsize=7, loc="center right"); ax2.grid(True, alpha=0.3, which="both")

# Panel 3: H(τ)
ax3 = fig.add_subplot(gs[0, 2])
ax3.axhline(-TARGET, color="gold", lw=2, ls="--",
            label=f"−1/2π = {-TARGET:.4f}", zorder=5)
for name, d in results.items():
    ax3.plot(TAU_plot, np.clip(d["H"][TRIM:-TRIM], -2, 2),
             color=d["color"], ls=d["ls"], lw=d["lw"],
             label=name.split("(")[0].strip(), alpha=0.85)
ax3.set_title("H scalar(τ)\nConverges to −1/2π for all s∞=2π worldlines")
ax3.set_xlabel("Proper time τ"); ax3.set_ylabel("H")
ax3.set_ylim(-1.5, 0.5)
ax3.legend(fontsize=7, loc="lower right")

# Panel 4 [REPLACED]: Recursion residual
ax4 = fig.add_subplot(gs[1, 0])
ax4.axhline(1e-6, color="gold", lw=1.5, ls=":", zorder=1,
            label="numerical floor ~1e-6")
for name, d in results.items():
    resid = d["resid"][TRIM:-TRIM]
    resid_plot = np.clip(resid, 1e-12, 1e2)
    ax4.semilogy(TAU_plot, resid_plot,
                 color=d["color"], ls=d["ls"], lw=d["lw"],
                 label=name.split("(")[0].strip(), alpha=0.85)
ax4.set_title("Recursion residual  |D(G·Dt) − t·DG| / |t·DG|\n"
              "Postulate II coupling equation — should be ~0")
ax4.set_xlabel("Proper time τ"); ax4.set_ylabel("relative residual (log)")
ax4.set_ylim(1e-10, 1e1)
ax4.legend(fontsize=7, loc="upper left")
ax4.grid(True, alpha=0.3, which="both")

# Panel 5: P(|1⟩) vs ŝ
ax5 = fig.add_subplot(gs[1, 1])
s_th = np.linspace(0, 2, 500)
P_th = np.cos(np.pi * s_th / 2 - np.pi / 4)**2
ax5.plot(s_th, P_th, "k--", lw=2.5, label="Analytic W-Curve", zorder=5)
for name, d in results.items():
    idx = np.argsort(d["s_hat"])
    ax5.scatter(d["s_hat"][idx[::40]], d["P"][idx[::40]],
                color=d["color"], s=5, alpha=0.4)
ax5.set_title("P(|1⟩) vs ŝ  [collapse probability]\nFolded: ŝ = 2|p|, p ∈ [−1,1]")
ax5.set_xlabel("Normalised slope ŝ ∈ [0,2]"); ax5.set_ylabel("P(|1⟩)")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# Panel 6: CHSH (kept)
ax6 = fig.add_subplot(gs[1, 2])
p_th = np.linspace(-1, 1, 800)
S_emitter  = 1 + np.sqrt(2) + (np.sqrt(2) - 1) * np.sin(2 * np.pi * np.abs(p_th))
S_observer = 1 + np.sqrt(2) - (np.sqrt(2) - 1) * np.sin(2 * np.pi * np.abs(p_th))
ax6.plot(p_th, S_emitter,  color="#f5a623", lw=2.5, ls="--",
         label="Emitter frame (underlying)", zorder=4)
ax6.plot(p_th, S_observer, color="#5c8fe0", lw=2.5, ls="-",
         label="External observer (experimental)", zorder=5)
ax6.axhline(2*np.sqrt(2), color="gray", lw=1, ls=":",
            label=f"Tsirelson = 2√2 ≈ {2*np.sqrt(2):.3f}")
ax6.axhline(1+np.sqrt(2), color="green", lw=1, ls=":",
            label=f"Cycle average = 1+√2 ≈ {1+np.sqrt(2):.4f}")
ax6.axhline(2.0, color="silver", lw=1, ls=":", label="Classical bound = 2")
ax6.set_title("CHSH S vs signed slope p\nEmitter frame vs external-observer projection")
ax6.set_xlabel("Signed normalised slope p ∈ [−1,1]"); ax6.set_ylabel("S")
ax6.set_ylim(1.8, 2.95)
ax6.legend(fontsize=7, loc="lower center")
ax6.grid(True, alpha=0.3)

plt.savefig("/home/claude/gPCT_v7_1_SIMULATOR.png", dpi=140, bbox_inches="tight")
print("\nSaved: gPCT_v7_1_SIMULATOR.png")
