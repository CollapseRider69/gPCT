"""
gPCT Simulator v5.0
Gravitational Phase-Cancellation Theory
Christopher Dean White
ORCID: 0009-0002-7866-7078
MIT License ©2025

Companion code to "A Minimal Statement of Gravitational Phase-Cancellation
Theory (gPCT)." Numerically verifies the recursion scalar H (Postulate II)
converges to −1/2π (Postulate III) for any worldline whose gravitational
phase slope s = Dφg is asymptotically constant (s∞ = 2π).

The primitive is slope. No cosmological model is assumed. Worldlines are
specified by their phase address — an arbitrary starting point in phase
space — and their asymptotic slope. The geometry that results from
s∞ = 2π is de Sitter; this is a consequence of the phase normalization
condition, not an input.

Five worldlines:
  A. Constant slope        s = 2π exactly         (baseline)
  B. Perturbed above       s starts at 2π+2       (phase address above attractor)
  C. Perturbed below       s starts at 2π−1.5     (phase address below attractor)
  D. Far phase address     s starts at 10         (distant phase address)
  E. s∞ → 0                decaying slope field   (framework breakdown case)

Functions implement:
  integrate_G   — Postulate II: DG = s
  integrate_t   — Postulate II: Dt·t = f(s), with f determined by H = −1/2π
  compute_H     — Postulate II recursion scalar: H = DG / (−G·D²G − (DG)²)
  w_curve       — Collapse prediction: P(|1⟩) = cos²(πŝ/2 − π/4)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

# Suppress expected NaN warning from worldline E (framework breakdown case)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Configuration
D_TAU  = 0.002
STEPS  = 1_000_000
TRIM   = 500          # trim this many points from plot edges to remove boundary artifacts
TAU    = np.arange(STEPS) * D_TAU
TARGET = 1.0 / (2 * np.pi)

# Worldlines
WORLDLINES = {
    "A: constant slope  (s = 2π)":      {
        "s": np.full(STEPS, 2*np.pi),
        "color": "#f5a623", "ls": "-",  "lw": 2.5,
    },
    "B: perturbed above (s → 2π)":      {
        "s": 2*np.pi + 2.0*np.exp(-0.3*TAU),
        "color": "#5c8fe0", "ls": "-",  "lw": 2,
    },
    "C: perturbed below (s → 2π)":      {
        "s": 2*np.pi - 1.5*np.exp(-0.3*TAU),
        "color": "#1D9E75", "ls": "-",  "lw": 2,
    },
    "D: far address     (s → 2π)":      {
        "s": 2*np.pi + 8.0*np.exp(-0.5*TAU),
        "color": "#9B59B6", "ls": "-",  "lw": 2,
    },
    "E: s∞ → 0  (framework breakdown)": {
        "s": 2.0 / (1 + TAU + 0.1),
        "color": "#e05c5c", "ls": "--", "lw": 1.5,
    },
}

# Core functions

def integrate_G(s_arr, d_tau):
    """DG = s  (Postulate II). Euler integration."""
    G = np.zeros(len(s_arr)); G[0] = 1.0
    for i in range(1, len(s_arr)):
        G[i] = G[i-1] + s_arr[i-1] * d_tau
    return G

def integrate_t(G_arr, s_arr, d_tau):
    """Relational time from Dt·t = f(s), with f = (1/2π)(G/s)."""
    WHITE = 1.0 / (2 * np.pi)
    t = np.zeros(len(s_arr)); t[0] = 1.0
    for i in range(1, len(s_arr)):
        si   = s_arr[i-1]
        dt   = WHITE * (G_arr[i] / si) if abs(si) > 1e-12 else 0.0
        t[i] = t[i-1] + dt * d_tau
    return t

def compute_H(G, d_tau, window=301):
    """Recursion scalar H = DG / (−G·D²G − (DG)²)  (Postulate II)."""
    Gs  = savgol_filter(G, window, 4)
    DG  = np.gradient(Gs, d_tau)
    D2G = np.gradient(DG, d_tau)
    den = -Gs * D2G - DG**2
    return np.where(np.abs(den) > 1e-4, DG / den, np.nan)

def w_curve(s_arr):
    """Collapse prediction P(|1⟩) = cos²(πŝ/2 − π/4)  (paper p.2)."""
    mx    = np.max(np.abs(s_arr)) + 1e-12
    p     = s_arr / mx
    s_hat = 2 * np.abs(p)
    P     = np.cos(np.pi * s_hat / 2 - np.pi / 4)**2
    return s_hat, P

# Run
results = {}
for name, cfg in WORLDLINES.items():
    s  = cfg["s"]
    G  = integrate_G(s, D_TAU)
    t  = integrate_t(G, s, D_TAU)
    H  = compute_H(G, D_TAU)
    sh, P = w_curve(s)
    results[name] = dict(s=s, G=G, t=t, H=H, s_hat=sh, P=P,
                         color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"])

# Diagnostics
print("=" * 72)
print("gPCT v5 — Phase-address worldlines")
print(f"  Target: H = −1/2π = {-TARGET:.6f}")
print("=" * 72)
print(f"\n{'Worldline':<42} {'H_mid':>10}  {'H_late':>10}  {'Δ(late)':>10}")
print("-" * 76)
for name, d in results.items():
    h_mid  = np.nanmedian(d["H"][STEPS//4 : 3*STEPS//4])
    h_late = np.nanmedian(d["H"][3*STEPS//4 :])
    delta  = h_late + TARGET
    print(f"{name:<42} {h_mid:>10.6f}  {h_late:>10.6f}  {delta:>10.6f}")
print("=" * 72)

# Plot helpers — trimmed tau axis to remove boundary artifacts
TAU_plot = TAU[TRIM:-TRIM]

# Plots
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "gPCT v5 — Phase-Address Worldlines\n"
    "Any worldline with s∞ = 2π converges to H = −1/2π  "
    "(no cosmological model assumed)",
    fontsize=13, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)

ax1 = fig.add_subplot(gs[0, 0])
ax1.axhline(2*np.pi, color="gold", lw=1.5, ls=":", zorder=1,
            label="s∞ = 2π")
for name, d in results.items():
    ax1.plot(TAU_plot, np.clip(d["s"][TRIM:-TRIM], 0, 16),
             color=d["color"], ls=d["ls"], lw=d["lw"],
             label=name.split("(")[0].strip(), alpha=0.9)
ax1.set_title("Slope s(τ)  [the primitive]\nPhase address varies")
ax1.set_xlabel("Proper time τ"); ax1.set_ylabel("s = Dφg")
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

# G(τ) — log scale so all worldlines are visible including far address D
ax2 = fig.add_subplot(gs[0, 1])
for name, d in results.items():
    G_plot = d["G"][TRIM:-TRIM]
    G_plot = np.where(G_plot > 0, G_plot, np.nan)  # log scale requires positive values
    ax2.plot(TAU_plot, G_plot,
             color=d["color"], ls=d["ls"], lw=d["lw"],
             label=name.split("(")[0].strip(), alpha=0.9)
ax2.set_yscale("log")
ax2.set_title("G(τ) = ∫s dτ  [log scale]\nLinear G ⟺ H = −1/2π")
ax2.set_xlabel("Proper time τ"); ax2.set_ylabel("G (log scale)")
ax2.legend(fontsize=7, loc="center right")
ax2.grid(True, alpha=0.3, which="both")

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

ax4 = fig.add_subplot(gs[1, 0])
ax4.axhline(-TARGET, color="gold", lw=2, ls="--",
            label=f"−1/2π = {-TARGET:.4f}", zorder=5)
late = int(0.6 * STEPS)
for name, d in results.items():
    H_late = d["H"][late:-TRIM]
    ax4.plot(TAU[late:-TRIM], np.clip(H_late, -0.5, 0.0),
             color=d["color"], ls=d["ls"], lw=d["lw"], alpha=0.85)
ax4.set_title("H scalar — late epoch\nAll s∞=2π cases converge exactly")
ax4.set_xlabel("Proper time τ"); ax4.set_ylabel("H")
ax4.set_ylim(-0.35, -0.05)
ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

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

# CHSH W-plot: S vs signed p ∈ [−1,1]
# Derived from P(|1⟩) = ½(1 + sin(2π|p|)) substituted into CHSH visibility:
#   S(p) = 1 + √2 + (√2 − 1) · sin(2π|p|)
# Oscillates between 2 and 2√2. Average over full cycle = 1 + √2 ≈ 2.4142.
# Crosses 1 + √2 exactly four times per cycle.
ax6 = fig.add_subplot(gs[1, 2])
p_th = np.linspace(-1, 1, 800)
S_th = 1 + np.sqrt(2) + (np.sqrt(2) - 1) * np.sin(2 * np.pi * np.abs(p_th))
ax6.plot(p_th, S_th, "k--", lw=2.5, label="gPCT: S vs signed p", zorder=5)
ax6.fill_between(p_th, S_th, 2, where=(S_th > 2), color="purple", alpha=0.15)
ax6.axhline(2 * np.sqrt(2), color="gray", lw=1, ls=":",
            label=f"Tsirelson = 2√2 ≈ {2*np.sqrt(2):.3f}")
ax6.axhline(1 + np.sqrt(2), color="green", lw=1, ls=":",
            label=f"Cycle average = 1+√2 ≈ {1+np.sqrt(2):.4f}")
ax6.axhline(2.0, color="silver", lw=1, ls=":", label="Classical bound = 2")
ax6.set_title("CHSH S vs signed slope p\nW-shape: experimental signature")
ax6.set_xlabel("Signed normalised slope p ∈ [−1,1]"); ax6.set_ylabel("S")
ax6.set_ylim(1.8, 2.95)
ax6.legend(fontsize=7, loc="lower center")
ax6.grid(True, alpha=0.3)

plt.savefig("gPCT_v5_SIMULATOR.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: gPCT_v5_SIMULATOR.png")
