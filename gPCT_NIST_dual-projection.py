"""
Dual-projection analysis of NIST bitdelay=0 runs with orthogonal residualization.
Shows:
  - signed p -> linear (odd) dependence
  - folded ŝ = 2|p| -> even-mode / W curvature
  - residualized even projection (isolating W-profile after linear removal)

Data:
  https://doi.org/10.18434/M33C71

  The NIST 2015 release contains multiple runs taken at different
  electronic-delay settings (the "bitdelay" parameter). Only the five
  bitdelay = 0 runs are used here. Non-zero bitdelay introduces a fixed
  offset between the setting-choice timestamp and the measurement
  timestamp; over a per-block average this offset smears any
  phase-dependent signal across adjacent values of the conditioning
  proxy, washing out the shape being tested. The bitdelay = 0 files
  preserve the timestamp alignment the analysis requires.

Author:  Christopher Dean White
ORCID:   0009-0002-7866-7078
License: ©2026 MIT
"""

import os
import numpy as np
import h5py
import ephem
from datetime import datetime, timezone
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------- configuration ----------
RUN_FILES = [
    "17_04_CH_pockel_100kHz.run.completeblind.dat.compressed.build.hdf5",
    "19_45_CH_pockel_100kHz.run.nolightconeshift.dat.compressed.build.hdf5",
    "23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.dat.compressed.build.hdf5",
    "00_25_CH_pockel_100kHz.run.ClassicalRNGXOR_2.dat.compressed.build.hdf5",
    "02_31_CH_pockel_100kHz.run.ClassicalRNGXOR_3.dat.compressed.build.hdf5",
]
RUN_STARTS = {
    "17_04_CH_pockel_100kHz.run.completeblind.dat.compressed.build.hdf5":
        datetime(2015, 9, 18, 23, 4, 0, tzinfo=timezone.utc),
    "19_45_CH_pockel_100kHz.run.nolightconeshift.dat.compressed.build.hdf5":
        datetime(2015, 9, 19, 1, 45, 0, tzinfo=timezone.utc),
    "23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.dat.compressed.build.hdf5":
        datetime(2015, 9, 19, 5, 55, 0, tzinfo=timezone.utc),
    "00_25_CH_pockel_100kHz.run.ClassicalRNGXOR_2.dat.compressed.build.hdf5":
        datetime(2015, 9, 19, 6, 25, 0, tzinfo=timezone.utc),
    "02_31_CH_pockel_100kHz.run.ClassicalRNGXOR_3.dat.compressed.build.hdf5":
        datetime(2015, 9, 19, 8, 31, 0, tzinfo=timezone.utc),
}
CLICK_MASK = 960
SYNC_HZ = 100_000
SCALE = 120
N_BINS = 8
HOLD_FRAC = 0.25
N_REPEATS = 30

BOULDER = ephem.Observer()
BOULDER.lat, BOULDER.lon, BOULDER.elevation = "40.0150", "-105.2705", 1655
# ------------------------------------

def compute_J_blocks(filepath, block_seconds):
    with h5py.File(filepath, "r") as f:
        a_clicks = np.asarray(f["alice/clicks"][:]) & CLICK_MASK
        b_clicks = np.asarray(f["bob/clicks"][:]) & CLICK_MASK
        a_set = np.asarray(f["alice/settings"][:])
        b_set = np.asarray(f["bob/settings"][:])
    n = min(len(a_clicks), len(b_clicks), len(a_set), len(b_set))
    block_ticks = int(block_seconds * SYNC_HZ)
    n_blocks = n // block_ticks
    J_list = []
    for b in range(n_blocks):
        sl = slice(b * block_ticks, (b + 1) * block_ticks)
        ac, bc = a_clicks[sl], b_clicks[sl]
        aset, bset = a_set[sl], b_set[sl]
        def count(sA, sB):
            return np.sum((aset == sA) & (bset == sB) & (ac > 0) & (bc > 0))
        n11, n12 = count(1, 1), count(1, 2)
        n21, n22 = count(2, 1), count(2, 2)
        s1 = np.sum((aset == 1) & (ac > 0))
        s2 = np.sum((aset == 2) & (ac > 0))
        J = (n11 + n22 - n12 - n21) - 0.5 * (s1 + s2)
        J_list.append(float(J))
    J_arr = np.array(J_list)
    basename = os.path.basename(filepath)
    start_unix = RUN_STARTS[basename].timestamp()
    times = start_unix + np.arange(len(J_arr)) * block_seconds
    return J_arr, times

def lunar_potential_rate(unix_times):
    rates = []
    moon = ephem.Moon()
    for t in unix_times:
        BOULDER.date = ephem.Date(datetime.fromtimestamp(t, timezone.utc))
        moon.compute(BOULDER)
        r1 = moon.earth_distance
        BOULDER.date += 60 / 86400.0
        moon.compute(BOULDER)
        r2 = moon.earth_distance
        rdot = (r2 - r1) / 60.0
        r_mid = 0.5 * (r1 + r2)
        rates.append(-rdot / (r_mid ** 2))
    return np.array(rates)

def normalize_and_fold(s):
    ptp = np.ptp(s)
    p = np.zeros_like(s) if ptp < 1e-12 else 2 * (s - s.min()) / ptp - 1
    return p, 2 * np.abs(p)

def gpct_shape(s_hat):
    return np.cos(np.pi * np.clip(s_hat, 0, 2) / 2 - np.pi / 4) ** 2

def bin_means(x, y, n_bins=N_BINS):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers, means, ses = [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x <= edges[i+1] if i == n_bins-1 else x < edges[i+1])
        if np.sum(m) >= 2:
            centers.append(np.mean(x[m]))
            means.append(np.mean(y[m]))
            ses.append(np.std(y[m]) / np.sqrt(np.sum(m)))
    return np.array(centers), np.array(means), np.array(ses)

def lin_folded(x, a, b):
    return a + b * x

def templ(x, a, b):
    return a + b * (gpct_shape(x) - np.mean(gpct_shape(x)))

# ---------- load continuous arc ----------
all_J, all_s = [], []
for fn in RUN_FILES:
    if not os.path.exists(fn):
        continue
    J, t = compute_J_blocks(fn, SCALE)
    if len(J):
        all_J.append(J)
        all_s.append(lunar_potential_rate(t))
J = np.concatenate(all_J)
s = np.concatenate(all_s)
p, s_hat = normalize_and_fold(s)
n = len(J)
print(f"Total blocks at {SCALE}s: {n}")

# ---------- check cross-projection orthogonality ----------
ortho_corr = np.corrcoef(p, s_hat)[0, 1]
print(f"Cross-projection correlation (p vs ŝ): {ortho_corr:.4f}")

# ---------- signed (odd) projection ----------
r_signed = np.corrcoef(p, J)[0, 1]
coef_lin = np.polyfit(p, J, 1)
print(f"Signed p: r = {r_signed:.3f}")

# ---------- folded (even) projection ----------
cen, mu, se = bin_means(s_hat, J)
try:
    popt_l, pcov_l = curve_fit(lin_folded, cen, mu, p0=[mu.mean(), 0], sigma=se)
    popt_t, pcov_t = curve_fit(templ, cen, mu, p0=[mu.mean(), 0], sigma=se)

    rss_l = np.sum(((mu - lin_folded(cen, *popt_l)) / se)**2)
    rss_t = np.sum(((mu - templ(cen, *popt_t)) / se)**2)
    contrast = (rss_l - rss_t) / (rss_t + 1e-12)

    a_fit, b_fit = popt_t
    a_err, b_err = np.sqrt(np.diag(pcov_t))
    print(f"\n--- Folded Fit Parameters (Scale: {SCALE}s) ---")
    print(f"Curvature Contrast : {contrast:.3f}")
    print(f"Vertical Offset (a): {a_fit:.2f} ± {a_err:.2f}")
    print(f"W-Amplitude (b)    : {b_fit:.2f} ± {b_err:.2f}")
    print("-" * 45)
except Exception as e:
    print("Folded fit failed:", e)
    contrast = np.nan
    popt_l = popt_t = None

# ---------- orthogonal residualization ----------
J_linear_trend = np.polyval(coef_lin, p)
J_residuals = J - J_linear_trend
cen_res, mu_res, se_res = bin_means(s_hat, J_residuals)
try:
    popt_res, pcov_res = curve_fit(templ, cen_res, mu_res, p0=[mu_res.mean(), 0], sigma=se_res)
    b_res, b_res_err = popt_res[1], np.sqrt(np.diag(pcov_res))[1]
    print(f"Residual W-Amplitude (b_res): {b_res:.2f} ± {b_res_err:.2f}")
except Exception as e:
    print("Residual fit failed:", e)
    popt_res = None

# ---------- continuity-preserving hold-out on folded ----------
rng = np.random.default_rng(42)
test_contrasts = []
for _ in range(N_REPEATS):
    hold = rng.random(n) < HOLD_FRAC
    if np.sum(hold) < N_BINS * 2 or np.sum(~hold) < N_BINS * 2:
        continue
    c_test = 0.0
    try:
        cen_h, mu_h, se_h = bin_means(s_hat[hold], J[hold])
        pl, _ = curve_fit(lin_folded, cen_h, mu_h, p0=[mu_h.mean(), 0], sigma=se_h)
        pt, _ = curve_fit(templ, cen_h, mu_h, p0=[mu_h.mean(), 0], sigma=se_h)
        rssl = np.sum(((mu_h - lin_folded(cen_h, *pl)) / se_h)**2)
        rsst = np.sum(((mu_h - templ(cen_h, *pt)) / se_h)**2)
        c_test = (rssl - rsst) / (rsst + 1e-12)
    except Exception:
        pass
    test_contrasts.append(c_test)
test_contrasts = np.array(test_contrasts)
print(f"Hold-out test contrast: mean={test_contrasts.mean():.3f}, "
      f"fraction>0={np.mean(test_contrasts>0):.2f}")

# ---------- 3-panel figure ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Signed Odd Projection
axes[0].scatter(p, J, s=12, alpha=0.6, c="k")
xx = np.linspace(p.min(), p.max(), 100)
axes[0].plot(xx, np.polyval(coef_lin, xx), "r-", lw=2, label=f"linear r={r_signed:.2f}")
axes[0].set_xlabel("Signed normalised slope p")
axes[0].set_ylabel("Block-averaged J")
axes[0].set_title("Odd projection (signed p)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panel 2: Folded Even Projection
if popt_t is not None:
    axes[1].errorbar(cen, mu, yerr=se, fmt="o", color="k", ms=6, capsize=3, label="binned data")
    xf = np.linspace(0, 2, 200)
    axes[1].plot(xf, lin_folded(xf, *popt_l), "b--", label="linear in ŝ")
    axes[1].plot(xf, templ(xf, *popt_t), "r-", lw=2, label=f"gPCT cos² (contrast={contrast:.2f})")
    axes[1].set_xlabel("Folded slope ŝ = 2|p|")
    axes[1].set_ylabel("Mean J")
    axes[1].set_title("Even projection (folded ŝ)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

# Panel 3: Residualized Even Projection
if popt_res is not None:
    axes[2].errorbar(cen_res, mu_res, yerr=se_res, fmt="o", color="k", ms=6, capsize=3, label="residual data")
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].plot(xf, popt_res[1] * (gpct_shape(xf) - np.mean(gpct_shape(xf))), "r-", lw=2, label=f"W-template (b={b_res:.2f})")
    axes[2].set_xlabel("Folded slope ŝ = 2|p|")
    axes[2].set_ylabel("Residual J (linear trend removed)")
    axes[2].set_title("Orthogonal residualized even mode")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gPCT_NIST_dual-projection_residualized.png", dpi=150)
print("\nSaved gPCT_NIST_dual-projection_residualized.png")
plt.show()
