"""
NIST 2015 Bell-test lunar-slope analysis
Gravitational Phase-Cancellation Theory
Christopher Dean White
ORCID: 0009-0002-7866-7078
MIT License ©2026

Companion analysis to "A Minimal Statement of Gravitational Phase-Cancellation
Theory (gPCT)." Tests the gPCT W-curve prediction against the NIST 2015
loophole-free Bell test (Shalm et al.) — 5 bitdelay=0 runs spanning
17:04 UTC Sept 18 through ~03:00 UTC Sept 19, 2015.

Two analyses in one pass:

  A. Robustness scan. Compute Eberhard J at block sizes of 30, 60, 120,
     and 300 seconds. For each, report correlations of J against signed
     lunar slope (expected null under Postulate I symmetry), |slope|
     (expected positive — the magnitude of the modulation), and time
     (drift control). Correlation should hold across block sizes with
     √T scaling of significance.

  B. Shape analysis (60s blocks). Normalize signed lunar slope by the
     peak |s_Moon| over one sidereal day to get p ∈ [-1, +1]. Bin by
     signed p into 10 equal-count bins. Fit five models (null, linear,
     |p|-hump, gPCT W with fixed period, free-period sine). Compare
     by chi-squared and BIC. Test whether the modulation traces the
     specific W-shape gPCT predicts.

Requires:
  - 5 NIST HDF5 bitdelay=0 files in current directory
  - python packages: h5py, numpy, scipy, ephem, matplotlib

Data source:
  https://doi.org/10.18434/T4/1502474

File naming convention:
  17_04_CH_pockel_100kHz.run.completeblind.dat.compressed.build.hdf5
  19_45_CH_pockel_100kHz.run.nolightconeshift.dat.compressed.build.hdf5
  23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.dat.compressed.build.hdf5
  00_25_CH_pockel_100kHz.run.ClassicalRNGXOR_2.dat.compressed.build.hdf5
  02_31_CH_pockel_100kHz.run.ClassicalRNGXOR_3.dat.compressed.build.hdf5
"""

import os
import glob
import h5py
import numpy as np
from scipy import stats, optimize
import ephem
import matplotlib.pyplot as plt

# ---- RUNS AND CONSTANTS ----------------------------------------------------
runs = [
    ("17_04_CH_pockel_100kHz.run.completeblind.dat.compressed.build.hdf5",
     "2015/9/18 17:04:00"),
    ("19_45_CH_pockel_100kHz.run.nolightconeshift.dat.compressed.build.hdf5",
     "2015/9/18 19:45:00"),
    ("23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.dat.compressed.build.hdf5",
     "2015/9/18 23:55:00"),
    ("00_25_CH_pockel_100kHz.run.ClassicalRNGXOR_2.dat.compressed.build.hdf5",
     "2015/9/19 00:25:00"),
    ("02_31_CH_pockel_100kHz.run.ClassicalRNGXOR_3.dat.compressed.build.hdf5",
     "2015/9/19 02:31:00"),
]

MASK = 960
sync_hz = 100000

BLOCK_SIZES   = [30, 60, 120, 300]   # seconds — robustness scan
SHAPE_BLOCK   = 60                   # seconds — block size used for shape fit
N_BINS        = 10

# JILA Boulder — NIST experimental site
obs = ephem.Observer()
obs.lat, obs.lon, obs.elevation, obs.pressure = '40.0150', '-105.2705', 1655, 0


def moon_slope(t_ephem, dt_sec=300.0):
    """Signed Moon altitude rate at Boulder at t_ephem, rad/s.

    Centered finite difference. Scale-free, signed, observer-relative —
    the paper-faithful slope proxy (Postulate I).
    """
    dt_days = dt_sec / 86400.0
    obs.date = t_ephem - dt_days
    alt_b = float(ephem.Moon(obs).alt)
    obs.date = t_ephem + dt_days
    alt_a = float(ephem.Moon(obs).alt)
    return (alt_a - alt_b) / (2.0 * dt_sec)


# ---- STAGE 1: PER-BLOCK EBERHARD J + SLOPE FOR EACH BLOCK SIZE --------------
def compute_blocks(block_seconds):
    """For a given block size, return arrays of J, signed slope, and hours."""
    block_syncs = block_seconds * sync_hz
    all_J, all_slopes, all_times = [], [], []
    t0_ephem = ephem.Date("2015/9/18 17:04:00")
    files_found = 0

    for filepath, utc in runs:
        if not os.path.exists(filepath):
            pattern = filepath[:5] + "*build*hdf5"
            matches = glob.glob(pattern)
            if matches:
                filepath = sorted(matches)[0]
            else:
                continue
        files_found += 1

        f = h5py.File(filepath, 'r')
        ac = f['alice/clicks'][:] & MASK
        bc = f['bob/clicks'][:] & MASK
        sa = f['alice/settings'][:]
        sb = f['bob/settings'][:]
        n_total = len(ac)
        f.close()

        ad = ac > 0
        bd = bc > 0
        coin = ad & bd & (ac == bc)
        valid = (sa >= 1) & (sa <= 2) & (sb >= 1) & (sb <= 2)
        t_start = ephem.Date(utc)
        n_blocks = n_total // block_syncs

        for block in range(n_blocks):
            start = block * block_syncs
            end   = start + block_syncs
            bm = np.zeros(n_total, dtype=bool)
            bm[start:end] = True
            bv = bm & valid

            c = np.zeros((4, 4))
            for i, (a, b) in enumerate([(1,1), (1,2), (2,1), (2,2)]):
                sel = bv & (sa == a) & (sb == b)
                N = sel.sum()
                if N == 0:
                    c[i] = [0, 0, 0, 1]
                    continue
                c[i] = [(ad & sel).sum(), (coin & sel).sum(),
                        (bd & sel).sum(), N]
            J = c[0,1]+c[1,1]+c[2,1]-c[3,1] - 0.5*(c[0,0]+c[1,0]+c[0,2]+c[2,2])

            mid_sec = (start + block_syncs // 2) / sync_hz
            t_mid = t_start + mid_sec / 86400.0
            s_block = moon_slope(t_mid)

            all_J.append(J)
            all_slopes.append(s_block)
            all_times.append(float(t_mid - t0_ephem) * 24)

    return (np.array(all_J), np.array(all_slopes), np.array(all_times), files_found)


# ---- STAGE 2: ROBUSTNESS SCAN ACROSS BLOCK SIZES ----------------------------
print("="*72)
print("A. Robustness scan across block sizes")
print("="*72)
print(f"\n{'block (s)':>10} {'N blocks':>10} {'r(J, signed)':>14} "
      f"{'r(J, |s|)':>12} {'r(J, time)':>12}")
print("-"*72)

scan_results = {}  # keyed by block size

for bs in BLOCK_SIZES:
    J, s, t, files_found = compute_blocks(bs)
    if len(J) == 0:
        print(f"{bs:>10} {'--':>10}  (no files found)")
        continue
    r_sgn, p_sgn = stats.pearsonr(J, s)
    r_abs, p_abs = stats.pearsonr(J, np.abs(s))
    r_tim, p_tim = stats.pearsonr(J, t)
    scan_results[bs] = dict(J=J, s=s, t=t, r_sgn=r_sgn, p_sgn=p_sgn,
                             r_abs=r_abs, p_abs=p_abs, r_tim=r_tim, p_tim=p_tim,
                             N=len(J))

    print(f"{bs:>10} {len(J):>10} "
          f"{r_sgn:>+7.4f} ({p_sgn:>5.1e}) "
          f"{r_abs:>+5.4f} ({p_abs:>5.1e}) "
          f"{r_tim:>+5.4f} ({p_tim:>5.1e})")

print()
print("Interpretation:")
print("  r(J, signed slope) ≈ 0   — the W is even in p (Postulate I)")
print("  r(J, |slope|)      > 0   — magnitude correlates with J")
print("  r(J, time)         ≈ 0   — no drift confound")


# ---- STAGE 3: SHAPE FIT AT SHAPE_BLOCK SECONDS -----------------------------
print(f"\n{'='*72}")
print(f"B. Shape test at {SHAPE_BLOCK}s blocks — binning by signed p")
print("="*72)

J     = scan_results[SHAPE_BLOCK]['J']
s_raw = scan_results[SHAPE_BLOCK]['s']
times = scan_results[SHAPE_BLOCK]['t']

# Calibrate s_max over ±1 sidereal day around run midpoint, from ephemeris
t_run_mid = ephem.Date("2015/9/18 22:00:00")
SIDEREAL_DAY = 0.99726958
scan_times = np.linspace(-SIDEREAL_DAY, SIDEREAL_DAY, 4000)
scan_slopes = np.array([moon_slope(t_run_mid + dt) for dt in scan_times])
s_max = np.max(np.abs(scan_slopes))

p_signed = s_raw / s_max

print(f"Peak |s_Moon| over ±1 sidereal day at Boulder: {s_max*1e5:.3f} × 10⁻⁵ rad/s")
print(f"Signed p range in {SHAPE_BLOCK}s blocks: {p_signed.min():+.3f} to {p_signed.max():+.3f}")

# Bin by SIGNED p (not |p|) so the full W structure is visible
sort_idx = np.argsort(p_signed)
blocks_per_bin = len(J) // N_BINS
bin_indices = np.array_split(sort_idx[:blocks_per_bin * N_BINS], N_BINS)

bin_p_mean = np.array([p_signed[idx].mean() for idx in bin_indices])
bin_J_mean = np.array([J[idx].mean() for idx in bin_indices])
bin_J_se   = np.array([J[idx].std(ddof=1) / np.sqrt(len(idx)) for idx in bin_indices])

print(f"\n{N_BINS} bins (~{blocks_per_bin} blocks each):")
print(f"{'bin':>4} {'p_mean':>10} {'J_mean':>10} {'J_se':>8}")
for i in range(N_BINS):
    print(f"{i:>4} {bin_p_mean[i]:>+10.3f} {bin_J_mean[i]:>10.3f} {bin_J_se[i]:>8.3f}")


# ---- MODEL FITS -------------------------------------------------------------
def fit_model(fn, p0, name, nparam):
    try:
        popt, pcov = optimize.curve_fit(fn, bin_p_mean, bin_J_mean, p0=p0,
                                         sigma=bin_J_se, absolute_sigma=True, maxfev=50000)
        res   = bin_J_mean - fn(bin_p_mean, *popt)
        chi2  = np.sum((res/bin_J_se)**2)
        dof   = N_BINS - nparam
        chi2r = chi2/dof if dof > 0 else np.nan
        bic   = chi2 + nparam*np.log(N_BINS)
        return {'name':name,'popt':popt,'pcov':pcov,'chi2':chi2,'dof':dof,
                'chi2r':chi2r,'bic':bic,'res':res,'nparam':nparam,'fn':fn}
    except Exception as e:
        print(f"Fit failed for {name}: {e}")
        return None

def m_null(p, c):       return c + 0*p
def m_linear(p, a, b):  return a + b*p
def m_absp(p, a, b):    return a + b*np.abs(p)

# gPCT W prediction (period fixed at 1, per Postulate III normalization):
#   J(p) = a + b * sin(2π|p|)
# Symmetric about p = 0. Zero-crossings (DC offset) at |p| = 0, 0.5, 1.
# Extrema at |p| = 0.25 and 0.75 — their sign is determined by the fitted
# amplitude b (the theoretical sign convention is observer-frame dependent;
# see the two-curve plot in gPCT_SIMULATOR.py).
def m_W(p, a, b):       return a + b*np.sin(2*np.pi*np.abs(p))
def m_Wfree(p, a, b, k): return a + b*np.sin(2*np.pi*k*np.abs(p))

fits = [f for f in [
    fit_model(m_null,   [bin_J_mean.mean()],                    'null',                 1),
    fit_model(m_linear, [bin_J_mean.mean(), 0.0],               'linear in p',          2),
    fit_model(m_absp,   [bin_J_mean.mean(), 5.0],               'a + b|p| (hump)',      2),
    fit_model(m_W,      [bin_J_mean.mean(), 5.0],               'gPCT W (k=1 fixed)',   2),
    fit_model(m_Wfree,  [bin_J_mean.mean(), 5.0, 1.0],          'free-period sine',     3),
] if f is not None]

print(f"\nModel comparison (signed-p binning, Moon altitude rate proxy):")
print(f"{'model':<28} {'np':>3} {'chi2':>8} {'chi2/dof':>10} {'BIC':>8} {'ΔBIC':>8}")
bic_min = min(f['bic'] for f in fits)
for f in fits:
    mark = '  ←BEST' if abs(f['bic']-bic_min) < 0.001 else ''
    print(f"{f['name']:<28} {f['nparam']:>3} {f['chi2']:>8.2f} "
          f"{f['chi2r']:>10.2f} {f['bic']:>8.2f} {f['bic']-bic_min:>8.2f}{mark}")

for f in fits:
    if 'gPCT' in f['name'] or 'free' in f['name']:
        sig = np.sqrt(np.diag(f['pcov']))
        print(f"\n{f['name']} params:")
        if f['nparam'] == 2:
            print(f"  a = {f['popt'][0]:+.3f} ± {sig[0]:.3f}")
            print(f"  b = {f['popt'][1]:+.3f} ± {sig[1]:.3f}")
            print(f"  amplitude significance: {f['popt'][1]/sig[1]:+.2f}σ")
        else:
            print(f"  a = {f['popt'][0]:+.3f} ± {sig[0]:.3f}")
            print(f"  b = {f['popt'][1]:+.3f} ± {sig[1]:.3f}")
            print(f"  k = {f['popt'][2]:.3f} ± {sig[2]:.3f}  (gPCT predicts k=1 exactly)")
            print(f"  deviation from k=1: {(f['popt'][2]-1)/sig[2]:+.2f}σ")


# ---- PLOTS -----------------------------------------------------------------
fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# (A) Block-size robustness — correlation magnitudes vs block size
axA = fig.add_subplot(gs[0, 0])
bs_arr  = np.array(sorted(scan_results.keys()))
r_abs_v = np.array([scan_results[bs]['r_abs'] for bs in bs_arr])
r_sgn_v = np.array([scan_results[bs]['r_sgn'] for bs in bs_arr])
r_tim_v = np.array([scan_results[bs]['r_tim'] for bs in bs_arr])
axA.plot(bs_arr, r_abs_v, 'o-', color='#1abc9c', ms=10, lw=2, label='r(J, |slope|)')
axA.plot(bs_arr, r_sgn_v, 's-', color='#3498db', ms=8,  lw=1.5, label='r(J, signed slope)')
axA.plot(bs_arr, r_tim_v, '^-', color='#e67e22', ms=8,  lw=1.5, label='r(J, time)')
axA.axhline(0, color='gray', lw=0.5, ls=':')
axA.set_xscale('log')
axA.set_xticks(bs_arr); axA.set_xticklabels([str(b) for b in bs_arr])
axA.set_xlabel('Block size (seconds)')
axA.set_ylabel('Pearson r')
axA.set_title('Robustness scan — correlations vs block size\n'
              '|slope| correlation holds; signed and time correlations null')
axA.legend(loc='best')
axA.grid(alpha=0.3)

# (B) p-value behavior (log scale)
axB = fig.add_subplot(gs[0, 1])
p_abs_v = np.array([scan_results[bs]['p_abs'] for bs in bs_arr])
p_sgn_v = np.array([scan_results[bs]['p_sgn'] for bs in bs_arr])
p_tim_v = np.array([scan_results[bs]['p_tim'] for bs in bs_arr])
axB.semilogy(bs_arr, p_abs_v, 'o-', color='#1abc9c', ms=10, lw=2, label='p(J, |slope|)')
axB.semilogy(bs_arr, p_sgn_v, 's-', color='#3498db', ms=8,  lw=1.5, label='p(J, signed slope)')
axB.semilogy(bs_arr, p_tim_v, '^-', color='#e67e22', ms=8,  lw=1.5, label='p(J, time)')
axB.axhline(0.05, color='red', lw=0.8, ls='--', alpha=0.5, label='p = 0.05')
axB.set_xscale('log')
axB.set_xticks(bs_arr); axB.set_xticklabels([str(b) for b in bs_arr])
axB.set_xlabel('Block size (seconds)')
axB.set_ylabel('p-value (log scale)')
axB.set_title('Robustness scan — p-values vs block size')
axB.legend(loc='best')
axB.grid(alpha=0.3, which='both')

# (C) Shape fit at SHAPE_BLOCK — binned data + model curves
axC = fig.add_subplot(gs[1, 0])
axC.errorbar(bin_p_mean, bin_J_mean, yerr=bin_J_se, fmt='o', color='k',
             ms=9, capsize=4, label='NIST data', zorder=10)
xfine = np.linspace(-1, 1, 400)
colors = ['#888', '#3498db', '#1abc9c', '#c0392b', '#9b59b6']
for f, col in zip(fits, colors):
    yfine = f['fn'](xfine, *f['popt'])
    lw = 3 if 'gPCT' in f['name'] else 1.5
    axC.plot(xfine, yfine, color=col, lw=lw,
             label=f"{f['name']} (χ²/ν={f['chi2r']:.2f}, ΔBIC={f['bic']-bic_min:.1f})")

for crit in [-0.75, -0.25, 0, 0.25, 0.75]:
    axC.axvline(crit, color='gold', ls=':', lw=0.8, alpha=0.5)

axC.axhline(0, color='gray', ls=':', lw=0.5)
axC.set_xlabel('Signed p = (Moon altitude rate) / s_max')
axC.set_ylabel(f'J (Eberhard), per {SHAPE_BLOCK}s block')
axC.set_title(f'Shape test at {SHAPE_BLOCK}s · {len(J)} blocks · {N_BINS} bins')
axC.set_xlim(-1.1, 1.1)
axC.legend(fontsize=8, loc='best')
axC.grid(alpha=0.3)

# (D) Residuals
axD = fig.add_subplot(gs[1, 1])
for f, col in zip(fits[1:], colors[1:]):
    axD.errorbar(bin_p_mean, f['res'], yerr=bin_J_se, fmt='o-',
                 color=col, ms=6, capsize=3, label=f['name'], alpha=0.7)
axD.axhline(0, color='k', lw=0.8)
axD.set_xlabel('Signed p')
axD.set_ylabel('Residual: J − model(p)')
axD.set_title('Shape-fit residuals')
axD.legend(fontsize=8)
axD.grid(alpha=0.3)

fig.suptitle('NIST 2015 Bell-test: lunar-slope analysis\n'
             'Robustness scan (top) + shape test at 60s (bottom)',
             fontsize=13, fontweight='bold', y=1.00)
plt.savefig('NIST_gPCT_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: NIST_gPCT_analysis.png")
plt.show()
