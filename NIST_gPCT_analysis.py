"""
NIST 2015 Bell-test temporal-modulation analysis

Tests for time-of-day modulation of the Eberhard J statistic in the NIST
2015 loophole-free Bell test (Shalm et al., PRL 115, 250401, 2015), using
the local Moon altitude rate at Boulder as the primary conditioning
variable, with two independent controls.

Stages:
  A. Robustness scan. Pearson r between J and (i) signed Moon slope,
     (ii) |Moon slope|, (iii) signed sin(LST), (iv) |sin(LST)|, and (v)
     time-from-run-start, computed at block sizes of 30, 60, 120, 300 s.
  B. Proxy-proxy diagnostic. Intercorrelation of the Moon and sidereal
     proxies at the shape-fit block size, plus partial correlations of
     J with each, controlling for the other.
  C. Shape fits at 60 s blocks. Normalize the Moon slope by its peak over
     a sidereal day to get signed p in [-1, +1], bin by signed p into ten
     equal-population bins, and fit five candidate shapes: null, linear,
     |p|-hump, W = a + b*sin(2*pi*|p|) with k = 1 fixed, and the same
     functional form with k free. Compare by chi-squared and BIC. Run
     the same pipeline on the sidereal proxy as a parallel control.
  D. Time-shuffle null. 10000 permutations of which J value pairs with
     which slope value. Reports empirical p-values for r(J, |Moon|),
     the W amplitude b from a bounded curve fit, and the W amplitude
     from a parameter-free projection statistic. Empirical p-values
     supplement the parametric Pearson p-values and are robust to any
     residual J autocorrelation within runs.

The W functional form sin(2*pi*|p|) is one of the candidate shapes tested
in stage C. It was chosen because it is the shape predicted by the gPCT
framework cited in the accompanying paper; the script itself makes no
theoretical claim and simply reports how well each shape fits.

Requirements:
  - 5 NIST HDF5 files in the working directory (see Data below)
  - python: h5py, numpy, scipy, ephem, matplotlib

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
License: MIT
"""

import os
import glob
import h5py
import numpy as np
from scipy import stats, optimize
import ephem
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NIST 2015 bitdelay = 0 runs. Each tuple is (filename, UTC start time).
# Only bitdelay = 0 runs are used. Non-zero bitdelay smears the per-block
# phase alignment between the setting choice and the measurement, which
# would wash out any shape-dependent signal.
RUN_FILES = [
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
RUN_START = "2015/9/18 17:04:00"   # earliest run start, used as t = 0
RUN_MIDPOINT = "2015/9/18 22:00:00"  # used to calibrate s_max over a sidereal day

# Click-mask bits used by the NIST analysis pipeline.
CLICK_MASK   = 960
SYNC_HZ      = 100000

# Block sizes for the robustness scan.
BLOCK_SIZES  = [30, 60, 120, 300]
# Block size used for the shape fit and the shuffle null.
SHAPE_BLOCK  = 60
# Number of equal-population bins in signed p.
N_BINS       = 10
# Number of permutations for the empirical null.
N_SHUFFLES   = 10000
# Sidereal day in solar days.
SIDEREAL_DAY = 0.99726958

# JILA Boulder -- NIST experimental site (WGS84).
BOULDER_LAT   = '40.0150'
BOULDER_LON   = '-105.2705'
BOULDER_ELEV  = 1655   # metres

obs = ephem.Observer()
obs.lat       = BOULDER_LAT
obs.lon       = BOULDER_LON
obs.elevation = BOULDER_ELEV
obs.pressure  = 0   # geometric positions, no refraction


# ---------------------------------------------------------------------------
# Geometric proxies
# ---------------------------------------------------------------------------

def moon_slope(t_ephem, dt_sec=300.0):
    """Signed Moon altitude rate at Boulder, in rad/s.

    Centered finite difference over +- dt_sec around t_ephem.
    """
    dt_days = dt_sec / 86400.0
    obs.date = t_ephem - dt_days
    alt_b = float(ephem.Moon(obs).alt)
    obs.date = t_ephem + dt_days
    alt_a = float(ephem.Moon(obs).alt)
    return (alt_a - alt_b) / (2.0 * dt_sec)


def sidereal_phase(t_ephem):
    """Signed sidereal phase at Boulder: sin(LST), dimensionless in [-1, +1].

    Pure-rotation proxy: no Moon, no Sun, no gravitating body. Tracks
    Boulder's spin only. Used as a control: if a J modulation is driven
    purely by apparatus coupling to Boulder's rotation, this proxy should
    pick it up. If the modulation is Moon-specific, this proxy should miss
    most of it.
    """
    obs.date = t_ephem
    lst_rad = float(obs.sidereal_time())   # radians, [0, 2*pi)
    return np.sin(lst_rad)


# ---------------------------------------------------------------------------
# Per-block J and per-block proxy values
# ---------------------------------------------------------------------------

def compute_blocks(block_seconds):
    """For a given block size, return per-block:
      - Eberhard J
      - signed Moon slope (rad/s)
      - signed sidereal proxy in [-1, +1]
      - mid-block time, in hours from the first run start
    All arrays are parallel.
    """
    block_syncs = block_seconds * SYNC_HZ
    all_J, all_s_moon, all_s_sid, all_times = [], [], [], []
    t0_ephem = ephem.Date(RUN_START)
    files_found = 0

    for filepath, utc in RUN_FILES:
        if not os.path.exists(filepath):
            pattern = filepath[:5] + "*build*hdf5"
            matches = glob.glob(pattern)
            if matches:
                filepath = sorted(matches)[0]
            else:
                continue
        files_found += 1

        f = h5py.File(filepath, 'r')
        ac = f['alice/clicks'][:] & CLICK_MASK
        bc = f['bob/clicks'][:]   & CLICK_MASK
        sa = f['alice/settings'][:]
        sb = f['bob/settings'][:]
        n_total = len(ac)
        f.close()

        ad    = ac > 0
        bd    = bc > 0
        coin  = ad & bd & (ac == bc)
        valid = (sa >= 1) & (sa <= 2) & (sb >= 1) & (sb <= 2)
        t_start  = ephem.Date(utc)
        n_blocks = n_total // block_syncs

        for block in range(n_blocks):
            start = block * block_syncs
            end   = start + block_syncs
            bm = np.zeros(n_total, dtype=bool)
            bm[start:end] = True
            bv = bm & valid

            # Eberhard J statistic: count clicks by setting combination.
            c = np.zeros((4, 4))
            for i, (a, b) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)]):
                sel = bv & (sa == a) & (sb == b)
                N = sel.sum()
                if N == 0:
                    c[i] = [0, 0, 0, 1]
                    continue
                c[i] = [(ad & sel).sum(), (coin & sel).sum(),
                        (bd & sel).sum(), N]
            J = (c[0, 1] + c[1, 1] + c[2, 1] - c[3, 1]
                 - 0.5 * (c[0, 0] + c[1, 0] + c[0, 2] + c[2, 2]))

            mid_sec = (start + block_syncs // 2) / SYNC_HZ
            t_mid   = t_start + mid_sec / 86400.0
            s_moon  = moon_slope(t_mid)
            s_sid   = sidereal_phase(t_mid)

            all_J.append(J)
            all_s_moon.append(s_moon)
            all_s_sid.append(s_sid)
            all_times.append(float(t_mid - t0_ephem) * 24)

    return (np.array(all_J),
            np.array(all_s_moon),
            np.array(all_s_sid),
            np.array(all_times),
            files_found)


# ---------------------------------------------------------------------------
# Shape fitting
# ---------------------------------------------------------------------------

# Candidate shapes for J(p). The W shape -- even-in-p sine with fixed unit
# period -- is the one whose detection is the primary question; the others
# are reference alternatives.
def m_null(p, c):           return c + 0 * p
def m_linear(p, a, b):      return a + b * p
def m_absp(p, a, b):        return a + b * np.abs(p)
def m_W(p, a, b):           return a + b * np.sin(2 * np.pi * np.abs(p))
def m_Wfree(p, a, b, k):    return a + b * np.sin(2 * np.pi * k * np.abs(p))


def w_projection(bin_J, bin_p):
    """Parameter-free amplitude estimator: project binned J means onto the
    W template sin(2*pi*|p|) centred at zero. Returns the coefficient that
    minimizes |J - a - b * template| in least squares. Equivalent to the
    b coefficient from a linear regression of (J - mean(J)) on the
    template; computed by linear algebra so no nonlinear optimizer is
    involved, and the result cannot diverge on noisy inputs.
    """
    template = np.sin(2 * np.pi * np.abs(bin_p))
    template_centered = template - template.mean()
    J_centered = bin_J - bin_J.mean()
    denom = (template_centered ** 2).sum()
    if denom < 1e-12:
        return np.nan
    return (J_centered * template_centered).sum() / denom


def run_shape_fit(p_arr, J_arr, label):
    """Bin (p, J) into N_BINS equal-population signed-p bins, fit five
    candidate shapes, print a comparison table, and return a result dict.
    """
    sort_idx = np.argsort(p_arr)
    bpb = len(J_arr) // N_BINS
    bin_idx = np.array_split(sort_idx[:bpb * N_BINS], N_BINS)
    bin_p   = np.array([p_arr[i].mean() for i in bin_idx])
    bin_J   = np.array([J_arr[i].mean() for i in bin_idx])
    bin_se  = np.array([J_arr[i].std(ddof=1) / np.sqrt(len(i)) for i in bin_idx])

    def fit(fn, p0, name, nparam):
        try:
            popt, pcov = optimize.curve_fit(
                fn, bin_p, bin_J, p0=p0,
                sigma=bin_se, absolute_sigma=True, maxfev=50000)
            res   = bin_J - fn(bin_p, *popt)
            chi2  = float(np.sum((res / bin_se) ** 2))
            dof   = N_BINS - nparam
            chi2r = chi2 / dof if dof > 0 else np.nan
            bic   = chi2 + nparam * np.log(N_BINS)
            return {'name': name, 'popt': popt, 'pcov': pcov,
                    'chi2': chi2, 'dof': dof, 'chi2r': chi2r,
                    'bic': bic, 'res': res, 'nparam': nparam, 'fn': fn}
        except Exception as e:
            print(f"  Fit failed: {name}: {e}")
            return None

    fits = [f for f in [
        fit(m_null,   [bin_J.mean()],                'null',                1),
        fit(m_linear, [bin_J.mean(), 0.0],           'linear in p',         2),
        fit(m_absp,   [bin_J.mean(), 5.0],           'a + b|p| (hump)',     2),
        fit(m_W,      [bin_J.mean(), 5.0],           'W (k=1 fixed)',       2),
        fit(m_Wfree,  [bin_J.mean(), 5.0, 1.0],      'W (k free)',          3),
    ] if f is not None]

    bic_min = min(f['bic'] for f in fits)
    print(f"\n  --- {label} ---")
    print(f"  bin populations: {[len(i) for i in bin_idx]}")
    print(f"  {'model':<24} {'chi2':>8} {'chi2/dof':>10} "
          f"{'BIC':>8} {'dBIC':>8}")
    for f in fits:
        mark = '  <- best' if abs(f['bic'] - bic_min) < 0.001 else ''
        print(f"  {f['name']:<24} {f['chi2']:>8.2f} "
              f"{f['chi2r']:>10.2f} {f['bic']:>8.2f} "
              f"{f['bic'] - bic_min:>8.2f}{mark}")

    # Report W amplitude.
    for f in fits:
        if f['name'] == 'W (k=1 fixed)':
            sig = np.sqrt(np.diag(f['pcov']))
            amp_sig = f['popt'][1] / sig[1]
            print(f"  W amplitude b = {f['popt'][1]:+.3f} "
                  f"+- {sig[1]:.3f}  ({amp_sig:+.2f} sigma)")
            break

    return dict(bin_p=bin_p, bin_J=bin_J, bin_se=bin_se,
                fits=fits, bic_min=bic_min)


# ---------------------------------------------------------------------------
# Shuffle test (parametric and parameter-free shape statistics)
# ---------------------------------------------------------------------------

def shuffle_test(J_arr, p_arr, n_shuffles, seed=12345,
                 b_bounds=(-200.0, 200.0)):
    """Permute J vs p pairing. For each shuffle, compute:
      - Pearson r(J, |p|)
      - W amplitude b via a bounded curve fit; reject and return NaN if the
        fit hits the parameter bound or if the standard error on b exceeds
        a quarter of the bound width (both indicate the optimizer did not
        find an interior minimum and the result should not be counted)
      - W amplitude b via the parameter-free projection statistic
    Returns three arrays of shuffled-statistic values, in that order.

    The shuffle preserves the empirical distributions of J and of p
    separately; only their pairing is randomized. The resulting null is
    robust to any per-run J distribution shape or J autocorrelation
    within blocks.
    """
    rng   = np.random.default_rng(seed)
    abs_p = np.abs(p_arr)

    sh_r       = np.empty(n_shuffles)
    sh_b_fit   = np.empty(n_shuffles)
    sh_b_proj  = np.empty(n_shuffles)

    # Bin assignments are based on the actual p; only the J values get
    # permuted on each shuffle. Pre-computing the bin index lists keeps
    # the inner loop fast.
    sort_idx    = np.argsort(p_arr)
    bpb         = len(J_arr) // N_BINS
    bin_idx     = np.array_split(sort_idx[:bpb * N_BINS], N_BINS)
    bin_p_fixed = np.array([p_arr[i].mean() for i in bin_idx])

    bound_warn_threshold = 0.99 * b_bounds[1]

    for k in range(n_shuffles):
        perm = rng.permutation(len(J_arr))
        Jp   = J_arr[perm]

        # Magnitude correlation.
        sh_r[k], _ = stats.pearsonr(Jp, abs_p)

        # Bounded W curve fit.
        bin_J  = np.array([Jp[i].mean() for i in bin_idx])
        bin_se = np.array([Jp[i].std(ddof=1) / np.sqrt(len(i))
                           for i in bin_idx])

        try:
            popt, pcov = optimize.curve_fit(
                m_W, bin_p_fixed, bin_J,
                p0=[bin_J.mean(), 0.0],
                sigma=bin_se, absolute_sigma=True,
                maxfev=10000,
                bounds=([-1e4, b_bounds[0]], [1e4, b_bounds[1]]),
            )
            sig_b = (np.sqrt(pcov[1, 1])
                     if np.isfinite(pcov[1, 1]) else np.inf)
            hit_bound  = abs(popt[1]) >= bound_warn_threshold
            huge_sigma = sig_b > abs(b_bounds[1]) / 4.0
            sh_b_fit[k] = np.nan if (hit_bound or huge_sigma) else popt[1]
        except Exception:
            sh_b_fit[k] = np.nan

        # Parameter-free projection.
        sh_b_proj[k] = w_projection(bin_J, bin_p_fixed)

    return sh_r, sh_b_fit, sh_b_proj


def partial_r(r_xy, r_xz, r_yz):
    """Partial correlation r_{xy.z} from three pairwise Pearson r values.

    Formula: r_{xy.z} = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    """
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom < 1e-9:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def main():
    # ---- Stage A: robustness scan -----------------------------------------
    print("=" * 84)
    print("A. Robustness scan -- correlations across block sizes")
    print("=" * 84)
    print(f"\n{'block (s)':>10} {'N':>6} "
          f"{'r(J,sgnMoon)':>14} {'r(J,|Moon|)':>14} "
          f"{'r(J,sgnSid)':>14} {'r(J,|Sid|)':>14} {'r(J,time)':>14}")
    print("-" * 84)

    scan_results = {}
    for bs in BLOCK_SIZES:
        J, s_moon, s_sid, t, files_found = compute_blocks(bs)
        if len(J) == 0:
            print(f"{bs:>10} {'--':>6}  (no NIST files in cwd)")
            continue

        r_sgn_m, p_sgn_m = stats.pearsonr(J, s_moon)
        r_abs_m, p_abs_m = stats.pearsonr(J, np.abs(s_moon))
        r_sgn_s, p_sgn_s = stats.pearsonr(J, s_sid)
        r_abs_s, p_abs_s = stats.pearsonr(J, np.abs(s_sid))
        r_tim,   p_tim   = stats.pearsonr(J, t)

        scan_results[bs] = dict(
            J=J, s_moon=s_moon, s_sid=s_sid, t=t, N=len(J),
            r_sgn_m=r_sgn_m, p_sgn_m=p_sgn_m,
            r_abs_m=r_abs_m, p_abs_m=p_abs_m,
            r_sgn_s=r_sgn_s, p_sgn_s=p_sgn_s,
            r_abs_s=r_abs_s, p_abs_s=p_abs_s,
            r_tim=r_tim, p_tim=p_tim,
        )

        print(f"{bs:>10} {len(J):>6} "
              f"{r_sgn_m:>+7.4f}({p_sgn_m:>6.1e}) "
              f"{r_abs_m:>+7.4f}({p_abs_m:>6.1e}) "
              f"{r_sgn_s:>+7.4f}({p_sgn_s:>6.1e}) "
              f"{r_abs_s:>+7.4f}({p_abs_s:>6.1e}) "
              f"{r_tim:>+7.4f}({p_tim:>6.1e})")

    if SHAPE_BLOCK not in scan_results:
        print(f"\nNo {SHAPE_BLOCK}s blocks available; aborting.")
        return

    # ---- Stage B: proxy-proxy diagnostic ----------------------------------
    print(f"\n{'=' * 84}")
    print("B. Proxy intercorrelation and partial r at "
          f"{SHAPE_BLOCK}s blocks")
    print("=" * 84)

    J      = scan_results[SHAPE_BLOCK]['J']
    s_moon = scan_results[SHAPE_BLOCK]['s_moon']
    s_sid  = scan_results[SHAPE_BLOCK]['s_sid']

    r_proxies, _ = stats.pearsonr(s_moon, s_sid)
    print(f"r(s_Moon, s_sidereal) over {SHAPE_BLOCK}s blocks: "
          f"{r_proxies:+.4f}")
    print("  (high |r| means the proxies measure mostly the same thing)")

    abs_m = np.abs(s_moon)
    abs_s = np.abs(s_sid)
    r_JM, _ = stats.pearsonr(J, abs_m)
    r_JS, _ = stats.pearsonr(J, abs_s)
    r_MS, _ = stats.pearsonr(abs_m, abs_s)

    r_JM_given_S = partial_r(r_JM, r_JS, r_MS)
    r_JS_given_M = partial_r(r_JS, r_JM, r_MS)

    print("\nPartial correlations (controlling for the other proxy):")
    print(f"  r(J, |Moon|)         = {r_JM:+.4f}   (raw)")
    print(f"  r(J, |Moon| | |Sid|) = {r_JM_given_S:+.4f}   "
          f"(Moon, partialling out sidereal)")
    print(f"  r(J, |Sid|)          = {r_JS:+.4f}   (raw)")
    print(f"  r(J, |Sid|  | |Moon|) = {r_JS_given_M:+.4f}   "
          f"(sidereal, partialling out Moon)")
    print()
    print("  If r(J,|Moon||Sid|) >> r(J,|Sid||Moon|): "
          "Moon-specific signal present.")
    print("  If r(J,|Moon||Sid|) ~~ r(J,|Sid||Moon|): "
          "both proxies tracking the same thing.")
    print("  If r(J,|Sid||Moon|) >> r(J,|Moon||Sid|): "
          "sidereal carries it; lunar attribution fails.")

    # ---- Stage C: shape fits ----------------------------------------------
    print(f"\n{'=' * 84}")
    print(f"C. Shape test at {SHAPE_BLOCK}s blocks "
          "-- Moon proxy and sidereal control")
    print("=" * 84)

    # Normalize Moon slope by its peak over a sidereal day centered on the
    # run midpoint, giving signed p in roughly [-1, +1].
    t_run_mid  = ephem.Date(RUN_MIDPOINT)
    scan_times = np.linspace(-SIDEREAL_DAY, SIDEREAL_DAY, 4000)
    scan_moon  = np.array([moon_slope(t_run_mid + dt) for dt in scan_times])
    s_moon_max = np.max(np.abs(scan_moon))
    p_moon     = s_moon / s_moon_max
    p_sid      = s_sid   # already in [-1, +1] by construction

    print("\nFitting same shape family against both proxies, "
          "same bin scheme:")
    fit_moon = run_shape_fit(p_moon, J, "MOON proxy")
    fit_sid  = run_shape_fit(p_sid,  J, "SIDEREAL proxy (control)")

    # ---- Stage D: time-shuffle null ---------------------------------------
    print(f"\n{'=' * 84}")
    print(f"D. Time-shuffle null -- {N_SHUFFLES} permutations")
    print("=" * 84)

    # Observed statistics. Pearson r is scale-invariant, so using p_moon
    # here gives the same value as using raw s_moon.
    obs_r, _ = stats.pearsonr(J, np.abs(p_moon))
    w_fit_dict = [f for f in fit_moon['fits']
                  if f['name'] == 'W (k=1 fixed)'][0]
    obs_b      = w_fit_dict['popt'][1]
    obs_b_proj = w_projection(fit_moon['bin_J'], fit_moon['bin_p'])

    print("\nObserved (Moon proxy):")
    print(f"  r(J, |Moon|)         = {obs_r:+.4f}")
    print(f"  W amplitude (fit)    = {obs_b:+.3f}")
    print(f"  W amplitude (proj.)  = {obs_b_proj:+.3f}   "
          "(parameter-free check)")

    print(f"\nRunning {N_SHUFFLES} shuffles (this takes about a minute)...")
    # The shuffle uses normalized p (in [-1, +1]) so the W template
    # sin(2*pi*|p|) spans its full dynamic range.
    sh_r, sh_b_fit, sh_b_proj = shuffle_test(J, p_moon, N_SHUFFLES)

    n_total      = len(sh_b_fit)
    sh_r         = sh_r[np.isfinite(sh_r)]
    sh_b_fit_ok  = sh_b_fit[np.isfinite(sh_b_fit)]
    sh_b_proj    = sh_b_proj[np.isfinite(sh_b_proj)]
    n_dropped    = n_total - len(sh_b_fit_ok)

    emp_p_r       = (np.sum(np.abs(sh_r)         >= np.abs(obs_r))
                     / len(sh_r))
    emp_p_b_fit   = ((np.sum(np.abs(sh_b_fit_ok) >= np.abs(obs_b))
                      / len(sh_b_fit_ok))
                     if len(sh_b_fit_ok) > 0 else np.nan)
    emp_p_b_proj  = (np.sum(np.abs(sh_b_proj)    >= np.abs(obs_b_proj))
                     / len(sh_b_proj))

    print("\nEmpirical null distributions:")
    print(f"  r(J, |Moon|):")
    print(f"     mean = {sh_r.mean():+.4f}, std = {sh_r.std():.4f}")
    print(f"     observed = {obs_r:+.4f}, "
          f"two-tailed empirical p = {emp_p_r:.4e}")
    print(f"     ({np.sum(np.abs(sh_r) >= np.abs(obs_r))} of "
          f"{len(sh_r)} shuffles as extreme or more)")

    print(f"\n  W amplitude (bounded curve fit, filtered):")
    if len(sh_b_fit_ok) > 0:
        print(f"     mean = {sh_b_fit_ok.mean():+.3f}, "
              f"std = {sh_b_fit_ok.std():.3f}")
        print(f"     observed = {obs_b:+.3f}, "
              f"two-tailed empirical p = {emp_p_b_fit:.4e}")
        print(f"     ({np.sum(np.abs(sh_b_fit_ok) >= np.abs(obs_b))} of "
              f"{len(sh_b_fit_ok)} shuffles as extreme or more)")
        print(f"     [filtered {n_dropped} of {n_total} "
              f"({100 * n_dropped / n_total:.1f}%) as non-converged]")
    else:
        print("     all curve fits filtered as non-converged")

    print(f"\n  W amplitude (parameter-free projection):")
    print(f"     mean = {sh_b_proj.mean():+.3f}, "
          f"std = {sh_b_proj.std():.3f}")
    print(f"     observed = {obs_b_proj:+.3f}, "
          f"two-tailed empirical p = {emp_p_b_proj:.4e}")
    print(f"     ({np.sum(np.abs(sh_b_proj) >= np.abs(obs_b_proj))} of "
          f"{len(sh_b_proj)} shuffles as extreme or more)")
    print()
    print("Interpretation:")
    print("  The projection p is the cleanest single number "
          "-- no optimizer involved.")
    print("  The bounded curve fit p should agree closely with the "
          "projection p;")
    print("  large disagreement would signal residual optimizer "
          "instability.")

    # ---- Plots ------------------------------------------------------------
    _make_plots(scan_results, fit_moon, fit_sid,
                obs_r, obs_b, obs_b_proj,
                sh_r, sh_b_proj,
                emp_p_r, emp_p_b_fit, emp_p_b_proj,
                len(J))


def _make_plots(scan_results, fit_moon, fit_sid,
                obs_r, obs_b, obs_b_proj,
                sh_r, sh_b_proj,
                emp_p_r, emp_p_b_fit, emp_p_b_proj,
                n_blocks):
    """Generate the six-panel summary figure and save to PNG."""
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28,
                           height_ratios=[1, 1, 1])

    bs_arr = np.array(sorted(scan_results.keys()))

    # (A) Robustness scan, both proxies.
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(bs_arr, [scan_results[b]['r_abs_m'] for b in bs_arr], 'o-',
             color='#1abc9c', ms=10, lw=2, label='r(J, |Moon|)')
    axA.plot(bs_arr, [scan_results[b]['r_abs_s'] for b in bs_arr], 'o-',
             color='#e74c3c', ms=10, lw=2,
             label='r(J, |sidereal|)  [control]')
    axA.plot(bs_arr, [scan_results[b]['r_sgn_m'] for b in bs_arr], 's',
             color='#3498db', ms=7, lw=1.3, ls='-', alpha=0.7,
             label='r(J, signed Moon)')
    axA.plot(bs_arr, [scan_results[b]['r_sgn_s'] for b in bs_arr], 's',
             color='#c0392b', ms=7, lw=1.3, ls='--', alpha=0.7,
             label='r(J, signed sidereal)')
    axA.plot(bs_arr, [scan_results[b]['r_tim'] for b in bs_arr], '^-',
             color='#e67e22', ms=7, lw=1.3, label='r(J, time)')
    axA.axhline(0, color='gray', lw=0.5, ls=':')
    axA.set_xscale('log')
    axA.set_xticks(bs_arr); axA.set_xticklabels([str(b) for b in bs_arr])
    axA.set_xlabel('Block size (s)')
    axA.set_ylabel('Pearson r')
    axA.set_title('Robustness scan: Moon proxy vs sidereal control')
    axA.legend(loc='best', fontsize=8)
    axA.grid(alpha=0.3)

    # (B) p-values.
    axB = fig.add_subplot(gs[0, 1])
    axB.semilogy(bs_arr, [scan_results[b]['p_abs_m'] for b in bs_arr],
                 'o-', color='#1abc9c', ms=10, lw=2,
                 label='p(J, |Moon|)')
    axB.semilogy(bs_arr, [scan_results[b]['p_abs_s'] for b in bs_arr],
                 'o-', color='#e74c3c', ms=10, lw=2,
                 label='p(J, |sidereal|)')
    axB.semilogy(bs_arr, [scan_results[b]['p_tim'] for b in bs_arr],
                 '^-', color='#e67e22', ms=7, lw=1.3,
                 label='p(J, time)')
    axB.axhline(0.05, color='red', lw=0.8, ls='--', alpha=0.5,
                label='p = 0.05')
    axB.set_xscale('log')
    axB.set_xticks(bs_arr); axB.set_xticklabels([str(b) for b in bs_arr])
    axB.set_xlabel('Block size (s)')
    axB.set_ylabel('p-value (log)')
    axB.set_title('Magnitude-correlation p-values')
    axB.legend(loc='best', fontsize=8)
    axB.grid(alpha=0.3, which='both')

    # (C) Moon-proxy shape fit.
    axC = fig.add_subplot(gs[1, 0])
    axC.errorbar(fit_moon['bin_p'], fit_moon['bin_J'],
                 yerr=fit_moon['bin_se'],
                 fmt='o', color='k', ms=9, capsize=4,
                 label='NIST data', zorder=10)
    xfine  = np.linspace(-1, 1, 400)
    colors = ['#888', '#3498db', '#1abc9c', '#c0392b', '#9b59b6']
    for f, col in zip(fit_moon['fits'], colors):
        lw = 3 if f['name'].startswith('W (k=1') else 1.4
        axC.plot(xfine, f['fn'](xfine, *f['popt']), color=col, lw=lw,
                 label=f"{f['name']} "
                       f"(dBIC={f['bic'] - fit_moon['bic_min']:.1f})")
    for crit in [-0.75, -0.25, 0, 0.25, 0.75]:
        axC.axvline(crit, color='gold', ls=':', lw=0.7, alpha=0.5)
    axC.axhline(0, color='gray', ls=':', lw=0.5)
    axC.set_xlabel('Signed p (Moon)')
    axC.set_ylabel(f'J per {SHAPE_BLOCK}s block')
    axC.set_title(f'Shape fit: MOON proxy ({n_blocks} blocks)')
    axC.set_xlim(-1.1, 1.1)
    axC.legend(fontsize=7, loc='best')
    axC.grid(alpha=0.3)

    # (D) Sidereal-proxy shape fit (control).
    axD = fig.add_subplot(gs[1, 1])
    axD.errorbar(fit_sid['bin_p'], fit_sid['bin_J'],
                 yerr=fit_sid['bin_se'],
                 fmt='o', color='k', ms=9, capsize=4,
                 label='NIST data', zorder=10)
    for f, col in zip(fit_sid['fits'], colors):
        lw = 3 if f['name'].startswith('W (k=1') else 1.4
        axD.plot(xfine, f['fn'](xfine, *f['popt']), color=col, lw=lw,
                 label=f"{f['name']} "
                       f"(dBIC={f['bic'] - fit_sid['bic_min']:.1f})")
    for crit in [-0.75, -0.25, 0, 0.25, 0.75]:
        axD.axvline(crit, color='gold', ls=':', lw=0.7, alpha=0.5)
    axD.axhline(0, color='gray', ls=':', lw=0.5)
    axD.set_xlabel('Signed p (sidereal, sin(LST))')
    axD.set_ylabel(f'J per {SHAPE_BLOCK}s block')
    axD.set_title('Shape fit: SIDEREAL proxy (no Moon) -- control')
    axD.set_xlim(-1.1, 1.1)
    axD.legend(fontsize=7, loc='best')
    axD.grid(alpha=0.3)

    # (E) Shuffle null for r(J, |Moon|).
    axE = fig.add_subplot(gs[2, 0])
    axE.hist(sh_r, bins=80, color='#7f8c8d', alpha=0.7, edgecolor='none')
    axE.axvline(obs_r, color='#c0392b', lw=2.5,
                label=f'observed = {obs_r:+.4f}')
    axE.axvline(-obs_r, color='#c0392b', lw=2.5, ls=':')
    axE.set_xlabel('r(J, |Moon|) under shuffle')
    axE.set_ylabel('shuffle count')
    axE.set_title(f'Empirical null for r(J, |Moon|)  '
                  f'(N={len(sh_r)} shuffles, p={emp_p_r:.2e})')
    axE.legend(loc='best', fontsize=9)
    axE.grid(alpha=0.3)

    # (F) Shuffle null for W projection amplitude.
    axF = fig.add_subplot(gs[2, 1])
    axF.hist(sh_b_proj, bins=80, color='#7f8c8d', alpha=0.7,
             edgecolor='none')
    axF.axvline(obs_b_proj, color='#c0392b', lw=2.5,
                label=f'observed = {obs_b_proj:+.3f}')
    axF.axvline(-obs_b_proj, color='#c0392b', lw=2.5, ls=':')
    axF.set_xlabel('W projection coefficient under shuffle')
    axF.set_ylabel('shuffle count')
    if np.isfinite(emp_p_b_fit):
        fit_note = f"bounded curve fit p={emp_p_b_fit:.2e}"
    else:
        fit_note = "bounded curve fit p=undefined"
    axF.set_title(f'Empirical null for W projection  '
                  f'(N={len(sh_b_proj)} shuffles, '
                  f'p={emp_p_b_proj:.2e})\n[cross-check: {fit_note}]')
    axF.legend(loc='best', fontsize=9)
    axF.grid(alpha=0.3)

    fig.suptitle('NIST 2015 Bell test: temporal modulation with '
                 'sidereal and shuffle controls',
                 fontsize=14, fontweight='bold', y=1.00)
    out_path = 'NIST_temporal_modulation_with_controls.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
