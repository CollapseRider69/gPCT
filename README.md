# Slope Traversal and Time: A Minimal Statement of Gravitational Phase-Cancellation Theory (gPCT)

**4-page core paper** (2025) — the complete, self-contained minimal statement.

- **[PDF](gPCT.pdf)** — final compiled version
- **[LaTeX source](gPCT.tex)** — fully compilable (`pdflatex`)
- **[Simulator](gPCT_SIMULATOR.py)** — numerical verification of the attractor and predictions

## Zenodo Archive

Permanent DOI: [10.5281/zenodo.17266831](https://doi.org/10.5281/zenodo.17266831)

## Overview

gPCT is a relational framework built from a single primitive: the gravitational phase slope $`s = D\phi_g`$ along a worldline. Dual recursion yields curvature, time, the Born-rule collapse prediction, and the White Equation attractor $`\mathcal{H} \to -1/2\pi`$ (with $`s_\infty = 2\pi`$) without quantizing gravity or adding new fields.

## Key Predictions

- Slope-modulated collapse: $`P(|1\rangle) = \cos^2\!\left( \frac{\pi s}{2} - \frac{\pi}{4} \right)`$
- CHSH W-shape: $`S(p) = 1 + \sqrt{2} + (\sqrt{2}-1)\sin(2\pi |p|)`$ (cycle average $`1 + \sqrt{2} \approx 2.4142`$)

## Running the Simulator

```bash
pip install numpy matplotlib scipy
python gPCT_SIMULATOR.py
```

The script generates:

- Convergence of any worldline with $`s_\infty = 2\pi`$ to $`\mathcal{H} = -1/2\pi`$
- The exact W-curve and CHSH signature
- All plots saved as `gPCT_v5_SIMULATOR.png`

## Citation

```bibtex
@misc{white2025gpct,
  author       = {White, Christopher Dean},
  title        = {Slope Traversal and Time: A Minimal Statement of Gravitational Phase-Cancellation Theory (gPCT)},
  year         = {2025},
  doi          = {10.5281/zenodo.17266831},
  url          = {https://doi.org/10.5281/zenodo.17266831}
}
```
