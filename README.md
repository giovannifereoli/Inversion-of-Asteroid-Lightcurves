# lcinv — asteroid lightcurve inversion

A complete, tested Python implementation of

> M. Kaasalainen & J. Torppa (2001), *Optimization Methods for Asteroid Lightcurve
> Inversion. I. Shape Determination*, **Icarus 153**, 24–36
> ([doi:10.1006/icar.2001.6673](https://doi.org/10.1006/icar.2001.6673))

together with the eight-step recipe of its companion paper (Paper II,
[doi:10.1006/icar.2001.6674](https://doi.org/10.1006/icar.2001.6674)), a Bayesian
layer for the same forward model, and a client for
[DAMIT](https://damit.cuni.cz) so it can be run on real photometry.

Both papers are in [`references/`](references/).

---

## What is implemented

Every numbered equation and every appendix of Paper I:

| Paper | What it is | Where |
|---|---|---|
| Eq. (1) | Scattering laws — Lambert, Lommel–Seeliger, their combination, Hapke | [`scattering.py`](src/lcinv/scattering.py) |
| Section 2 | Ray-traced lightcurves of nonconvex bodies, with local-blocker precomputation | [`raytracer.py`](src/lcinv/raytracer.py) |
| Eq. (2), (4) | The linear forward problem `L = A g` | [`convex.py`](src/lcinv/convex.py) |
| Eq. (3) | Convexity constraint `Σ nⱼ gⱼ = 0` | `convex.py`, `nonconvexity_residual` |
| Eq. (5), (6), (7) | χ², exponential positivity, renormalised χ² | `convex.FacetInversion` |
| Eq. (8), (9), (10) | Exponential spherical-harmonics curvature function | `convex.HarmonicInversion` |
| Eq. (11), (12) | Shape/albedo separation | [`albedo.py`](src/lcinv/albedo.py) |
| Eq. (13) | Relative-brightness χ² | `convex.Objective.RELATIVE` |
| Eq. (14) | Optimal scale coefficient | `lightcurve.optimal_scale` |
| Section 3.5 | The four test bodies | [`shapes.py`](src/lcinv/shapes.py) |
| Eq. (15), (16) | Nonconvex inversion, spherical and cylindrical series | [`nonconvex.py`](src/lcinv/nonconvex.py) |
| Section 4 | Analytic ∂A/∂r, ∂μ/∂r; "sunk area" regulariser | `nonconvex.py` |
| Appendix A | Octant triangulation | [`triangulation.py`](src/lcinv/triangulation.py) |
| Appendix B | Gift-wrapping convex hull | [`convexhull.py`](src/lcinv/convexhull.py) |
| Appendix C, Eq. (17)–(19) | Minkowski minimisation, dual transform | [`minkowski.py`](src/lcinv/minkowski.py) |
| Paper II, Eq. (2) + recipe | Period sampling, pole grid, the 8 steps | [`pipeline.py`](src/lcinv/pipeline.py) |
| — | Posterior sampling with `emcee` | [`bayes.py`](src/lcinv/bayes.py) |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot,bayes,notebook]"
```

Only `numpy` and `scipy` are required; plotting, MCMC and notebook support are extras.

---

## Quick start

```python
import lcinv

client = lcinv.DamitClient()
model, data = client.bundle(4966)          # (269) Justitia — downloads and caches

result = lcinv.InversionPipeline(data).run(model.spin)
print(result.report())

result.shape.polyhedron.to_obj("justitia.obj")
```

From the command line:

```bash
lcinv fetch 4966                                   # download a target
lcinv invert 4966 --restarts 3 --out out.obj       # full Paper II recipe
lcinv demo --shape peanut --noise 0.05             # synthetic end-to-end check
```

---

## Worked example

[`notebooks/justitia_inversion.ipynb`](notebooks/justitia_inversion.ipynb) runs the
whole thing on **(269) Justitia** (DAMIT model 4966, 53 lightcurves, 2339 points):
the direct problem, convex inversion, facet refinement, Minkowski reconstruction,
nonconvex inversion, and an MCMC pole posterior. It is committed with outputs.

Selected results from that run:

| | this code | DAMIT model 4966 |
|---|---|---|
| pole λ, β | 76.6°, −74.3° | 73.0°, −81.0° |
| pole difference | **6.8°** | — |
| rms residual | 1.40 % | — |
| a : b : c | 1.234 : 1.120 : 1 | 1.412 : 1.228 : 1 |
| residual nonconvexity | 0.0052 | — |

That last number is the Eq. (3) ratio, which Section 3.5 measures at 0.001–0.007 for
constant-albedo bodies — so no albedo variegation is called for.

**A caveat the data force.** Justitia never exceeds a 19.6° solar phase angle, and
Section 3.5 is explicit that "due to the lack of shadowing effects, shape information
in observations made at small solar phases is often restricted to the general
dimensions of the target". The overall dimensions here are trustworthy; fine surface
detail is not, and the nonconvex fit does not beat the convex one.

---

## Validation against the original code

DAMIT distributes Ďurech's C translation of Kaasalainen's original Fortran
([software page](https://damit.cuni.cz/projects/damit/pages/software_download)).
Running its `convexinv` on its own shipped 37-lightcurve test set, with its shipped
`input_convexinv`, and running this package on the same data with the same settings:

| | reference `convexinv` | `lcinv` |
|---|---|---|
| λ | 253.308° | 253.246° |
| β | −16.918° | −17.311° |
| P | 5.761982 h | 5.761980 h |
| χ² | 0.378227 | 0.353610 |

Both start from the same initial guess (λ = 220°, β = 0°, P = 5.76198 h). `lcinv`
reaches a slightly lower χ² because the C code stops at its 50-iteration cap while
still improving. That dataset is committed as `tests/data/test_lcs_rel.txt` and the
comparison runs as a test (`tests/test_convex.py::TestAgainstReferenceImplementation`).

Other checks worth naming:

* the Appendix B gift-wrapping hull agrees with Qhull to 1e-9 in volume and area on
  eight point clouds, including degenerate coplanar ones;
* the Section 2 local-blocker tracer reproduces a brute-force all-pairs tracer
  **exactly** on the peanut, Castalia-like and binary bodies;
* the Section 4 analytic derivatives ∂A/∂r and ∂μ/∂r match finite differences to 1e-5;
* Minkowski minimisation recovers a cube, a 3×2×1 brick, a sphere and an ellipsoid
  from their own facet areas;
* every analytic Jacobian is checked against finite differences.

---

## Tests

```bash
pytest                    # everything (~4 min)
pytest -m "not slow"      # fast subset (~30 s)
```

209 tests. Long-running numerical checks are marked `slow`.

---

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

See [`docs/`](docs/) — an overview, a page per paper section mapping equations to
code, and the API reference.

---

## Limitations

Stated plainly, because they affect what the results mean:

* **Hapke roughness is not implemented.** `lcinv.Hapke` covers the single-scattering
  albedo, Henyey–Greenstein phase function, opposition surge and the Chandrasekhar
  `H` approximation, but *not* the macroscopic-roughness correction (θ̄). DAMIT's
  Justitia model uses θ̄ = 20°, so the notebook fits with the Lommel-Seeliger +
  Lambert law both papers use instead of silently misapplying published parameters.
* **Albedo separation recovers asymmetry, not a map.** This is the paper's own
  position — "albedo effects are in principle quantitatively inseparable from shape
  effects", and the recovered contrast is a lower bound that depends on the ratio of
  the two regularisation weights.
* **Nonconvex inversion is qualitative.** Section 4: "the existence of, say, valleys
  is indicated, but the depths of the valleys are not very precise." On synthetic
  data the code recovers 6.1 % nonconvexity against a true 11.8 % — the concavity is
  found, its depth under-estimated.
* **The MCMC needs long chains.** The posterior is sharp and strongly correlated;
  the notebook's demonstration chain is far shorter than τ warrants.
* Sidereal period and pole are *inputs* to Paper I. `pipeline.py` adds Paper II's
  machinery for solving them, but a blind period search over a wide range is
  expensive and is not what this package is optimised for.

---

## Citing

If you use this, cite the papers rather than the code:

* Kaasalainen, M. & Torppa, J. (2001). *Optimization Methods for Asteroid Lightcurve
  Inversion. I. Shape Determination.* Icarus **153**, 24–36.
* Kaasalainen, M., Torppa, J. & Muinonen, K. (2001). *Optimization Methods for
  Asteroid Lightcurve Inversion. II. The Complete Inverse Problem.* Icarus **153**,
  37–51.
* Ďurech, J., Sidorin, V. & Kaasalainen, M. (2010). *DAMIT: a database of asteroid
  models.* A&A **513**, A46.

DAMIT data are CC BY 4.0; the per-lightcurve references are preserved in
`curve.meta["references"]` and printed at the end of the notebook.

---

## Licence

MIT — see [`LICENSE`](LICENSE). The bundled papers in `references/` are the
publishers' copyright and are included for convenience only.
