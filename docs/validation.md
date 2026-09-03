# Validation

The point of reimplementing a paper is to get the same answers. Here is how that
is checked.

## Against the original code

DAMIT distributes Kaasalainen's original Fortran, translated to C by Josef Ďurech
([software page](https://damit.cuni.cz/projects/damit/pages/software_download)).
It ships `convexinv` (Section 3.2 plus Paper II), `conjgradinv` (Section 3.1),
`lcgenerator` (Section 2), and a 37-lightcurve test set.

Running `convexinv` on `test_lcs_rel` with its own shipped `input_convexinv`
(λ = 220°, β = 0°, P = 5.76198 h all free; convexity regularisation 0.1;
harmonics 6×6; 8 rows; LSL with $a$ = 0.5, $d$ = 0.1, $k$ = −0.5, $c$ = 0.1),
against `lcinv` on the same data with the same settings and the same initial guess:

| | reference `convexinv` | `lcinv` |
|---|---|---|
| λ | 253.308141° | 253.246° |
| β | −16.917887° | −17.311° |
| P | 5.761982 h | 5.761980 h |
| χ² | 0.378227 | 0.353610 |
| rms | 0.015667 | 0.015163 |

`lcinv` reaches a slightly lower χ² because the C code stops at its 50-iteration
cap while still improving; the two agree on the pole to a fraction of a degree and
on the period to seven significant figures.

Reading the C source also settled the conventions, and confirmed that:

* `mrqcof.c` computes exactly Eq. (13) — the model is divided by its own mean and
  the residual by the observed mean;
* `conv.c` computes exactly Eq. (3);
* the convexity constraint really is three extra rows with zero data and weight
  $1/\text{conw}$, as Section 3.3 describes;
* `bright.c` uses $S = \mu\mu_0(c_l + c_{ls}/(\mu+\mu_0))$ and multiplies by
  $f(\alpha) = 1 + a e^{-\alpha/d} + k\alpha$ — matching `LommelSeeligerLambert`
  and `PhaseFunction`;
* `convexinv` adds the same closing dark facet as `close_facet_areas`, and prints
  its size.

That test set is committed as `tests/data/test_lcs_rel.txt`, and the comparison
runs as `tests/test_convex.py::TestAgainstReferenceImplementation`.

## Against a DAMIT model

The notebook inverts (269) Justitia from scratch and lands 6.8° from DAMIT's
published pole, with a 1.4 % rms fit to 53 lightcurves.

## Internal consistency

| Check | Result |
|---|---|
| Appendix B gift-wrapping vs Qhull, 8 point clouds incl. degenerate coplanar | volume and area agree to 1e-9 |
| Section 2 local-blocker tracer vs brute-force all-pairs tracer | **exact** on peanut, Castalia-like, binary |
| Ray-traced convex body vs plain sum over facets | agrees to 1e-15 |
| Section 4 analytic ∂A/∂r, ∂μ/∂r vs finite differences | 1e-5 |
| All inversion Jacobians vs finite differences | ~1e-4 |
| Minkowski on a cube, a 3×2×1 brick, a sphere, an ellipsoid | recovered from their own facet areas |
| Real spherical harmonics orthonormality on the sphere | 2e-3 (quadrature-limited) |
| Octant triangulation facet/vertex counts | exactly $8N^2$, $4N^2+2$ |
| Closure $\sum_j A_j\mathbf{n}_j = 0$ on every generated body | < 1e-12 |
| Analytic sphere and ellipsoid volumes, areas, inertia tensors | to discretisation error |

## Reproducing the paper's own findings

Some of the paper's stated results fall out as tests:

* **Residual nonconvexity 0.001–0.007 for constant-albedo bodies** (Section 3.5) —
  reproduced on synthetic ellipsoids and on Justitia (0.0052).
* **5–10 % noise does not require regularisation** (Section 3.5) — the solution
  degrades gracefully and stays convex.
* **No albedo indication for a constant-albedo surface** (Section 3.5) —
  `AlbedoSeparation` returns a flat albedo map and an unchanged shape.
* **The convex hull is brighter than the body it encloses** (Section 3.5) — the
  hull's mean lightcurve exceeds the body's.
* **The facet method improves on the series** (Section 3.5) — χ² drops on both
  synthetic and real data.

## Running them

```bash
pytest                    # 209 tests, ~4 min
pytest -m "not slow"      # ~30 s
pytest --cov=lcinv
```
