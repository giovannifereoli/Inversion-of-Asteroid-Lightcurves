# The complete recipe (Paper II)

Paper I solves the shape with the rotation state given. Paper II closes the loop
and ends with a numbered recipe, which `lcinv.pipeline` implements.

## Step 1 — period sampling

If the data span many apparitions, the period axis is filled with densely packed
local minima. Paper II Eq. (2) gives their separation:

$$\frac{\Delta P}{P} \approx \frac{P}{2T}, \qquad T = \max|t - t_0|$$

Scanning coarser than this steps straight over minima.

```python
from lcinv import period_scan, pole_grid
periods, chi2 = period_scan(data, (5.0, 9.0), poles=pole_grid(6))
best = periods[chi2.argmin()]
```

## Step 2 — initial pole

> It is important to use all (typically four) poles as initial guesses, since the
> best ellipsoid pole is not necessarily in the correct "valley." Another
> possibility is simply to use, say, a few directions in each octant.

`pole_grid(n)` gives that grid.

## Steps 3–8

```python
pipeline = lcinv.InversionPipeline(data, geometry, law=law, lmax=6)
result = pipeline.run(
    spin,
    convexity_weights=(0.1, 1.0),   # step 3: "try different ... weights"
    fit_pole=True, fit_period=True,
    refine_facets=True,             # step 4
    separate_albedo=None,           # step 5: automatic on residual nonconvexity
    reconstruct=True,               # step 7: Minkowski
    n_restarts=5,                   # step 8: error estimates
)
print(result.report())
```

| Step | What happens | Implemented by |
|---|---|---|
| 3 | Function series with pole, period, scattering free; several regularisation weights; keep the best | `HarmonicInversion` |
| 4 | Switch to separate facets, starting from step 3 | `FacetInversion` |
| 5 | Separate shape and albedo *if* residual nonconvexity is real | `AlbedoSeparation` |
| 6 | Scale factors → solar phase function | `PhaseFunction`, `fit_scattering=True` |
| 7 | Shape from facet areas | `minkowski_solve` |
| 8 | Repeat from nearby starts for error estimates | `n_restarts` |

Step 5 fires automatically when `InversionResult.nonconvexity` exceeds
`nonconvexity_threshold` (default 0.01), which sits just above the 0.001–0.007 band
Section 3.5 measures for constant-albedo bodies.

Step 8 reports `pole_scatter` (degrees) and `period_scatter` (hours) across the
restarts. Every step is recorded in `result.log`.

## What the recipe cannot do for you

Paper II is candid that the data set decides the outcome:

> The potential insufficiency of the data shows during the inversion procedure;
> even large data sets may not be as informative as smaller but more varied ones.
> Often the pole and the period can still be expected to be accurate even if no
> details can be obtained for the shape.
