# Nonconvex inversion (Section 4)

> This problem is a very demanding one, the main complications being that all
> uniqueness theorems are lost and the parameter space is usually plagued by local
> minima.

Instead of facet areas, the parameters are now the vertex radii of a triangulated
surface, described by a short series. Trial lightcurves come from the ray tracer,
so shadowing is modelled rather than assumed away.

## Two parametrisations

Equation (15), radii along spherical directions:

$$r(\theta,\varphi) = \exp\left(\sum_{lm} c_{lm} Y_{lm}(\theta,\varphi)\right)$$

```python
series = lcinv.RadialShapeSeries(n_rows=6, lmax=4)     # 25 coefficients
```

Equation (16), a horizontal cylindrical system, which Section 5 recommends for
contact binaries — "may sometimes be better described by moving vertices along the
radius directions of a horizontal cylindrical coordinate system":

$$\rho(x,\phi) = \exp\left(\sum_{jk} c_{jk}\,x^j e^{ik\phi}\right)$$

```python
series = lcinv.CylindricalShapeSeries.from_body(convex_result, n_x=14, n_phi=20)
```

## Running it

The paper is firm that the starting point matters — "the initial guess should be a
good one (e.g., the series fitted to a convex inversion result)":

```python
inv = lcinv.NonconvexInversion(data, spin, series=series, law=law)
result = inv.run(initial=series.fit(convex_body))
```

Truncate early. Section 4 uses degree and order four, "since the effects of
detailed nonconvexities are certainly drowned in the noise — this is why convex
inversion actually offers better spatial resolution than nonconvex inversion".

## Regularisation

> A useful method is to minimize the area "sunk below" the convex hull of the
> current result, i.e., to encourage convexity. The regularization term consists of
> the sum of the areas of the facets not in the convex hull, each multiplied by the
> average "height" of the vertices of possible blockers above the local horizon.

Both ingredients already fall out of the ray tracer's local-blocker bookkeeping,
so `convexity_penalty(tracer)` is cheap:

```python
lcinv.NonconvexInversion(..., regularisation=1.0)
```

On (269) Justitia, raising the weight walks the solution back to convexity:

| weight | χ² | sunk area | nonconvexity |
|---|---|---|---|
| 0 | 0.322 | 1.008 | 23.3 % |
| 1 | 0.438 | 0.024 | 0.6 % |
| 20 | 0.509 | 0.001 | 0.0 % |

## The analytic derivatives

Section 4 gives them explicitly:

$$\frac{\partial A}{\partial r} = \frac{\hat{\mathbf{r}}\cdot(\mathbf{d}\times\mathbf{n})}{2},
\qquad
\frac{\partial\mu}{\partial r} = \frac{1}{A}\left[\frac{\hat{\mathbf{r}}\cdot(\mathbf{d}\times\mathbf{E})}{2} - \mu\frac{\partial A}{\partial r}\right]$$

`facet_radius_derivatives` implements both, verified against finite differences to
$10^{-5}$. They give the gradient of the *unshadowed* contribution; the shadowing
test is piecewise constant in the parameters, so the solver differences the full
ray-traced model.

## How much to believe

Paper II sets the bar:

> At least as good a $\chi^2$ as that of the convex model, and the same major
> surface features from different initial values and regularization strengths. In
> practice, these conditions typically seem to be fulfilled no better than
> moderately well.

On synthetic peanut data `lcinv` recovers 6.1 % nonconvexity against a true 11.8 %:
the concavity is found, its depth under-estimated. That is Section 4's own
expectation — "the existence of, say, valleys is indicated, but the depths of the
valleys are not very precise".

On Justitia, whose phase angle never exceeds 20°, the nonconvex fit does **not**
beat the convex one. There is simply too little shadowing signal in the data.
