# lcinv

A complete implementation of

> M. Kaasalainen & J. Torppa (2001), *Optimization Methods for Asteroid Lightcurve
> Inversion. I. Shape Determination*, **Icarus 153**, 24–36.

What an asteroid lightcurve gives you is a one-dimensional brightness series. What
you want is a three-dimensional shape. The paper's central insight is that this
apparently hopeless inverse problem is not actually ill-posed once you insist the
answer be *physically possible*:

> The positivity constraint is quite sufficient by itself for removing the apparent
> ill-posedness of the problem. No particular regularization methods are necessary:
> the problem is stabilized simply by demanding that the result of inversion be
> restricted to feasible shapes.

Concretely: the brightness is linear in the facet areas $g_j$ of a convex polyhedron
(Eq. 2), those areas must be positive, and writing $g_j = \exp(a_j)$ (Eq. 6) makes
that automatic. The problem becomes nonlinear but well-behaved, with a single
minimum.

## Where each part of the paper lives

| Paper | Module |
|---|---|
| Eq. (1), scattering laws | [`lcinv.scattering`](api.md#scattering-laws) |
| Section 2, nonconvex direct problem | [`lcinv.raytracer`](direct-problem.md) |
| Section 3.1, facet areas + conjugate gradients | [`lcinv.convex.FacetInversion`](convex-inversion.md) |
| Section 3.2, harmonics + Levenberg–Marquardt | [`lcinv.convex.HarmonicInversion`](convex-inversion.md) |
| Section 3.3, albedo separation | [`lcinv.albedo`](convex-inversion.md#albedo) |
| Section 3.4, relative photometry | [`lcinv.convex.Objective`](convex-inversion.md#objectives) |
| Section 3.5, the four test bodies | [`lcinv.shapes`](direct-problem.md#test-bodies) |
| Section 4, nonconvex inversion | [`lcinv.nonconvex`](nonconvex.md) |
| Appendix A, octant triangulation | [`lcinv.triangulation`](direct-problem.md#triangulation) |
| Appendix B, gift-wrapping hull | [`lcinv.convexhull`](direct-problem.md#convex-hulls) |
| Appendix C, Minkowski minimisation | [`lcinv.minkowski`](minkowski.md) |
| Paper II, the eight-step recipe | [`lcinv.pipeline`](pipeline.md) |
| Posterior sampling (added here) | [`lcinv.bayes`](bayesian.md) |

## In thirty seconds

```python
import lcinv

model, data = lcinv.DamitClient().bundle(4966)     # (269) Justitia
result = lcinv.InversionPipeline(data).run(model.spin)
print(result.report())
```

## Next

* [Getting started](getting-started.md) — install and first run
* [Validation](validation.md) — how this is checked against the original C code
* The worked notebook, `notebooks/justitia_inversion.ipynb`
