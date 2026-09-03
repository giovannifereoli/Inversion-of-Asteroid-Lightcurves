# Convex inversion (Section 3)

## The problem

The convex inverse problem is linear (Eq. 2):

$$\mathbf{L} = \mathbf{A}\,\mathbf{g}, \qquad
A_{ij} = S_j\!\left(\mu^{(ij)}, \mu_0^{(ij)}\right) \varpi_j$$

where $g_j$ are the facet areas of a convex polyhedron. Solving it directly fails:

> The standard solution of (2) by minimizing the square norm ... using
> least-squares normal equations or singular-value decomposition would usually
> produce negative $g_j$ values.

The cure is not regularisation but positivity, Eq. (6):

$$g_j = \exp(a_j)$$

> The values of $a_j$ are not constrained, so this is much more practicable than
> using penalty or barrier functions. Also, since the surfaces of constant $\chi^2$
> are convex surfaces (hyperellipsoids) in $g$-space, there is one and only one
> vector $\mathbf{g}$ with $g_j \ge 0$ ... that minimizes $\chi^2$.

## Two parametrisations

They are "rather complementary".

### Harmonic series — Section 3.2

$$G(\vartheta,\psi) = \exp\left(\sum_{lm} a_{lm} Y_{lm}(\vartheta,\psi)\right),
\qquad g_j = G(\vartheta_j,\psi_j)\,\Delta\sigma_j$$

40–100 coefficients, minimised by Levenberg–Marquardt. This is the one to start
with: it "converges very efficiently toward the correct solution even with a poor
initial guess", and keeps converging when the pole, period or scattering law are
freed.

```python
inv = lcinv.HarmonicInversion(
    data, geometry, spin, lmax=6, law=law,
    convexity_weight=0.1, fit_pole=True, fit_period=True,
)
result = inv.run()
```

The initial guess is the paper's: a least-squares fit of $\log G$ for a triaxial
ellipsoid, whose curvature function is
$G(\mathbf{n}) = (abc)^2 / (a^2n_x^2 + b^2n_y^2 + c^2n_z^2)^2$.

### Facet areas — Section 3.1

One parameter per facet, of order 1000 of them, minimised by conjugate gradients.
Use it to polish a harmonic solution:

```python
polish = lcinv.FacetInversion(data, geometry, result.spin, law=law)
better = polish.run(initial=result.areas)
```

> The polyhedron approach is better than the smooth function one: the latter
> produced very good inversion shapes in all cases as well, but the large planar
> areas were not so clearly defined.

## Objectives

Three $\chi^2$ forms, selected with `Objective`:

| | Equation | When |
|---|---|---|
| `ABSOLUTE` | (5) | calibrated photometry, known scale |
| `RENORMALISED` | (7) | calibrated, each curve scaled to mean unity |
| `RELATIVE` | (13) | relative photometry — the usual case |

Equation (13) normalises *both* sides:

$$\chi^2_{\rm rel} = \sum_i \left\| \frac{\mathbf{L}^{(i)}}{\bar{L}^{(i)}} -
\frac{\mathbf{A}^{(i)}\mathbf{g}}{\langle \mathbf{A}^{(i)}\mathbf{g}\rangle}\right\|^2$$

which "discards all scale factors and thus keeps the number of free parameters as
low as possible". A consequence worth remembering: the overall size of the body
cancels, so the constant term of the series is frozen. `lcinv` does this
automatically in both the convex and the nonconvex solvers.

## Convexity

Equation (3), $\sum_j \mathbf{n}_j g_j = 0$, is necessary and sufficient for $g$ to
describe a convex polyhedron. It enters as extra rows, exactly as Section 3.3
describes — "adding three zero elements to $L$ and three new rows to $A$":

```python
lcinv.FacetInversion(..., convexity_weight=0.1, convexity_components="xyz")
lcinv.FacetInversion(..., convexity_components="z")     # Section 3.4's advice
```

The size of the residual is the diagnostic:

> The ratio of the size of the residual nonconvexity (3) to the total surface area
> varied between 0.001 and 0.007 for the four shapes.

`InversionResult.nonconvexity` reports it. Above about 0.01, consider albedo.

## Albedo

Section 3.3 separates shape from albedo by minimising Eq. (11), with albedos
confined to $[a,b]$ by the logistic map of Eq. (12).

```python
loose = lcinv.FacetInversion(..., convexity_components="none").run()
split = lcinv.AlbedoSeparation(geometry, loose.areas).run()
```

!!! note "What this can and cannot give you"
    The paper is unambiguous: "in all realistic cases albedo effects are in
    principle quantitatively inseparable from shape effects", and the result
    "describes the albedo asymmetry rather than the actual distribution". The
    recovered contrast is a lower bound; the ratio `lambda_shape / lambda_albedo`
    decides how much variation is attributed to albedo rather than shape. What the
    method does reliably is the negative result — a constant-albedo body produces
    no albedo signal at all.
