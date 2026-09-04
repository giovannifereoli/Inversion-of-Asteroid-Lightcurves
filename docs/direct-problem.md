# The direct problem (Section 2)

Before you can invert lightcurves you have to be able to compute them.

## Scattering

Equation (1) gives the brightness contributed by a patch $ds$ that is both visible
and illuminated:

$$dL = S(\mu, \mu_0)\,\varpi\,ds$$

with $\mu = \mathbf{E}\cdot\mathbf{n}$, $\mu_0 = \mathbf{E}_0\cdot\mathbf{n}$. The
paper names two laws:

* Lambert, $S_L = \mu\mu_0$
* Lommel–Seeliger, $S_{LS} = \mu\mu_0/(\mu+\mu_0)$

and Section 3.5 uses "a combination of Lommel-Seeliger and Lambert laws with equal
weights" throughout — `LommelSeeligerLambert(lambert_weight=1.0)`.

For real photometry you also need the empirical solar phase function of Paper II,
$f(\alpha) = a\,e^{-\alpha/d} + k\alpha + 1$, which is `PhaseFunction`. That
combination is DAMIT's `LSL` model.

```python
from lcinv import LommelSeeligerLambert, PhaseFunction

paper = LommelSeeligerLambert(1.0)                                   # simulations
damit = LommelSeeligerLambert(0.1, PhaseFunction(0.5, 0.1, -0.5))    # real data
```

Every law returns exactly zero where $\mu \le 0$ or $\mu_0 \le 0$ — the "of course"
in the paper's remark that $A_{ij}$ vanishes there.

### Hapke macroscopic roughness

`lcinv.Hapke` implements the single-scattering albedo, Henyey–Greenstein phase
function, opposition surge, Chandrasekhar $H$ approximation **and** the
macroscopic-roughness correction of Hapke (1984; 1993, ch. 12). The true cosines are
replaced by the effective $\mu_e$, $\mu_{0e}$ of a surface carrying unresolved
slopes of mean angle $\bar\theta$, and the result is multiplied by the shadowing
function $S(i, e, \psi)$; the azimuth follows from
$\cos\psi = (\cos\alpha - \mu\mu_0)/(\sin i \sin e)$.

Setting $\bar\theta = 0$ reproduces the smooth law to machine precision. The
correction darkens the surface, weakly near opposition and strongly at large phase
angle — about 2% at $\alpha = 2^\circ$ but 46% at $\alpha = 60^\circ$ for
$\bar\theta = 40^\circ$.

This is a *sub-facet* model: it describes slopes below the resolution of the mesh,
and is complementary to the resolved inter-facet shadowing that
[`RayTracer`](direct-problem.md) computes.

!!! note "Identifiability"
    Under `Objective.RELATIVE` the albedo $w$ multiplies the whole law and cancels
    in Eq. (13), so `Hapke.free_parameter_mask` holds it fixed. Every law also
    declares `parameter_bounds`; without them a fit will return a negative
    opposition surge, which is marginally better and physically meaningless.

## Triangulation

Appendix A's octant scheme divides the sphere into eight octants, each into $N$
rows of equal polar-angle spacing, giving $8N^2$ facets and $4N^2+2$ vertices, with
"the polar facets ... about the same size as the equatorial ones".

```python
from lcinv import octant_triangulation
mesh = octant_triangulation(8)      # 512 facets, 258 vertices
```

## Ray tracing

For a nonconvex body the lightcurve must be computed numerically. Section 2's
scheme avoids testing every facet against every other:

> First one checks which vertices are above each facet's local horizon and which
> facets connected to these vertices are facing this facet. These facets are the
> possible local blockers of light ... The facets for which no vertices appear
> above the local horizon belong to the convex hull.

```python
from lcinv import RayTracer, paper_shape

tracer = RayTracer(paper_shape("peanut"))
print(tracer.hull_facet_mask.sum(), "facets can never be blocked")
L = tracer.brightness(earth, sun, law)
```

Facets that represent a large fraction of the surface can be sampled at several
points — "in a hexagonal mesh with small random perturbations" — via
`RayTracer(body, n_subpoints=10)`. The paper notes the plain centroid check "is
quite accurate if there are hundreds of facets".

Binary objects work with the same code: the blocker search runs over all vertices,
so the two components correctly eclipse each other.

## Convex hulls

Appendix B's gift-wrapping algorithm is implemented literally, including step 8
(merging coplanar triangles into polygonal faces). Those merged faces matter:

> The convex hulls typically contain large planar parts forming bridges over the
> valleys of the original shapes; such facets have a key role in convex inversion.

```python
from lcinv import convex_hull
hull = convex_hull(points, method="giftwrap")     # Appendix B
hull = convex_hull(points, method="qhull")        # SciPy
hull = convex_hull(points)                        # "auto": N**2 below 1000 points
```

Both routes are checked against each other to 1e-9 in volume and area, including
on degenerate coplanar clouds.

## Test bodies

Section 3.5's four bodies "of increasing nonconvexity" are reconstructed in the
same spirit as the originals (which were never published numerically):

```python
from lcinv import paper_shape, PAPER_SHAPE_NAMES
PAPER_SHAPE_NAMES        # ('irregular', 'castalia', 'peanut', 'binary')
body = paper_shape(3)    # or paper_shape("peanut")
```

`gaussian_random_sphere` implements the Muinonen (1998) family the paper
repeatedly points to.

## The key empirical fact

> It is actually quite surprising how similar the lightcurves of even a strongly
> nonconvex body and its convex hull are.

This is why convex inversion works, and equally why the depth of a concavity is not
recoverable from photometry. `lcinv.optimal_scale` implements Eq. (14), the scale
coefficient the paper uses when comparing a body with its hull.

## Optional Rust kernels

The occlusion test and the local-blocker precomputation are available as an
optional Rust extension in `src/lcinv_rust/`, built with

```bash
maturin develop --release -m src/lcinv_rust/Cargo.toml
```

`RayTracer` picks it up automatically (`backend="auto"`); `backend="python"`
forces the NumPy path. `lcinv.raytracer.ACCELERATED` reports which is in use.

The kernels mirror the NumPy code exactly, tolerances included — the same
`1e-14` determinant cut-off, the same back-face cull, the same `8 * eps` nudge
off the surface — so the test suite can compare them element for element.
Parallelism is over *observations* (`rayon`), which is the axis with no shared
state.

Typical gain is 40–70x on nonconvex bodies, and a nonconvex inversion of
(269) Justitia drops from 164 s to 4.4 s.

!!! note "It is not a different algorithm"
    If the two backends ever disagree, that is a bug in one of them. The tests
    assert identical hull masks and blocker sets, and lightcurves matching to
    ~1e-16.
