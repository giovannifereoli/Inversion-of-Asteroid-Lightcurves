# Getting started

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot,bayes,notebook]"
```

`numpy` and `scipy` are the only hard requirements. The extras add matplotlib
(`plot`), emcee and corner (`bayes`), Jupyter (`notebook`), and the test tools
(`dev`).

To run the notebook, register the environment as a kernel:

```bash
python -m ipykernel install --user --name lcinv --display-name "Python (lcinv)"
```

then choose **Python (lcinv)** in Jupyter. The notebook's first cell fails with an
explicit message if the kernel does not have `lcinv` installed.

## A synthetic run

The fastest way to see the whole method work is on a body whose answer you know:

```python
import numpy as np
import lcinv

spin = lcinv.SpinState(lam=60.0, beta=25.0, period=6.0, t0=2450000.0, phi0=0.0)
law = lcinv.LommelSeeligerLambert(1.0)          # the paper's equal-weight law
truth = lcinv.paper_shape("peanut")             # Section 3.5, shape 3

# ... generate lightcurves with lcinv.RayTracer, then:
geometry = lcinv.FacetGeometry.from_sphere(8)   # 512 normals
result = lcinv.HarmonicInversion(data, geometry, spin, lmax=6, law=law).run()
shape = result.shape(geometry)
```

Or simply:

```bash
lcinv demo --shape peanut --curves 10 --noise 0.02
```

which builds the body, ray-traces lightcurves, inverts them and compares the
recovered axis ratios with the true convex hull.

## A real target

```bash
lcinv fetch 4966                               # (269) Justitia, cached under data/damit
lcinv invert 4966 --restarts 3 --out out.obj
```

Everything DAMIT serves is cached on disk, so subsequent runs work offline.

## Choosing the resolution

Section 3.5 is specific about the number of facet normals:

> It turned out that the number of parameters should be of order 1000
> (corresponding to evenly distributed surface normals) to make the result
> independent of the exact choice of the normal directions.

`FacetGeometry.from_sphere(N)` gives $8N^2$ normals, so `N = 11` gives 968. `N = 8`
(512) is a good compromise while experimenting. The harmonic series wants far
fewer parameters — "typically from, say, 40 to 100", i.e. `lmax` between 6 and 9.

## What the data need to look like

Section 3.5 again:

> Ten curves were used here for each case: they suffice so long as they cover a
> wide range of observing geometries and there are sufficiently many lightcurve
> points. ... For a more detailed solution, there should be at least a few
> lightcurves with $\alpha$ greater than, say, 20°.

`LightcurveSet.summary()` reports exactly these quantities, and
`select_geometries(n)` picks a spread-out subset rather than a contiguous block.
