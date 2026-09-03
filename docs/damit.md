# Working with real data

## DAMIT

[DAMIT](https://damit.cuni.cz) publishes lightcurves and derived models for over a
thousand asteroids, in exactly the form this package needs: brightnesses with the
asteroid-centric ecliptic vectors to the Sun and the Earth attached, already
light-time corrected.

```python
client = lcinv.DamitClient(cache_dir="data/damit")

asteroid = client.find_asteroid(number=269)      # or name="Justitia"
models = client.models_for(asteroid["id"])       # best quality flag first

model, data = client.bundle(4966, fmt="json")    # model + its lightcurves
```

Everything is cached, so scripts re-run offline.

!!! note "Three different id numbers"
    The minor-planet **number** (269), the DAMIT **asteroid id** (2858) and the
    DAMIT **model id** (4966) are all different. The number in an
    `/asteroid_models/view/<id>` URL is the model id, and that is what `bundle`
    takes; lightcurve exports are keyed on the asteroid id.

## What comes back

```python
model.spin          # SpinState: lambda, beta, period, t0, phi0, yorp
model.law           # scattering law from the lsm code and parameters
model.shape         # the published Polyhedron
model.quality_flag  # 0-5; 3+ means "reliable, based on large photometric data sets"

data.summary()      # curve count, points, phase-angle range, span
data[0].meta["references"]   # bibliography, with fmt="json"
```

## Choosing a target

Section 3.5's requirements are worth checking before inverting anything:

```python
s = data.summary()
print(s["n_curves"], s["phase_max_deg"], s["n_above_20deg"])
```

> For a more detailed solution, there should be at least a few lightcurves with
> $\alpha$ greater than, say, 20°.

Main-belt asteroids are never seen at large phase angles from Earth, which caps
what any inversion can recover about their surfaces — the overall dimensions will
be fine, the detail will not. Near-Earth objects reach much larger $\alpha$.

If a set is large, take a spread rather than a block:

```python
subset = data.select_geometries(12)
```

## Other data sources

`LightcurveSet` needs, per point: a light-time corrected epoch, a brightness in
*intensity* units, and the asteroid-centric ecliptic Sun and Earth vectors in AU.

For data that lack the vectors — [ALCDEF](https://alcdef.org), or your own
observations — compute them from an ephemeris service (JPL Horizons via
`astroquery.jplhorizons`, say), convert magnitudes to intensities with
$I = 10^{-0.4 m}$, and apply `Lightcurve.light_time_corrected()` if the epochs are
not corrected already.

```python
curve = lcinv.Lightcurve(
    jd=jd, brightness=10 ** (-0.4 * mag),
    sun=sun_xyz_au, earth=earth_xyz_au,
    calibrated=False, name="2024-03-11 R",
).light_time_corrected()
```

## Citing what you use

DAMIT content is CC BY 4.0. Cite Ďurech, Sidorin & Kaasalainen (2010), and the
per-lightcurve references DAMIT preserves — `fmt="json"` keeps them in
`curve.meta["references"]`.
