# Bayesian inversion

Not in the papers. The papers estimate uncertainty by repetition — Paper II's step
8 says to "repeat steps 2–7 with different (but nearby) initial values and
scattering laws to obtain error estimates". This module keeps exactly the same
forward model and the same exponential positivity of Eq. (6), but samples the
posterior with [`emcee`](https://emcee.readthedocs.io) instead of minimising.

That buys three things:

* uncertainties on the pole and period, rather than a scatter of restarts;
* an explicit noise level, marginalised over instead of assumed;
* the convexity constraint Eq. (3) as a *prior*, so its strength is stated in
  probabilistic terms rather than as a weight on extra rows.

```python
inv = lcinv.BayesianInversion(
    data, geometry, spin,
    lmax=3,                    # keep the dimension near 20
    fit_pole=True, fit_period=False,
    convexity_sigma=0.01,      # Section 3.5's 0.001-0.007 sets the scale
)

deterministic = lcinv.HarmonicInversion(
    data, geometry, spin, lmax=3, law=law,
    convexity_weight=0.0, convexity_components="none", fit_pole=True,
).run()

posterior = inv.run(
    n_walkers=44, n_steps=4000, burn=2000,
    start=inv.start_from_result(deterministic),
)
print(posterior.summary())
```

Starting the chain at the optimiser's solution matters: the posterior is sharp, and
burn-in should explore the mode rather than search for it. Doing so takes the
acceptance fraction from ~0.02 to ~0.18 on the Justitia data.

## Priors

* **Pole** — isotropic, $p(\beta) \propto \cos\beta$, i.e. uniform per unit solid
  angle. Anything else biases the pole towards the ecliptic poles.
* **Shape coefficients** — zero-mean Gaussian of width `coefficient_scale`, the
  Bayesian counterpart of a smoothness prior. The constant term is left flat since
  it is a scale factor.
* **Period** — uniform in a narrow window. The $\chi^2(P)$ surface is densely
  multimodal (Paper II Eq. 2), so a broad prior would sample nonsense; run a
  period scan first.
* **Noise** — $\log\sigma$ uniform over a wide range.

## Reading the output

```python
posterior.summary()          # median and 16th/84th percentiles
posterior.spin_samples()     # (n, k) of the sampled rotation parameters
posterior.shape_samples(20)  # 20 bodies drawn from the posterior
```

The spread of `shape_samples` is the honest picture of which features the data
actually constrain.

!!! warning "Degenerate coordinates"
    For a near-pole-on asteroid, ecliptic longitude is close to meaningless — all
    $\lambda$ describe nearly the same direction. Quoting $\lambda \pm \sigma$ then
    misleads. Convert to a direction and quote the angular separation instead; the
    Justitia notebook does this.

!!! warning "Chain length"
    The posterior is sharp and strongly correlated, with integrated autocorrelation
    times of order 100 steps for 20 parameters. Check `posterior.autocorr_time`;
    a demonstration chain of a few hundred steps is not a converged one.
