# References

The two papers this package implements, plus pointers to the original code and
the data source.

## The papers

| File | Citation |
|---|---|
| `Kaasalainen_Torppa_2001_Icarus_153_24_PaperI.pdf` | M. Kaasalainen & J. Torppa (2001), *Optimization Methods for Asteroid Lightcurve Inversion. I. Shape Determination*, Icarus **153**, 24–36. [doi:10.1006/icar.2001.6673](https://doi.org/10.1006/icar.2001.6673) |
| `Kaasalainen_Torppa_Muinonen_2001_Icarus_153_37_PaperII.pdf` | M. Kaasalainen, J. Torppa & K. Muinonen (2001), *Optimization Methods for Asteroid Lightcurve Inversion. II. The Complete Inverse Problem*, Icarus **153**, 37–51. [doi:10.1006/icar.2001.6674](https://doi.org/10.1006/icar.2001.6674) |

**Paper I** is the shape problem with the rotation state given, and is what
`lcinv` implements equation by equation. **Paper II** closes the loop — pole,
period and scattering parameters — and ends with the eight-step recipe that
`lcinv.pipeline` follows.

These PDFs are the publisher's copyright and are kept here for convenience.

## Prior work cited by Paper I

Referred to throughout as KLLB and KLL, and the source of the Minkowski
machinery in Appendix C:

* Kaasalainen, M., Lamberg, L., Lumme, K. & Bowell, E. (1992a). *Interpretation of
  lightcurves of atmosphereless bodies. I.* A&A **259**, 318–332. — "KLLB"
* Kaasalainen, M., Lamberg, L. & Lumme, K. (1992b). *Interpretation of lightcurves
  of atmosphereless bodies. II.* A&A **259**, 333–340. — "KLL"
* Lamberg, L. (1993). *On the Minkowski Problem and the Lightcurve Operator.*
  Ann. Acad. Sci. Fenn. Ser. A I Math. Diss. **87**.
* Muinonen, K. (1998). *Introducing the Gaussian shape hypothesis for asteroids and
  comets.* A&A **332**, 1087–1098. — the Gaussian random spheres of
  `lcinv.gaussian_random_sphere`.
* Hudson, R. S. & Ostro, S. J. (1994). *Shape of Asteroid 4769 Castalia from
  inversion of radar images.* Science **263**, 940–943. — the body behind Section
  3.5's shape 2.

## The original implementation

Kaasalainen's Fortran, translated to C by Josef Ďurech, is distributed by DAMIT:

* <https://damit.cuni.cz/projects/damit/pages/software_download>

It provides `convexinv` (Section 3.2 plus Paper II), `conjgradinv` (Section 3.1)
and `lcgenerator` (Section 2). `lcinv` is validated against `convexinv` on its own
shipped test data — see the table in the top-level README and
`tests/test_convex.py::TestAgainstReferenceImplementation`. That dataset is
committed as `tests/data/test_lcs_rel.txt`.

## Data

* Ďurech, J., Sidorin, V. & Kaasalainen, M. (2010). *DAMIT: a database of asteroid
  models.* A&A **513**, A46. [doi:10.1051/0004-6361/200912693](https://doi.org/10.1051/0004-6361/200912693)
* DAMIT: <https://damit.cuni.cz> — content licensed CC BY 4.0.

The worked example uses [model 4966](https://damit.cuni.cz/asteroid_models/view/4966),
(269) Justitia. Its lightcurves carry their own bibliography, preserved by
`DamitClient.lightcurves(..., fmt="json")` in `curve.meta["references"]`.
