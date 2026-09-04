"""Lightcurve containers, file formats and the paper's normalisations.

The observational side of Eq. (2), ``L = A g``.  A :class:`Lightcurve` is one
observing run: brightnesses at a sequence of epochs together with the
asteroid-centric vectors towards the Sun and the Earth.  The distinction the
paper draws in Sections 3.4 and 3.5 between *absolute* and *relative*
photometry is carried on each curve, because it selects the objective
function:

* absolute data can use Eq. (5) or the renormalised Eq. (7);
* relative data must use Eq. (13), which "renormalises both the observed and
  the model lightcurves to mean brightnesses of unity".

Section 3.1 explains why the renormalisation matters at all:

    In practice, the observed brightnesses are usually several times smaller at
    large solar phase angles than near opposition.  Therefore it is
    advantageous to replace the standard chi^2 of (5) by a renormalized
    chi^2_ren [...] this form thus normalizes each lightcurve to oscillate
    around unity, giving each observing geometry equal weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import SpinState, phase_angle

__all__ = [
    "Lightcurve",
    "LightcurveSet",
    "optimal_scale",
    "SPEED_OF_LIGHT_AU_PER_DAY",
]

#: Light travel time is applied in units of AU per day.
SPEED_OF_LIGHT_AU_PER_DAY = 173.144632674240


@dataclass
class Lightcurve:
    """One lightcurve: brightnesses plus their observing geometry.

    Parameters
    ----------
    jd:
        ``(N,)`` light-time corrected Julian dates.
    brightness:
        ``(N,)`` brightnesses in intensity units (not magnitudes).
    sun:
        ``(N, 3)`` asteroid-centric ecliptic vector towards the Sun, in AU.
    earth:
        ``(N, 3)`` asteroid-centric ecliptic vector towards the Earth, in AU.
    calibrated:
        ``True`` for absolute photometry reduced to unit distances, ``False``
        for relative photometry.  DAMIT stores this as the ``0``/``1`` code on
        each lightcurve header line.
    name:
        Optional label carried through plots and reports.
    weight:
        Relative weight of this curve in the objective.  Eq. (13) sums over
        *points*, so a sparse curve with hundreds of points spread over years
        can dominate chi-squared while constraining the shape far less per
        point than a dense night does.  Paper II's remedy is to weight sparse
        and dense data separately; :meth:`LightcurveSet.balance_weights` sets
        these automatically.
    meta:
        Free-form metadata (references, filter, observer, ...).
    """

    jd: np.ndarray
    brightness: np.ndarray
    sun: np.ndarray
    earth: np.ndarray
    calibrated: bool = False
    name: str = ""
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.jd = np.asarray(self.jd, dtype=float).ravel()
        self.brightness = np.asarray(self.brightness, dtype=float).ravel()
        self.sun = np.atleast_2d(np.asarray(self.sun, dtype=float))
        self.earth = np.atleast_2d(np.asarray(self.earth, dtype=float))
        n = len(self.jd)
        if len(self.brightness) != n:
            raise ValueError("jd and brightness must have the same length")
        for label, arr in (("sun", self.sun), ("earth", self.earth)):
            if arr.shape != (n, 3):
                raise ValueError(f"{label} must have shape ({n}, 3)")
        if np.any(self.brightness <= 0):
            raise ValueError("brightnesses must be positive intensities")

    def __len__(self) -> int:
        return len(self.jd)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = "calibrated" if self.calibrated else "relative"
        return (
            f"Lightcurve({self.name or 'unnamed'}, {len(self)} pts, {kind}, "
            f"alpha={np.degrees(self.mean_phase_angle):.1f} deg)"
        )

    # ------------------------------------------------------------------
    @property
    def mean_brightness(self) -> float:
        """``L-bar``, the mean brightness of Eqs. (7) and (13)."""
        return float(self.brightness.mean())

    @property
    def normalised(self) -> np.ndarray:
        """``L / L-bar`` - the observed side of Eq. (13)."""
        return self.brightness / self.mean_brightness

    @property
    def phase_angles(self) -> np.ndarray:
        """``(N,)`` solar phase angles in radians."""
        return phase_angle(self.sun, self.earth)

    @property
    def mean_phase_angle(self) -> float:
        """Mean solar phase angle in radians."""
        return float(self.phase_angles.mean())

    @property
    def amplitude_mag(self) -> float:
        """Peak-to-peak amplitude in magnitudes."""
        return float(2.5 * np.log10(self.brightness.max() / self.brightness.min()))

    def unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalised ecliptic Sun and Earth directions."""
        s = self.sun / np.linalg.norm(self.sun, axis=1, keepdims=True)
        e = self.earth / np.linalg.norm(self.earth, axis=1, keepdims=True)
        return s, e

    def body_directions(self, spin: SpinState) -> tuple[np.ndarray, np.ndarray]:
        """Sun and Earth unit vectors ``E0`` and ``E`` in the body frame.

        This is the step that turns an observation into the ``mu`` and ``mu0``
        of Eq. (4), "for the observation i (in the asteroid's frame of
        reference)".
        """
        s, e = self.unit_vectors()
        return spin.to_asteroid_frame(s, self.jd), spin.to_asteroid_frame(e, self.jd)

    def light_time_corrected(self) -> Lightcurve:
        """Copy with epochs reduced by the asteroid-Earth light travel time.

        DAMIT epochs are already corrected, so this is only for data brought in
        from elsewhere (ALCDEF, an observer's own file).
        """
        delta = np.linalg.norm(self.earth, axis=1)
        return Lightcurve(
            self.jd - delta / SPEED_OF_LIGHT_AU_PER_DAY,
            self.brightness,
            self.sun,
            self.earth,
            self.calibrated,
            self.name,
            self.weight,
            dict(self.meta, light_time_corrected=True),
        )

    def with_noise(self, level: float, seed: int | None = None) -> Lightcurve:
        """Copy with multiplicative Gaussian noise of relative size ``level``.

        Section 3.5 reports that "even considerable noise (from 5 to 10%) [...]
        does not cause a need for regularization".
        """
        rng = np.random.default_rng(seed)
        factor = 1.0 + level * rng.standard_normal(len(self))
        return Lightcurve(
            self.jd,
            self.brightness * np.maximum(factor, 1e-6),
            self.sun,
            self.earth,
            self.calibrated,
            self.name,
            self.weight,
            dict(self.meta, noise=level),
        )


class LightcurveSet:
    """An ordered collection of :class:`Lightcurve` objects.

    Section 3.5 sets the requirement on such a set: "they suffice so long as
    they cover a wide range of observing geometries and there are sufficiently
    many lightcurve points [...] there should be at least a few lightcurves
    with alpha greater than, say, 20 degrees".  :meth:`summary` reports
    exactly those numbers.
    """

    def __init__(self, curves: list[Lightcurve] | tuple[Lightcurve, ...] = ()) -> None:
        self.curves: list[Lightcurve] = list(curves)

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self):
        return iter(self.curves)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return LightcurveSet(self.curves[item])
        return self.curves[item]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LightcurveSet({len(self)} curves, {self.n_points} points)"

    def append(self, curve: Lightcurve) -> None:
        """Add one curve."""
        self.curves.append(curve)

    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Total number of photometric points."""
        return int(sum(len(c) for c in self.curves))

    @property
    def brightness(self) -> np.ndarray:
        """All brightnesses concatenated."""
        return np.concatenate([c.brightness for c in self.curves]) if self.curves else np.zeros(0)

    @property
    def counts(self) -> np.ndarray:
        """``(M,)`` number of points in each curve."""
        return np.asarray([len(c) for c in self.curves], dtype=np.int64)

    @property
    def offsets(self) -> np.ndarray:
        """``(M + 1,)`` start index of each curve in the concatenated arrays."""
        return np.concatenate([[0], np.cumsum(self.counts)])

    @property
    def phase_angles(self) -> np.ndarray:
        """All phase angles concatenated, in radians."""
        return np.concatenate([c.phase_angles for c in self.curves])

    @property
    def all_calibrated(self) -> bool:
        """True when every curve is absolute photometry."""
        return bool(self.curves) and all(c.calibrated for c in self.curves)

    def body_directions(self, spin: SpinState) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated body-frame Sun and Earth unit vectors for the whole set."""
        pairs = [c.body_directions(spin) for c in self.curves]
        return (
            np.vstack([p[0] for p in pairs]),
            np.vstack([p[1] for p in pairs]),
        )

    def filter(
        self,
        min_points: int = 0,
        min_phase_deg: float | None = None,
        max_phase_deg: float | None = None,
        calibrated: bool | None = None,
    ) -> LightcurveSet:
        """Select a subset by point count, phase-angle range or calibration."""
        out = []
        for c in self.curves:
            if len(c) < min_points:
                continue
            a = np.degrees(c.mean_phase_angle)
            if min_phase_deg is not None and a < min_phase_deg:
                continue
            if max_phase_deg is not None and a > max_phase_deg:
                continue
            if calibrated is not None and c.calibrated is not calibrated:
                continue
            out.append(c)
        return LightcurveSet(out)

    def select_geometries(self, n: int, seed: int | None = 0) -> LightcurveSet:
        """Pick ``n`` curves spread as widely as possible in observing geometry.

        Greedy farthest-point selection on the mean phase-angle-bisector
        direction and phase angle, so the result honours Section 3.5's "wide
        range of observing geometries" rather than clustering on one
        apparition.
        """
        if n >= len(self.curves):
            return LightcurveSet(self.curves)
        feats = []
        for c in self.curves:
            s, e = c.unit_vectors()
            bis = (s + e).mean(axis=0)
            bis /= max(np.linalg.norm(bis), 1e-12)
            feats.append(np.concatenate([bis, [c.mean_phase_angle]]))
        feats = np.asarray(feats)
        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(len(feats)))]
        while len(chosen) < n:
            d = np.linalg.norm(feats[:, None, :] - feats[None, chosen, :], axis=2).min(axis=1)
            d[chosen] = -1.0
            chosen.append(int(np.argmax(d)))
        return LightcurveSet([self.curves[i] for i in sorted(chosen)])

    @property
    def weights(self) -> np.ndarray:
        """``(M,)`` per-curve weights."""
        return np.asarray([c.weight for c in self.curves], dtype=float)

    def balance_weights(
        self, sparse_alpha_span_deg: float = 2.0, mode: str = "per_curve"
    ) -> "LightcurveSet":
        """Set per-curve weights so that no one curve dominates chi-squared.

        Parameters
        ----------
        sparse_alpha_span_deg:
            A curve whose solar phase angle spans more than this is treated as
            *sparse* (survey photometry spanning many apparitions) rather than
            a dense single-night run.
        mode:
            ``"per_curve"`` gives every lightcurve the same total weight,
            ``1 / n_points``, so each observing geometry counts once - the
            spirit of Eq. (7)'s "democratization".  ``"sparse"`` leaves dense
            curves at unit weight and down-weights only the sparse ones by
            their point count.  ``"none"`` resets every weight to one.

        Returns
        -------
        LightcurveSet
            The same curves with :attr:`Lightcurve.weight` set.
        """
        if mode not in ("per_curve", "sparse", "none"):
            raise ValueError("mode must be 'per_curve', 'sparse' or 'none'")
        for curve in self.curves:
            if mode == "none":
                curve.weight = 1.0
                continue
            span = float(np.degrees(np.ptp(curve.phase_angles)))
            sparse = span > sparse_alpha_span_deg
            if mode == "per_curve" or sparse:
                curve.weight = 1.0 / max(len(curve), 1)
            else:
                curve.weight = 1.0
        return self

    def summary(self) -> dict:
        """Descriptive statistics, keyed for printing or logging."""
        if not self.curves:
            return {"n_curves": 0, "n_points": 0}
        alpha = np.degrees([c.mean_phase_angle for c in self.curves])
        return {
            "n_curves": len(self.curves),
            "n_points": self.n_points,
            "n_calibrated": int(sum(c.calibrated for c in self.curves)),
            "phase_min_deg": float(alpha.min()),
            "phase_max_deg": float(alpha.max()),
            "n_above_20deg": int((alpha > 20.0).sum()),
            "jd_min": float(min(c.jd.min() for c in self.curves)),
            "jd_max": float(max(c.jd.max() for c in self.curves)),
            "max_amplitude_mag": float(max(c.amplitude_mag for c in self.curves)),
        }

    def with_noise(self, level: float, seed: int | None = 0) -> LightcurveSet:
        """Copy with independent noise added to every curve."""
        rng = np.random.default_rng(seed)
        return LightcurveSet(
            [c.with_noise(level, int(rng.integers(1 << 31))) for c in self.curves]
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_damit_txt(cls, path: str | Path) -> LightcurveSet:
        """Read a DAMIT ``lc.txt`` export.

        The format is: total number of lightcurves; then per curve a header
        line ``npoints code`` with ``code`` 0 for relative and 1 for
        calibrated, followed by ``npoints`` lines of light-time corrected JD,
        brightness, and the ecliptic asteroid-centric ``x, y, z`` of the Sun
        and of the Earth in AU.
        """
        text = Path(path).read_text().split()
        pos = 0
        n_curves = int(text[pos])
        pos += 1
        curves = []
        for i in range(n_curves):
            n_pts, code = int(text[pos]), int(text[pos + 1])
            pos += 2
            block = np.asarray(text[pos : pos + 8 * n_pts], dtype=float).reshape(n_pts, 8)
            pos += 8 * n_pts
            curves.append(
                Lightcurve(
                    jd=block[:, 0],
                    brightness=block[:, 1],
                    sun=block[:, 2:5],
                    earth=block[:, 5:8],
                    calibrated=bool(code),
                    name=f"lc{i + 1:03d}",
                )
            )
        return cls(curves)

    def to_damit_txt(self, path: str | Path) -> None:
        """Write the set in DAMIT ``lc.txt`` form."""
        lines = [str(len(self.curves))]
        for c in self.curves:
            lines.append(f"{len(c)} {int(c.calibrated)}")
            for k in range(len(c)):
                lines.append(
                    f"{c.jd[k]:.6f} {c.brightness[k]:.6e} "
                    + " ".join(f"{v:.8f}" for v in c.sun[k])
                    + " "
                    + " ".join(f"{v:.8f}" for v in c.earth[k])
                )
        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def from_damit_json(cls, path: str | Path) -> LightcurveSet:
        """Read a DAMIT ``lc.json`` export, keeping its per-curve metadata.

        The JSON export is a list of ``{"LightCurve": {...}, "Reference":
        [...]}`` records.  ``LightCurve.points`` holds the same eight columns
        as :meth:`from_damit_txt` but as one whitespace-separated string, and
        ``scale`` is the ``0``/``1`` relative/calibrated code.  Unlike the
        plaintext export this form carries the bibliographic references, which
        are kept in :attr:`Lightcurve.meta`.
        """
        raw = json.loads(Path(path).read_text())
        entries = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        curves = []
        for i, record in enumerate(entries):
            entry = record.get("LightCurve", record)
            block = np.asarray(str(entry["points"]).split(), dtype=float).reshape(-1, 8)
            meta = {k: v for k, v in entry.items() if k != "points"}
            refs = record.get("Reference") or []
            if refs:
                meta["references"] = [
                    {k: r.get(k) for k in ("bibcode", "author_short", "year", "title")}
                    for r in refs
                ]
            curves.append(
                Lightcurve(
                    jd=block[:, 0],
                    brightness=block[:, 1],
                    sun=block[:, 2:5],
                    earth=block[:, 5:8],
                    calibrated=bool(int(entry.get("scale", 0) or 0)),
                    name=str(entry.get("display_label") or entry.get("id") or f"lc{i + 1:03d}"),
                    meta=meta,
                )
            )
        return cls(curves)


def optimal_scale(observed: np.ndarray, model: np.ndarray) -> float:
    """Eq. (14) - the scale factor that best matches ``model`` to ``observed``.

    Section 3.5 rescales the convex hull's lightcurves before comparing them,
    "finding a scale coefficient such that the lightcurves of the two shapes
    are as similar as possible (this corresponds to a simple shrinking of the
    convex hull)".  That coefficient minimises ``|L - c L_ch|**2``, giving

    .. math::  c = \\frac{\\sum_i L_i L_{ch,i}}{\\sum_i L_{ch,i}^2}

    and, per the paper, is "computed from all lightcurve data (not for each
    lightcurve separately)".

    Parameters
    ----------
    observed:
        Reference brightnesses ``L``.
    model:
        Brightnesses to be rescaled, ``L_ch``.

    Returns
    -------
    float
    """
    obs = np.asarray(observed, dtype=float).ravel()
    mod = np.asarray(model, dtype=float).ravel()
    if obs.shape != mod.shape:
        raise ValueError("observed and model must have the same shape")
    denom = float(mod @ mod)
    if denom <= 0:
        raise ValueError("model lightcurve is identically zero")
    return float(obs @ mod) / denom
