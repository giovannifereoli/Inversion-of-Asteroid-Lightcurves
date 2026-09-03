"""Client for DAMIT, the Database of Asteroid Models from Inversion Techniques.

DAMIT (https://damit.cuni.cz) publishes the lightcurves and the derived models
for well over a thousand asteroids, in exactly the form this package needs:
brightnesses with the asteroid-centric ecliptic vectors to the Sun and Earth
already attached and already light-time corrected.  Its models are themselves
made "using the light-curve inversion method developed by Kaasalainen & Torppa
(2001) and Kaasalainen et al. (2001)", so a DAMIT model is the natural
reference to check an inversion against.

Everything is cached on disk, so a script can be re-run offline.

Licence note: DAMIT content is released under CC BY 4.0.  Cite Ďurech,
Sidorin & Kaasalainen (2010), A&A 513, A46, and the per-model references that
:meth:`DamitClient.lightcurves` preserves in each curve's metadata.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import SpinState
from .lightcurve import LightcurveSet
from .mesh import Polyhedron
from .scattering import ScatteringLaw, make_scattering_law

__all__ = ["DamitClient", "DamitModel", "compare_poles", "DEFAULT_CACHE"]

#: Where downloads are kept unless a different directory is given.
DEFAULT_CACHE = Path("data/damit")


@dataclass
class DamitModel:
    """A DAMIT model: its rotation state, scattering law and shape.

    Attributes
    ----------
    model_id:
        DAMIT ``asteroid_models.id``.
    asteroid_id:
        DAMIT ``asteroids.id``, which is what the lightcurve export is keyed on.
    number, name:
        Minor-planet number and name, where known.
    spin:
        Rotation state from ``spin.txt``.
    law:
        Scattering law from the model's ``lsm`` code and parameters.
    shape:
        The published polyhedron, or ``None`` if it was not downloaded.
    record:
        The raw row from the ``asteroid_models`` table.
    """

    model_id: int
    asteroid_id: int
    number: str
    name: str
    spin: SpinState
    law: ScatteringLaw
    shape: Polyhedron | None
    record: dict

    @property
    def quality_flag(self) -> float | None:
        """DAMIT's 0-5 reliability flag, or ``None`` when unset."""
        v = self.record.get("quality_flag")
        return float(v) if v else None

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. ``"(269) Justitia [model 4966]"``."""
        head = f"({self.number}) {self.name}".strip() if self.number else self.name
        return f"{head or 'unnamed'} [model {self.model_id}]"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DamitModel({self.label}, lambda={self.spin.lam}, beta={self.spin.beta}, "
            f"P={self.spin.period} h)"
        )


class DamitClient:
    """Download and cache DAMIT tables, lightcurves and models.

    Parameters
    ----------
    cache_dir:
        Directory for downloaded files.
    timeout:
        Per-request timeout in seconds.
    retries:
        Number of attempts per request.
    base_url:
        Override the service root (useful for mirrors or tests).
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE,
        timeout: float = 120.0,
        retries: int = 3,
        base_url: str = "https://damit.cuni.cz/projects/damit",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    def _fetch(self, path: str, filename: str, refresh: bool = False) -> Path:
        """Download ``path`` into the cache, returning the local file."""
        target = self.cache_dir / filename
        if target.exists() and not refresh and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{path.lstrip('/')}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "lcinv (asteroid lightcurve inversion)"}
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                if not payload:
                    raise OSError("empty response")
                target.write_bytes(payload)
                return target
            except (urllib.error.URLError, OSError, TimeoutError) as exc:  # pragma: no cover
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise ConnectionError(f"could not download {url}: {last}")  # pragma: no cover

    def _table(self, name: str, refresh: bool = False) -> list[dict]:
        path = self._fetch(f"exports/table/{name}", f"{name}.csv", refresh)
        # The exports carry a UTF-8 byte-order mark.
        text = path.read_text(encoding="utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))

    # ------------------------------------------------------------------
    def asteroids(self, refresh: bool = False) -> list[dict]:
        """The ``asteroids`` table."""
        return self._table("asteroids", refresh)

    def models(self, refresh: bool = False) -> list[dict]:
        """The ``asteroid_models`` table."""
        return self._table("asteroid_models", refresh)

    def find_asteroid(
        self, number: int | str | None = None, name: str | None = None
    ) -> dict:
        """Look up one asteroid by minor-planet number or by name.

        Parameters
        ----------
        number:
            Permanent number, e.g. ``269``.
        name:
            Permanent name, e.g. ``"Justitia"`` (case-insensitive).

        Returns
        -------
        dict
            The matching row of the ``asteroids`` table.
        """
        if number is None and name is None:
            raise ValueError("give a number or a name")
        rows = self.asteroids()
        for row in rows:
            if number is not None and str(row["number"]) == str(number):
                return row
            if name is not None and (row["name"] or "").lower() == name.lower():
                return row
        raise LookupError(f"no DAMIT asteroid matching number={number!r} name={name!r}")

    def models_for(self, asteroid_id: int | str) -> list[dict]:
        """Every model belonging to one asteroid, best quality flag first."""
        rows = [m for m in self.models() if str(m["asteroid_id"]) == str(asteroid_id)]
        return sorted(rows, key=lambda r: -float(r["quality_flag"] or 0))

    # ------------------------------------------------------------------
    def lightcurves(
        self, asteroid_id: int | str, fmt: str = "txt", refresh: bool = False
    ) -> LightcurveSet:
        """Download an asteroid's lightcurves.

        Parameters
        ----------
        asteroid_id:
            DAMIT ``asteroids.id`` - *not* the minor-planet number and not the
            model id.  :meth:`find_asteroid` returns it.
        fmt:
            ``"txt"`` for the plaintext export, or ``"json"`` to also pick up
            each curve's bibliographic references.
        refresh:
            Re-download even if a cached copy exists.

        Returns
        -------
        LightcurveSet
        """
        if fmt == "txt":
            path = self._fetch(
                f"light_curves/exportAllForAsteroid/{asteroid_id}/plaintext",
                f"{asteroid_id}.lc.txt", refresh,
            )
            return LightcurveSet.from_damit_txt(path)
        if fmt == "json":
            path = self._fetch(
                f"light_curves/exportAllForAsteroid/{asteroid_id}/json",
                f"{asteroid_id}.lc.json", refresh,
            )
            return LightcurveSet.from_damit_json(path)
        raise ValueError("fmt must be 'txt' or 'json'")

    def spin(self, model_id: int | str, refresh: bool = False) -> tuple[SpinState, ScatteringLaw]:
        """Read a model's ``spin.txt``.

        The first line holds ``lambda``, ``beta`` and ``P``; the second ``t0``
        and ``phi0``; the third, when present, the scattering parameters
        ``p1`` to ``p5``.
        """
        path = self._fetch(
            f"generated_files/open/AsteroidModel/{model_id}/spin.txt",
            f"{model_id}.spin.txt", refresh,
        )
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        lam, beta, period = (float(x) for x in lines[0].split()[:3])
        t0, phi0 = (float(x) for x in lines[1].split()[:2])
        record = self._model_record(model_id)
        params = [x for x in lines[2].split()] if len(lines) > 2 else []
        code = (record.get("lsm") or "LSL").strip() or "LSL"
        law = make_scattering_law(code, [float(p) for p in params] if params else None)
        return SpinState(lam, beta, period, t0, phi0, float(record.get("yorp") or 0.0)), law

    def shape(self, model_id: int | str, refresh: bool = False) -> Polyhedron:
        """Download a model's shape as a :class:`~lcinv.mesh.Polyhedron`."""
        path = self._fetch(
            f"generated_files/open/AsteroidModel/{model_id}/shape.obj",
            f"{model_id}.shape.obj", refresh,
        )
        return Polyhedron.from_obj(path)

    def _model_record(self, model_id: int | str) -> dict:
        for row in self.models():
            if str(row["id"]) == str(model_id):
                return row
        raise LookupError(f"no DAMIT model with id {model_id}")

    def model(self, model_id: int | str, with_shape: bool = True) -> DamitModel:
        """Assemble a complete :class:`DamitModel`.

        Parameters
        ----------
        model_id:
            DAMIT ``asteroid_models.id``, the number in a
            ``/asteroid_models/view/<id>`` URL.
        with_shape:
            Also download the published polyhedron.

        Returns
        -------
        DamitModel
        """
        record = self._model_record(model_id)
        spin, law = self.spin(model_id)
        asteroid = next(
            (a for a in self.asteroids() if str(a["id"]) == str(record["asteroid_id"])), {}
        )
        return DamitModel(
            model_id=int(model_id),
            asteroid_id=int(record["asteroid_id"]),
            number=str(asteroid.get("number") or ""),
            name=str(asteroid.get("name") or asteroid.get("designation") or ""),
            spin=spin,
            law=law,
            shape=self.shape(model_id) if with_shape else None,
            record=record,
        )

    def bundle(
        self, model_id: int | str, fmt: str = "json"
    ) -> tuple[DamitModel, LightcurveSet]:
        """A model together with its asteroid's lightcurves.

        This is the one call an example script needs.

        Returns
        -------
        model, data
        """
        model = self.model(model_id)
        return model, self.lightcurves(model.asteroid_id, fmt=fmt)


def compare_poles(a: SpinState, b: SpinState) -> float:
    """Angle in degrees between two spin axes."""
    cos = float(np.clip(a.pole_vector() @ b.pole_vector(), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
