"""The API documentation is generated from docstrings, so the docstrings are tested.

`mkdocs` renders whatever it finds; it cannot tell that a parameter is
undocumented or that a section heading is misspelt.  These checks do, so a gap
fails the suite instead of quietly producing an empty table on the site.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import pytest

import lcinv

#: Numpydoc section headings that mkdocstrings understands.  A misspelt or
#: unsupported heading renders as body text rather than a table.
KNOWN_SECTIONS = {
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Notes",
    "Examples",
    "References",
    "See Also",
    "Warnings",
    "Other Parameters",
}


def _modules():
    return sorted(m.name for m in pkgutil.iter_modules(lcinv.__path__))


def _public_members(module):
    """Every name a module exports, plus the public methods of its classes."""
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if obj is None:
            continue
        yield name, obj
        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if attr.startswith("_"):
                    continue
                if inspect.isfunction(member) or isinstance(member, property):
                    yield f"{name}.{attr}", member


def _doc(obj) -> str:
    target = obj.fget if isinstance(obj, property) else obj
    return inspect.getdoc(target) or ""


@pytest.mark.parametrize("module_name", _modules())
class TestDocstringCoverage:
    def test_module_has_a_docstring(self, module_name):
        module = importlib.import_module(f"lcinv.{module_name}")
        assert (module.__doc__ or "").strip(), f"lcinv.{module_name} has no docstring"

    def test_every_public_object_has_a_docstring(self, module_name):
        module = importlib.import_module(f"lcinv.{module_name}")
        missing = [
            name for name, obj in _public_members(module) if not _doc(obj).strip()
        ]
        assert not missing, f"undocumented in lcinv.{module_name}: {missing}"

    def test_every_multi_argument_callable_documents_its_parameters(self, module_name):
        """A callable taking more than one argument needs a Parameters section.

        Dataclasses are exempt when they describe their fields under
        ``Attributes`` instead, which is what numpydoc prescribes for them.
        """
        module = importlib.import_module(f"lcinv.{module_name}")
        missing = []
        for name, obj in _public_members(module):
            doc = _doc(obj)
            if not doc.strip() or isinstance(obj, property):
                continue
            if inspect.isclass(obj) and (
                dataclasses.is_dataclass(obj) or "Attributes" in doc
            ):
                continue
            try:
                signature = inspect.signature(obj)
            except (ValueError, TypeError):  # pragma: no cover - C-level callables
                continue
            args = [p for p in signature.parameters if p not in ("self", "cls")]
            if len(args) > 1 and "Parameters" not in doc:
                missing.append(f"{name} ({len(args)} args)")
        assert not missing, f"no Parameters section in lcinv.{module_name}: {missing}"

    def test_section_headings_are_ones_mkdocstrings_understands(self, module_name):
        """Catch a misspelt heading, which would silently render as prose."""
        module = importlib.import_module(f"lcinv.{module_name}")
        bad = []
        for name, obj in _public_members(module):
            lines = _doc(obj).split("\n")
            for i, line in enumerate(lines[:-1]):
                underline = lines[i + 1].strip()
                heading = line.strip()
                if (
                    heading
                    and underline
                    and set(underline) == {"-"}
                    and len(underline) == len(heading)
                    and heading not in KNOWN_SECTIONS
                ):
                    bad.append(f"{name}: {heading!r}")
        assert not bad, f"unknown docstring sections in lcinv.{module_name}: {bad}"


def test_the_package_itself_is_documented():
    assert (lcinv.__doc__ or "").strip()


def test_every_module_is_in_a_documentation_group():
    """The API pages are generated from an explicit grouping; nothing may fall out."""
    import ast
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "docs" / "scripts" / "gen_api.py"
    # Read GROUPS declaratively rather than executing the script, which needs
    # mkdocs_gen_files and a live build context.
    tree = ast.parse(script.read_text())
    groups = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and any(
            getattr(t, "id", None) == "GROUPS"
            for t in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        )
    )
    grouped = {
        name
        for entry in ast.literal_eval(groups)
        for name in entry[1]
    }
    assert set(_modules()) <= grouped, (
        f"not in any GROUPS entry of gen_api.py: {sorted(set(_modules()) - grouped)}"
    )
