"""Generate one API page per module, at build time.

Run by ``mkdocs-gen-files``.  Walking the package rather than listing modules by
hand means a new module gets a documented page automatically, and a deleted one
cannot leave a stale page behind.
"""

from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = "lcinv"

#: Ordered groups, so the navigation follows the paper rather than the alphabet.
GROUPS: list[tuple[str, list[str]]] = [
    ("Observations", ["lightcurve", "damit"]),
    ("Geometry", ["geometry", "mesh", "triangulation", "convexhull", "sphharm"]),
    ("Physics", ["scattering", "raytracer", "shapes"]),
    ("Inversion", ["convex", "albedo", "nonconvex", "minkowski"]),
    ("Workflow", ["pipeline", "bayes"]),
    ("Presentation", ["plotting", "style", "cli"]),
]

modules = sorted(p.stem for p in (SRC / PACKAGE).glob("*.py") if p.stem != "__init__")
grouped = {name for _, names in GROUPS for name in names}
missing = [m for m in modules if m not in grouped]
if missing:  # a new module must not silently vanish from the docs
    GROUPS.append(("Other", sorted(missing)))

nav_lines = ["* [Overview](index.md)"]
for group, names in GROUPS:
    nav_lines.append(f"* {group}")
    for name in names:
        if name not in modules:
            continue
        doc_path = Path("api", f"{name}.md")
        with mkdocs_gen_files.open(doc_path, "w") as fd:
            print(f"# `{PACKAGE}.{name}`\n", file=fd)
            print(f"::: {PACKAGE}.{name}", file=fd)
        mkdocs_gen_files.set_edit_path(doc_path, Path("src", PACKAGE, f"{name}.py"))
        # SUMMARY.md sits inside api/, so its links are relative to it.
        nav_lines.append(f"    * [{PACKAGE}.{name}]({name}.md)")

with mkdocs_gen_files.open("api/index.md", "w") as fd:
    print("# API reference\n", file=fd)
    print(
        "Generated from the package's own docstrings. Every page below "
        "corresponds to one module of `lcinv`; the grouping follows the "
        "structure of the paper rather than the alphabet.\n",
        file=fd,
    )
    for group, names in GROUPS:
        present = [n for n in names if n in modules]
        if not present:
            continue
        print(f"## {group}\n", file=fd)
        for name in present:
            module = __import__(f"{PACKAGE}.{name}", fromlist=["_"])
            summary = (module.__doc__ or "").strip().split("\n")[0]
            print(f"- [`{PACKAGE}.{name}`]({name}.md) — {summary}", file=fd)
        print(file=fd)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as fd:
    fd.write("\n".join(nav_lines) + "\n")
