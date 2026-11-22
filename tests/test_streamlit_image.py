"""
The Streamlit image contains what the console imports, and the API needs none of it.

Dockerfile.streamlit copied only app/ and .streamlit/, while the console has imported
src.utils.admin_client since 2026-09-24: in the image every page failed with
"No module named 'src'" and the container's health check (Streamlit's own server)
still passed. These tests rebuild the image's file layout from the Dockerfile's COPY
lines and import the console there, and check the package split
(requirements.txt for the API, requirements-ui.txt for the console).
"""
from __future__ import annotations

import ast
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.streamlit"


def _join(base: str, name: str) -> str:
    return f"{base}/{name}" if base else name


def _copies() -> list[tuple[str, str]]:
    """(repo-relative source file, path inside the image's WORKDIR) for every COPY line."""
    out = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY":
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        *sources, dest = args
        into_dir = dest.endswith("/") or dest in (".", "./") or len(sources) > 1
        base = dest.removeprefix("./").rstrip("/")
        base = "" if base == "." else base
        for src in sources:
            path = ROOT / src
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and "__pycache__" not in f.parts:
                        out.append((f.relative_to(ROOT).as_posix(), _join(base, f.relative_to(path).as_posix())))
            else:
                out.append((src.removeprefix("./"), _join(base, path.name) if into_dir else base))
    return out


def _imports(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        for f in [path] if path.is_file() else path.rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module)
    return names


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(canonicalize_name(Requirement(line).name))
    return names


def test_every_copy_lands_where_the_repository_has_it():
    # The console finds app/ and src/ through its project root, so the image must keep
    # the repository's layout under WORKDIR.
    copies = _copies()
    assert copies
    for source, target in copies:
        assert source == target, f"{source} is copied to {target}"


def test_every_src_module_the_console_imports_is_in_the_image():
    copied = {target for _, target in _copies()}
    needed = sorted(n for n in _imports([ROOT / "app"]) if n == "src" or n.startswith("src."))
    assert needed, "the console no longer imports src: drop the src COPY lines"
    for module in needed:
        parts = module.split(".")
        for i in range(1, len(parts)):  # every enclosing package
            assert "/".join(parts[:i]) + "/__init__.py" in copied, f"{module}: package {parts[:i]} not copied"
        as_file = "/".join(parts) + ".py"
        as_pkg = "/".join(parts) + "/__init__.py"
        assert as_file in copied or as_pkg in copied, f"{module} is imported by app/ but not copied"


def test_the_console_imports_in_the_image_layout(tmp_path):
    for source, target in _copies():
        dest = tmp_path / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, dest)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["API_URL"] = "http://127.0.0.1:9"  # nothing listens there; no request leaves the machine
    proc = subprocess.run(
        [sys.executable, "-c", "import app.streamlit_app"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_the_console_needs_only_what_requirements_ui_lists():
    # The console's own code and the src modules the image copies for it, lazy imports included.
    copied_src = [ROOT / source for source, _ in _copies() if source.startswith("src/") and source.endswith(".py")]
    assert copied_src
    stdlib = set(sys.stdlib_module_names)
    third_party = {n.split(".")[0] for n in _imports([ROOT / "app", *copied_src])} - stdlib - {"app", "src", "__future__"}
    owners = md.packages_distributions()
    listed = _requirement_names(ROOT / "requirements-ui.txt")
    for module in sorted(third_party):
        dists = {canonicalize_name(d) for d in owners.get(module, [])}
        assert dists & listed, f"app/ imports {module} ({dists or 'no distribution'}), absent from requirements-ui.txt"


def test_the_api_does_not_import_the_console_packages():
    api_imports = {n.split(".")[0] for n in _imports([ROOT / "api", ROOT / "src", ROOT / "run_api.py"])}
    assert not api_imports & {"streamlit", "pyarrow"}
    assert not _requirement_names(ROOT / "requirements.txt") & {"streamlit", "pyarrow"}


def test_the_ui_image_installs_its_requirements_at_the_tested_versions():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"COPY requirements-ui\.txt constraints\.txt \./", text)
    assert "--only-binary :all: -c constraints.txt -r requirements-ui.txt" in text
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements-ui.txt" in dev, "CI installs requirements-dev.txt and must test the console"
