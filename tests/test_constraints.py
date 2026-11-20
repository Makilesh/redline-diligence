"""
constraints.txt pins what the image and CI install to the versions the project was measured
with (scripts/update_constraints.py). These checks keep the pins, the requirement ranges and
the separately pinned torch from drifting apart; a drift would otherwise surface as a failed
image build on a 2-core VM, or as an image that silently differs from what was tested.
"""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> list[Requirement]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            reqs += _requirements(path.parent / line[3:].strip())
            continue
        reqs.append(Requirement(line))
    return reqs


def _pins() -> dict[str, str]:
    pins = {}
    for raw in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            assert version and not re.search(r"[<>=!~*,; ]", version), f"not an exact pin: {raw!r}"
            pins[canonicalize_name(name)] = version
    return pins


def test_every_requirement_is_pinned_inside_its_own_range():
    pins = _pins()
    reqs = _requirements(ROOT / "requirements-dev.txt")
    assert len(reqs) > 25  # 34 direct requirements on 2026-11-20
    for req in reqs:
        name = canonicalize_name(req.name)
        if name == "torch":
            continue
        assert name in pins, f"{req.name} has no pin in constraints.txt (scripts/update_constraints.py)"
        assert req.specifier.contains(pins[name], prereleases=True), \
            f"constraints.txt pins {req.name}=={pins[name]}, outside requirements' {req.specifier}"


def test_torch_and_its_cuda_wheels_are_left_to_the_cpu_install():
    pins = _pins()
    assert not [n for n in pins if n == "torch" or n == "triton" or n.startswith(("nvidia-", "cuda-"))]


def test_the_image_and_both_ci_jobs_install_the_same_torch_and_use_the_constraints():
    dockerfile = (ROOT / "Dockerfile.hf").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    torch = re.search(r"^ARG TORCH_VERSION=(\S+)$", dockerfile, re.M).group(1)
    ci_torch = re.findall(r"pip install torch==(\S+) --index-url https://download.pytorch.org/whl/cpu", ci)
    assert ci_torch and set(ci_torch) == {torch}, (torch, ci_torch)
    assert "-c constraints.txt -r requirements.txt" in dockerfile
    assert "COPY requirements.txt constraints.txt ./" in dockerfile
    installs = re.findall(r"run: pip install (.*-r requirements-dev\.txt)", ci)
    assert installs and all("-c constraints.txt" in cmd for cmd in installs), installs
