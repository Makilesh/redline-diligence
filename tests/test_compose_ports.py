"""
The base docker-compose.yml is for one machine: its Postgres password defaults to
`password` and Qdrant has no authentication, so every port it publishes is bound
to 127.0.0.1. Published on all interfaces, any networked host running the README's
quick start exposed both. The production overlay publishes nothing but Caddy.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ports(path: Path) -> dict[str, list[str]]:
    services = yaml.safe_load(path.read_text(encoding="utf-8"))["services"]
    return {name: [str(p) for p in (svc.get("ports") or [])] for name, svc in services.items()}


def test_the_base_compose_file_publishes_on_localhost_only():
    published = _ports(ROOT / "docker-compose.yml")
    assert any(published.values()), "expected the base file to publish ports"
    for service, ports in published.items():
        for port in ports:
            assert port.startswith("127.0.0.1:"), f"{service} publishes {port} on every interface"
