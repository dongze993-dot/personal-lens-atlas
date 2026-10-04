"""Install verified Windows CPython 3.11 wheels without slow index scanning.

Some older pip builds spend several minutes parsing NumPy's enormous simple
index page before they discover that the newest NumPy release no longer has a
Python 3.11 wheel.  This helper asks PyPI's small JSON metadata endpoint,
chooses compatible prebuilt wheels, and asks pip to install those direct URLs
with dependency resolution disabled.  Dependency relationships are resolved
from the same package metadata before installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import InvalidVersion, Version


PYTHON_VERSION = Version("3.11.9")
PYPI_API = "https://pypi.org/pypi"
ROOT_REQUIREMENTS = (
    "mediapipe==1.0.1",
    "opencv-contrib-python==5.0.0.93",
    # NumPy 2.5+ has no CPython 3.11 wheel.  2.4.5 is the latest compatible
    # line verified for this project's Windows x64 CPython 3.11 target.
    "numpy==2.4.5",
)


class BootstrapError(RuntimeError):
    """A compatible wheel could not be selected or installed."""


@dataclass
class PackagePlan:
    name: str
    specifier: SpecifierSet = field(default_factory=SpecifierSet)
    selected_version: Version | None = None
    selected_url: str | None = None
    processed_dependencies: bool = False

    def accepts(self, version: Version) -> bool:
        return not str(self.specifier) or self.specifier.contains(version, prereleases=False)


def _json_from_pypi(path: str) -> dict[str, Any]:
    request = Request(
        f"{PYPI_API}/{path}",
        headers={"Accept": "application/json", "User-Agent": "gaze-correction-mvp/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _python_requirement_matches(value: str | None) -> bool:
    if not value:
        return True
    try:
        return SpecifierSet(value).contains(PYTHON_VERSION, prereleases=True)
    except Exception:
        # A malformed metadata marker should not make us choose an unsupported
        # wheel.  Let pip's later verification surface a clear error instead.
        return False


def _wheel_score(filename: str) -> int | None:
    """Rank only wheels that work on Windows x64 CPython 3.11."""

    lowered = filename.casefold()
    if lowered.endswith("-cp311-cp311-win_amd64.whl"):
        return 0
    abi3_match = re.search(r"-cp(\d+)-abi3-win_amd64\.whl$", lowered)
    if abi3_match and int(abi3_match.group(1)) <= 311:
        return 1
    if (
        lowered.endswith("none-any.whl")
        or lowered.endswith("none-win_amd64.whl")
    ) and ("-py3-" in lowered or "-py2.py3-" in lowered):
        return 2
    return None


class WheelResolver:
    def __init__(self) -> None:
        self._plans: dict[str, PackagePlan] = {}
        self._environment = default_environment()
        self._environment.update(
            {
                "implementation_name": "cpython",
                "implementation_version": "3.11.9",
                "platform_machine": "AMD64",
                "platform_system": "Windows",
                "python_full_version": "3.11.9",
                "python_version": "3.11",
                "sys_platform": "win32",
                "extra": "",
            }
        )

    def add_requirement(self, requirement_text: str) -> bool:
        requirement = Requirement(requirement_text)
        if requirement.marker and not requirement.marker.evaluate(self._environment):
            return False
        name = canonicalize_name(requirement.name)
        incoming = SpecifierSet(str(requirement.specifier))
        plan = self._plans.get(name)
        if plan is None:
            self._plans[name] = PackagePlan(name=name, specifier=incoming)
            return True

        merged = SpecifierSet(
            ",".join(part for part in (str(plan.specifier), str(incoming)) if part)
        )
        if str(merged) == str(plan.specifier):
            return False
        plan.specifier = merged
        if plan.selected_version is not None and not plan.accepts(plan.selected_version):
            plan.selected_version = None
            plan.selected_url = None
            plan.processed_dependencies = False
        return True

    def resolve(self) -> list[PackagePlan]:
        for root in ROOT_REQUIREMENTS:
            self.add_requirement(root)

        while True:
            changed = False
            for plan in list(self._plans.values()):
                if plan.selected_version is None:
                    self._select_wheel(plan)
                    changed = True
                if not plan.processed_dependencies:
                    details = _json_from_pypi(
                        f"{quote(plan.name)}/{quote(str(plan.selected_version))}/json"
                    )
                    for dependency in details["info"].get("requires_dist") or []:
                        changed = self.add_requirement(dependency) or changed
                    plan.processed_dependencies = True
                    changed = True
            if not changed:
                break

        return sorted(self._plans.values(), key=lambda item: item.name)

    @staticmethod
    def _select_wheel(plan: PackagePlan) -> None:
        project = _json_from_pypi(f"{quote(plan.name)}/json")
        releases = project.get("releases", {})
        candidates: list[tuple[Version, int, dict[str, Any]]] = []
        for version_text, files in releases.items():
            try:
                version = Version(version_text)
            except InvalidVersion:
                continue
            if not plan.accepts(version):
                continue
            for item in files:
                if item.get("yanked") or not _python_requirement_matches(
                    item.get("requires_python")
                ):
                    continue
                score = _wheel_score(item.get("filename", ""))
                if score is not None:
                    candidates.append((version, score, item))

        if not candidates:
            raise BootstrapError(
                f"No Windows x64 CPython 3.11 wheel satisfies {plan.name}{plan.specifier}."
            )
        version, _score, item = max(
            candidates,
            key=lambda candidate: (candidate[0], -candidate[1]),
        )
        plan.selected_version = version
        plan.selected_url = item["url"]


def _already_satisfies(plan: PackagePlan) -> bool:
    try:
        installed = Version(importlib.metadata.version(plan.name))
    except importlib.metadata.PackageNotFoundError:
        return False
    return plan.accepts(installed)


def install(plan: PackagePlan) -> None:
    if plan.selected_url is None or plan.selected_version is None:
        raise BootstrapError(f"Package {plan.name} was not resolved.")
    if _already_satisfies(plan):
        print(f"  [OK] {plan.name} {plan.selected_version}", flush=True)
        return

    print(f"  Installing {plan.name} {plan.selected_version}...", flush=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--only-binary=:all:",
        "--timeout",
        "60",
        "--retries",
        "2",
        plan.selected_url,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise BootstrapError(f"pip could not install {plan.name}.")


def main() -> int:
    print(
        "Resolving compatible prebuilt wheels for Windows x64 + Python 3.11...",
        flush=True,
    )
    resolver = WheelResolver()
    plans = resolver.resolve()
    print(
        f"Resolved {len(plans)} packages. Installing into the isolated environment...",
        flush=True,
    )
    for plan in plans:
        install(plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"\n[ERROR] {exc}", flush=True)
        raise SystemExit(1)
