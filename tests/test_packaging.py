"""Packaging invariants.

These are cheap and they would have caught three of the worst defects in the 0.2.x
releases: a distribution that declared no dependencies, a stdlib module listed as a
requirement, and three disagreeing version strings.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import requires, version
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _requirement_name(spec: str) -> str:
    """``"scikit-learn>=1.4"`` -> ``"scikit-learn"``."""
    for sep in (" ", ";", "[", "=", "<", ">", "!", "~"):
        spec = spec.split(sep)[0]
    return spec.strip().lower()


def test_dependencies_are_declared():
    """bug #18: the 0.2.1 distribution declared zero dependencies, so a fresh
    ``pip install eazypredict`` produced a package that could not be imported."""
    declared = requires("eazypredict") or []
    core = {_requirement_name(r) for r in declared if "extra ==" not in r}
    assert {"scikit-learn", "numpy", "pandas"} <= core


def test_no_stdlib_modules_declared_as_dependencies(pyproject):
    """bug #9: ``requirements.txt`` listed ``pickle``, which has no PyPI distribution,
    so ``pip install -r requirements.txt`` failed outright."""
    declared = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}
    assert not declared & set(sys.stdlib_module_names)


def test_version_is_single_sourced(pyproject):
    """bug #10: setup.cfg said 0.2.0, __init__.py said 0.1.2, the sdist said 0.2.1."""
    import eazypredict

    assert eazypredict.__version__ == version("eazypredict")
    assert eazypredict.__version__ == pyproject["project"]["version"]


def test_no_compiled_artifacts_tracked_in_git():
    """bug #30: a ``.pyc`` was committed to the repository."""
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert [f for f in tracked if f.endswith((".pyc", ".pyo"))] == []


def test_legacy_packaging_files_removed():
    """``setup.cfg`` and ``requirements.txt`` were replaced by ``pyproject.toml``;
    leaving them behind reintroduces the conflicting-metadata problem."""
    assert not (PROJECT_ROOT / "setup.cfg").exists()
    assert not (PROJECT_ROOT / "requirements.txt").exists()


def test_the_type_marker_ships(pyproject):
    """``package-data`` referenced a ``py.typed`` that did not exist, so the wheel
    carried the ``Typing :: Typed`` classifier while every downstream type checker
    ignored our annotations."""
    import eazypredict

    assert (Path(eazypredict.__file__).parent / "py.typed").is_file()
    assert "py.typed" in pyproject["tool"]["setuptools"]["package-data"]["eazypredict"]


def test_boosters_are_an_optional_extra(pyproject):
    """bug #19: xgboost and lightgbm must not be core requirements."""
    core = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}
    extras = pyproject["project"]["optional-dependencies"]
    assert not core & {"xgboost", "lightgbm", "matplotlib"}
    assert {"xgboost", "lightgbm"} == {_requirement_name(r) for r in extras["boosters"]}
