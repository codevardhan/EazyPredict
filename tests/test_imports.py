"""Import-path integrity.

`eazypredict.EazyClassifier` is both a class exported from the package and, in the
0.2.x layout, a module. If the submodule were imported normally, Python would rebind
the package attribute to the *module*, so whether `from eazypredict import
EazyClassifier` gives you a class or a module would depend on import order - a
spectacularly confusing failure. These run in subprocesses because the bug is order
dependent and a single in-process test would be contaminated by earlier imports.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ORDERS = {
    "legacy_first": (
        "import eazypredict.EazyClassifier\nfrom eazypredict import EazyClassifier\n"
    ),
    "new_first": (
        "from eazypredict import EazyClassifier\nimport eazypredict.EazyClassifier\n"
    ),
    "legacy_only": "from eazypredict.EazyClassifier import EazyClassifier as _\n",
    "new_only": "from eazypredict import EazyClassifier\n",
}


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import warnings\nwarnings.simplefilter('ignore')\n" + script,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("name", list(ORDERS))
def test_class_is_never_shadowed_by_the_legacy_module(name):
    script = ORDERS[name] + (
        "from eazypredict import EazyClassifier as C\n"
        "assert isinstance(C, type), f'expected a class, got {type(C)}'\n"
        "assert C.__name__ == 'EazyClassifier'\n"
        "print('ok')\n"
    )
    result = _run(script)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("name", list(ORDERS))
def test_legacy_module_path_resolves_to_the_shim(name):
    script = ORDERS[name] + (
        "from eazypredict.EazyClassifier import EazyClassifier as L\n"
        "assert isinstance(L, type)\n"
        "assert L.__name__ == 'LegacyEazyClassifier'\n"
        "from eazypredict.EazyRegressor import EazyRegressor as R\n"
        "assert R.__name__ == 'LegacyEazyRegressor'\n"
        "print('ok')\n"
    )
    result = _run(script)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_import_does_not_pull_in_boosters():
    """Importing the package must not pay the cost of - or require - xgboost and
    lightgbm. They are an optional extra, imported lazily on use."""
    result = _run(
        "import sys, eazypredict\n"
        "leaked = [m for m in ('xgboost', 'lightgbm') if m in sys.modules]\n"
        "assert not leaked, f'eagerly imported: {leaked}'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_package_import_is_side_effect_free_on_warnings():
    """A library must not install global warning filters. LazyPredict-style tools
    commonly call `warnings.filterwarnings("ignore")` at import, which silently
    suppresses warnings across the user's entire session.

    Our own dependencies (scipy, numpy) do register filters, so the baseline is taken
    after they are imported and only eazypredict's own contribution is measured.
    """
    result = _run(
        "import warnings, sklearn, pandas, numpy, scipy.special\n"
        "before = list(warnings.filters)\n"
        "import eazypredict\n"
        "added = [f for f in warnings.filters if f not in before]\n"
        "assert not added, f'import installed global filters: {added}'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_global_sklearn_config_is_not_mutated():
    """`sklearn.set_config` is process-global. A library that calls it silently
    changes the behaviour of every other estimator in the user's session."""
    result = _run(
        "import sklearn\n"
        "before = sklearn.get_config()\n"
        "import eazypredict\n"
        "assert sklearn.get_config() == before, 'import mutated the sklearn config'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
