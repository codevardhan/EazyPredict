"""The README is executed, not just read.

Every ```python block in the README is run, in order, against one shared namespace, so
a snippet may build on the one above it. This is the only thing that stops documentation
drifting away from the code: the published examples called
``fit(X_train, y_train, X_test, y_test)`` for two years while the implementation took
``(X_train, X_test, y_train, y_test)``, and nothing caught it because nothing ran them.

Blocks that are prose rather than code - expected output, directory listings, the
before/after sketch - use a plain fence and are skipped.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

_PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def python_blocks(text: str) -> list[str]:
    return _PYTHON_BLOCK.findall(text)


def test_the_readme_has_runnable_examples():
    """Guards the guard: a renamed file or a changed fence would otherwise make the
    test below pass by finding nothing to run."""
    assert len(python_blocks(README.read_text())) >= 10


def test_the_readme_uses_the_current_import_path():
    text = README.read_text()
    assert "from eazypredict import EazyClassifier" in text
    assert "from eazypredict.EazyClassifier import" not in text


@pytest.mark.slow
def test_every_readme_example_runs(tmp_path, monkeypatch):
    """Runs the whole README top to bottom. Slow, because two of the examples fit the
    full default estimator list, which is exactly what readers will do first."""
    # The saving example writes to a relative path; keep it out of the repository.
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__readme__"}

    for index, block in enumerate(python_blocks(README.read_text())):
        with warnings.catch_warnings():
            # Convergence and voting-fallback warnings are part of what the examples
            # legitimately produce; the suite's `error::FutureWarning` filter stays on.
            warnings.simplefilter("ignore", UserWarning)
            try:
                exec(compile(block, f"README.md[block {index}]", "exec"), namespace)
            except Exception as error:  # pragma: no cover - only on a broken README
                pytest.fail(
                    f"README block {index} failed with "
                    f"{type(error).__name__}: {error}\n\n{block}"
                )


def test_the_saving_example_names_the_files_it_claims(tmp_path, monkeypatch):
    """The README shows a directory listing. If `save_dir` ever changed its layout, the
    listing would silently become a lie."""
    from sklearn.datasets import make_classification

    from eazypredict import EazyClassifier

    monkeypatch.chdir(tmp_path)
    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    EazyClassifier(
        estimators=["LogisticRegression", "DecisionTreeClassifier"],
        save_dir="runs/experiment-1",
        random_state=0,
    ).fit(X, y)

    written = set(os.listdir(tmp_path / "runs" / "experiment-1"))
    listing = README.read_text().split("runs/experiment-1/")[1].split("```")[0]
    for name in written:
        assert name in listing, f"{name} is written but not documented"
