"""Shared fixtures.

Datasets are deliberately tiny (``n_samples <= 80``, ``n_features <= 6``) so the
default suite stays fast. Tests that need the full default estimator list, boosters,
or scikit-learn's conformance checks must be marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

N_SAMPLES = 80
N_FEATURES = 6
SEED = 0


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# --------------------------------------------------------------------------- data


@pytest.fixture(scope="session")
def binary_data():
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=4,
        n_redundant=1,
        random_state=SEED,
    )


@pytest.fixture(scope="session")
def multiclass_data():
    """Three classes. Exercises the averaged-metric and OVR AUC paths."""
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=SEED,
    )


@pytest.fixture(scope="session")
def imbalanced_data():
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=4,
        n_redundant=1,
        weights=[0.85, 0.15],
        random_state=SEED,
    )


@pytest.fixture(scope="session")
def string_label_data(binary_data):
    """Non-integer labels. XGBoost requires ``y in {0..n-1}`` and will reject these
    unless we label-encode internally."""
    X, y = binary_data
    return X, np.array(["setosa", "versicolor"])[y]


@pytest.fixture(scope="session")
def multilabel_data():
    return make_multilabel_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=3, random_state=SEED
    )


@pytest.fixture(scope="session")
def regression_data():
    return make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES, noise=5.0, random_state=SEED
    )


@pytest.fixture(scope="session")
def messy_frame():
    """A DataFrame shaped like real-world data, and the acceptance test for automatic
    preprocessing.

    Contains every column type that crashes a bare scikit-learn estimator: missing
    numerics, an object column with missing values, a ``pd.Categorical``, a boolean, a
    zero-variance column, and a high-cardinality column in the spirit of the Titanic
    ``Cabin`` field.
    """
    rng = np.random.default_rng(SEED)
    n = N_SAMPLES

    numeric = rng.normal(size=n)
    numeric[rng.choice(n, size=n // 10, replace=False)] = np.nan

    categorical = rng.choice(["alpha", "beta", "gamma"], size=n).astype(object)
    categorical[rng.choice(n, size=n // 10, replace=False)] = None

    X = pd.DataFrame(
        {
            "num_with_nan": numeric,
            "num_int": rng.integers(0, 100, size=n),
            "cat_object": categorical,
            "cat_dtype": pd.Categorical(rng.choice(["red", "blue"], size=n)),
            "flag": rng.random(size=n) > 0.5,
            "constant": 1.0,
            "high_cardinality": [f"id_{i % 40}" for i in range(n)],
        }
    )
    y = pd.Series((rng.random(n) > 0.5).astype(int), name="target")
    return X, y


# ---------------------------------------------------------------------- estimators


@pytest.fixture
def fast_classifiers():
    """Two cheap estimators. Use these anywhere the test is not about the defaults."""
    return ["LogisticRegression", "DecisionTreeClassifier"]


@pytest.fixture
def fast_regressors():
    return ["LinearRegression", "DecisionTreeRegressor"]


class ExplodingClassifier(ClassifierMixin, BaseEstimator):
    """Raises during ``fit``. A real estimator failure, not a metric failure."""

    def fit(self, X, y):
        raise RuntimeError("boom")

    def predict(self, X):  # pragma: no cover - never reached
        raise AssertionError("unreachable")


@pytest.fixture
def failing_estimator():
    return ExplodingClassifier()


@pytest.fixture
def tmp_save_dir(tmp_path):
    return tmp_path / "models"
