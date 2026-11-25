"""The 0.2.x compatibility shims.

The legacy `fit` took `(X_train, X_test, y_train, y_test)` while every published
example called `(X_train, y_train, X_test, y_test)`. The shim preserves the
*implemented* order: quietly reinterpreting the arguments would train half of all
existing callers on their test set without raising anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from eazypredict.EazyClassifier import EazyClassifier as LegacyClassifier
from eazypredict.EazyRegressor import EazyRegressor as LegacyRegressor

PAIR = ["LogisticRegression", "DecisionTreeClassifier"]

#: Every legacy `fit` emits this; matching it keeps the assertion specific.
LEGACY_FIT = r"fit\(X, y\)"


@pytest.fixture
def split(binary_data):
    X, y = binary_data
    return X[:60], X[60:], y[:60], y[60:]


def test_legacy_fit_warns_and_returns_three_tuple(split):
    X_train, X_test, y_train, y_test = split
    clf = LegacyClassifier(classififers=PAIR)
    with pytest.warns(DeprecationWarning, match="fit\\(X, y\\)"):
        models, predictions, results = clf.fit(X_train, X_test, y_train, y_test)

    assert set(models) == set(PAIR)
    assert set(predictions) == set(PAIR)
    assert list(results.columns) == ["Accuracy", "f1 score", "ROC AUC score"]
    assert len(results) == 2


def test_legacy_shim_preserves_the_implemented_argument_order(split):
    """The second positional argument is the test *features*, as it always was."""
    X_train, X_test, y_train, y_test = split
    clf = LegacyClassifier(classififers=PAIR)
    with pytest.warns(DeprecationWarning, match=LEGACY_FIT):
        _, predictions, _ = clf.fit(X_train, X_test, y_train, y_test)
    assert len(predictions["LogisticRegression"]) == len(y_test)


def test_legacy_misspelled_parameter_still_accepted(split):
    """`classififers` was misspelled in the 0.2.x public API, and is in the README."""
    clf = LegacyClassifier(classififers=PAIR)
    assert clf.classififers == PAIR


def test_legacy_voting_ensemble_warns(split):
    X_train, X_test, y_train, y_test = split
    clf = LegacyClassifier(classififers=PAIR)
    with pytest.warns(DeprecationWarning, match=LEGACY_FIT):
        models, _, results = clf.fit(X_train, X_test, y_train, y_test)
    with pytest.warns(DeprecationWarning, match="fitVotingEnsemble is deprecated"):
        ensemble, frame = clf.fitVotingEnsemble(models, results, num_models=2)
    assert "Models" in frame.columns
    assert len(ensemble.predict(X_test)) == len(y_test)


def test_legacy_regressor(regression_data):
    X, y = regression_data
    clf = LegacyRegressor(regressors=["LinearRegression", "DecisionTreeRegressor"])
    with pytest.warns(DeprecationWarning, match=LEGACY_FIT):
        models, _, results = clf.fit(X[:60], X[60:], y[:60], y[60:])
    assert list(results.columns) == ["RMSE", "R Squared"]
    assert set(models) == {"LinearRegression", "DecisionTreeRegressor"}


def test_legacy_sort_keys_are_translated(split):
    """`f1_score` and `roc_score` were the 0.2.x spellings, and both were unusable:
    one raised IndexError, the other AttributeError."""
    X_train, X_test, y_train, y_test = split
    for sort_by in ("f1_score", "roc_score"):
        clf = LegacyClassifier(classififers=PAIR, sort_by=sort_by)
        with pytest.warns(DeprecationWarning, match=LEGACY_FIT):
            _, _, results = clf.fit(X_train, X_test, y_train, y_test)
        assert len(results) == 2


def test_legacy_classes_are_subclasses_of_the_new_api():
    from eazypredict import EazyClassifier, EazyRegressor

    assert issubclass(LegacyClassifier, EazyClassifier)
    assert issubclass(LegacyRegressor, EazyRegressor)


def test_legacy_pandas_target_still_works(split):
    """The 0.2.x code called `y_train.values.ravel()`, so DataFrame targets were the
    only shape its ensemble path supported. They must keep working."""
    X_train, X_test, y_train, y_test = split
    clf = LegacyClassifier(classififers=PAIR)
    with pytest.warns(DeprecationWarning, match=LEGACY_FIT):
        _, _, results = clf.fit(
            pd.DataFrame(X_train),
            pd.DataFrame(X_test),
            pd.DataFrame({"y": y_train}),
            pd.DataFrame({"y": y_test}),
        )
    assert np.isfinite(results["Accuracy"]).all()
