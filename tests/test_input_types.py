"""The X/y container matrix.

0.2.x worked only for the container types its examples happened to use: `fit` special
-cased ndarray targets while `fitVotingEnsemble` called `.values.ravel()`
unconditionally, so a numpy target crashed the ensemble. Parametrising over containers
is what stops that class of bug returning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eazypredict import EazyClassifier, EazyRegressor

PAIR = ["LogisticRegression", "DecisionTreeClassifier"]


def _as_X(X: np.ndarray, kind: str):
    columns = [f"f{i}" for i in range(X.shape[1])]
    if kind == "ndarray":
        return X
    if kind == "dataframe":
        return pd.DataFrame(X, columns=columns)
    if kind == "dataframe_mixed_dtypes":
        frame = pd.DataFrame(X, columns=columns)
        frame["category"] = np.where(frame["f0"] > 0, "high", "low")
        return frame
    raise AssertionError(kind)


def _as_y(y: np.ndarray, kind: str):
    if kind == "ndarray_1d":
        return y
    if kind == "ndarray_2d_column":
        return y.reshape(-1, 1)
    if kind == "series":
        return pd.Series(y, name="target")
    if kind == "dataframe_1col":
        return pd.DataFrame({"target": y})
    if kind == "list":
        return list(y)
    if kind == "strings":
        return np.array(["no", "yes"])[y]
    if kind == "categorical":
        return pd.Categorical(np.array(["no", "yes"])[y])
    raise AssertionError(kind)


X_KINDS = ["ndarray", "dataframe", "dataframe_mixed_dtypes"]
Y_KINDS = [
    "ndarray_1d",
    "ndarray_2d_column",
    "series",
    "dataframe_1col",
    "list",
    "strings",
    "categorical",
]


@pytest.mark.parametrize("y_kind", Y_KINDS)
@pytest.mark.parametrize("X_kind", X_KINDS)
def test_classifier_accepts_container_types(binary_data, X_kind, y_kind):
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(
        _as_X(X, X_kind), _as_y(y, y_kind)
    )
    assert (clf.results_["status"] == "ok").all()
    assert len(clf.predict(_as_X(X, X_kind))) == len(y)


@pytest.mark.parametrize("y_kind", Y_KINDS)
@pytest.mark.parametrize("X_kind", X_KINDS)
def test_voting_ensemble_accepts_container_types(binary_data, X_kind, y_kind):
    """The same matrix through the ensemble path, where `.values.ravel()` used to be."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(
        _as_X(X, X_kind), _as_y(y, y_kind)
    )
    ensemble = clf.fit_voting_ensemble(n_models=2)
    assert len(ensemble.predict(clf._X_train)) == len(clf._y_train)


@pytest.mark.parametrize(
    "y_kind", ["ndarray_1d", "ndarray_2d_column", "series", "list"]
)
@pytest.mark.parametrize("X_kind", X_KINDS)
def test_regressor_accepts_container_types(regression_data, X_kind, y_kind):
    X, y = regression_data
    reg = EazyRegressor(
        estimators=["LinearRegression", "DecisionTreeRegressor"],
        cv=None,
        random_state=0,
    ).fit(_as_X(X, X_kind), _as_y(y, y_kind))
    assert (reg.results_["status"] == "ok").all()
