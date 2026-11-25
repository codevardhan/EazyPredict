"""Conformance with the scikit-learn estimator contract.

The full `check_estimator` suite is deliberately not run by default: it fits the
estimator dozens of times, and because each of our fits trains a whole leaderboard,
the cost is multiplied by the number of candidate models. The fast hand-written gate
below covers the parts of the contract that actually protect us - above all, that
`__init__` stores parameters verbatim and `fit` does not mutate them, which is what
structurally prevents the 0.2.x "second fit returns nothing" bug.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor

from eazypredict import EazyClassifier, EazyRegressor

ESTIMATORS = [EazyClassifier, EazyRegressor]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_get_params_matches_init_signature(cls):
    signature = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert set(cls().get_params()) == signature


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_init_does_not_touch_its_arguments(cls):
    """scikit-learn requires __init__ to store parameters unmodified."""
    estimator = cls()
    for name, value in estimator.get_params().items():
        assert getattr(estimator, name) is value


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_no_fitted_attributes_before_fit(cls):
    estimator = cls()
    leaked = [
        name
        for name in vars(estimator)
        if name.endswith("_") and not name.startswith("_")
    ]
    assert leaked == []


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_clone_round_trips(cls):
    estimator = cls(sort_by="accuracy" if cls is EazyClassifier else "rmse")
    assert clone(estimator).get_params() == estimator.get_params()


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_set_params_round_trips(cls):
    estimator = cls()
    assert estimator.set_params(**estimator.get_params()).get_params() == (
        cls().get_params()
    )


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_repr_is_evaluable(cls):
    estimator = cls(cv=3, random_state=0)
    namespace = {"EazyClassifier": EazyClassifier, "EazyRegressor": EazyRegressor}
    assert eval(repr(estimator), namespace).get_params() == estimator.get_params()


def test_estimator_types_are_advertised():
    assert is_classifier(EazyClassifier())
    assert is_regressor(EazyRegressor())


def test_fit_returns_self(binary_data):
    X, y = binary_data
    clf = EazyClassifier(estimators=["LogisticRegression"], cv=None, random_state=0)
    assert clf.fit(X, y) is clf


def test_score_from_mixin(binary_data):
    """ClassifierMixin.score must work, which requires predict() to be well behaved."""
    X, y = binary_data
    clf = EazyClassifier(estimators=["LogisticRegression"], cv=None, random_state=0)
    assert 0.0 <= clf.fit(X, y).score(X, y) <= 1.0


def test_missing_target_says_so(binary_data):
    """`np.asarray(None)` makes a 0-d object array, and the downstream complaint about
    an "unknown label type" gives no hint that y was simply omitted."""
    X, _ = binary_data
    with pytest.raises(ValueError, match="target y is None"):
        EazyClassifier(estimators=["LogisticRegression"], cv=None).fit(X, None)


def test_continuous_target_is_rejected_by_the_classifier(regression_data):
    """Otherwise every distinct float becomes its own class and the user gets a
    plausible-looking leaderboard built on nonsense."""
    X, y = regression_data
    with pytest.raises(ValueError, match=r"[Uu]nknown label type|continuous"):
        EazyClassifier(estimators=["LogisticRegression"], cv=None).fit(X, y)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_column_vector_target_warns_before_flattening(cls, binary_data):
    """`df[["target"]]` is too common to reject, but silently ravelling it hides the
    case where the caller meant a multi-output target."""
    from sklearn.exceptions import DataConversionWarning

    X, y = binary_data
    estimator = cls(
        estimators=[
            "LogisticRegression" if cls is EazyClassifier else "LinearRegression"
        ],
        cv=None,
        random_state=0,
    )
    with pytest.warns(DataConversionWarning, match="column-vector y"):
        estimator.fit(X, y.reshape(-1, 1))


def test_strict_array_likes_are_not_rejected_by_shape_inspection(binary_data):
    """`np.shape`/`np.ndim` dispatch through `__array_function__`, which a strict
    array-like may refuse - with a TypeError unrelated to the caller's data."""

    class OnlyArrayProtocol:
        def __init__(self, data):
            self._data = data

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._data, dtype=dtype)

        def __array_function__(self, func, types, args, kwargs):
            raise TypeError(f"refusing {func.__name__}")

    X, y = binary_data
    clf = EazyClassifier(
        estimators=["DecisionTreeClassifier"], preprocess=False, cv=None, random_state=0
    ).fit(OnlyArrayProtocol(X), y)
    assert clf.n_features_in_ == X.shape[1]


#: Checks that encode an assumption eazypredict deliberately does not share.
#: Passed to `check_estimator(expected_failed_checks=...)`, which is the sklearn >=1.6
#: replacement for the removed `_xfail_checks` estimator tag.
EXPECTED_FAILED_CHECKS = {
    "check_estimators_nan_inf": (
        "eazypredict tolerates NaN by design - it imputes rather than rejecting, which "
        "is the entire point of automatic preprocessing. Even with preprocess=False, "
        "the inner DecisionTreeClassifier has handled NaN natively since sklearn 1.3, "
        "so nothing raises and the check's premise does not hold."
    ),
    "check_estimator_sparse_tag": (
        "Whether sparse input works depends entirely on which candidate models the "
        "user picked, so no single static tag can be truthful. `sparse=False` is the "
        "honest conservative answer: sparse input raises for GaussianNB, "
        "HistGradientBoosting and for every configuration with preprocess=True "
        "(StandardScaler cannot centre a sparse matrix). This check only objects "
        "because the one narrow configuration it is handed happens to work."
    ),
}


@pytest.mark.slow
def test_full_sklearn_check_estimator():
    """Opt-in conformance run against a single-model, no-preprocessing configuration."""
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(
        EazyClassifier(
            estimators=["DecisionTreeClassifier"],
            preprocess=False,
            cv=None,
            on_error="raise",
        ),
        expected_failed_checks=EXPECTED_FAILED_CHECKS,
    )
