"""Cross-validated leaderboards.

`cv=5` is the default because a single split of a few hundred rows makes the ranking
mostly noise: the gap between adjacent models is routinely smaller than the
split-to-split variance. `cv=None` is the explicit fast path.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from eazypredict import EazyClassifier, EazyRegressor

PAIR = ["LogisticRegression", "DecisionTreeClassifier"]


def test_cv_is_the_default():
    assert EazyClassifier().cv == 5
    assert EazyRegressor().cv == 5


def test_cv_adds_a_std_column_per_metric(binary_data):
    """A mean without a spread invites over-reading a 0.01 difference."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=3, random_state=0).fit(X, y)
    for metric in clf.metrics_:
        assert f"{metric}_std" in clf.results_.columns
    assert (clf.results_["accuracy_std"] >= 0).all()


def test_single_split_has_no_std_columns(binary_data):
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    assert not [c for c in clf.results_.columns if c.endswith("_std")]


def test_classifier_stratifies_by_default(binary_data):
    _, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=3, random_state=0)
    assert isinstance(clf._cv_splitter(y), StratifiedKFold)


def test_stratify_can_be_turned_off(binary_data):
    _, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=3, stratify=False, random_state=0)
    assert isinstance(clf._cv_splitter(y), KFold)


def test_regressor_uses_plain_kfold(regression_data):
    _, y = regression_data
    reg = EazyRegressor(cv=3, random_state=0)
    assert isinstance(reg._cv_splitter(y), KFold)


def test_an_explicit_splitter_is_honoured(binary_data):
    X, y = binary_data
    splitter = KFold(n_splits=2, shuffle=True, random_state=0)
    clf = EazyClassifier(estimators=PAIR, cv=splitter, random_state=0).fit(X, y)
    assert clf._cv_splitter(y) is splitter
    assert np.isfinite(clf.results_["accuracy"]).all()


def test_cv_results_are_reproducible(binary_data):
    """Without a seeded splitter *and* seeded estimators, two identical calls return
    different leaderboards - which makes the comparison worthless."""
    X, y = binary_data
    columns = ["accuracy", "f1"]
    first = EazyClassifier(estimators=PAIR, cv=3, random_state=42).fit(X, y)
    second = EazyClassifier(estimators=PAIR, cv=3, random_state=42).fit(X, y)
    np.testing.assert_allclose(
        first.results_[columns].to_numpy(), second.results_[columns].to_numpy()
    )


def test_the_best_model_is_refitted_on_all_training_data(binary_data):
    """`cross_validate` discards its fold models, so without an explicit refit the
    returned `best_estimator_` would be unfitted and `predict` would raise."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=3, random_state=0).fit(X, y)
    assert len(clf.predict(X)) == len(y)


def test_a_failing_model_does_not_abort_the_cv_run(binary_data, failing_estimator):
    """`error_score=np.nan` plus per-model isolation: one broken estimator costs you
    one row, not the whole multi-minute run."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[("boom", failing_estimator), "LogisticRegression"],
        cv=3,
        random_state=0,
        on_error="ignore",
    ).fit(X, y)
    assert clf.results_.loc["boom", "status"] == "failed"
    assert clf.results_.loc["LogisticRegression", "status"] == "ok"
    assert "boom" in clf.errors_


def test_failed_models_sort_last_whatever_the_direction(binary_data, failing_estimator):
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[("boom", failing_estimator), "LogisticRegression"],
        cv=3,
        random_state=0,
        on_error="ignore",
    ).fit(X, y)
    assert clf.results_.index[-1] == "boom"
    assert clf.best_model_name_ == "LogisticRegression"


def test_on_error_raise_surfaces_the_original_exception(binary_data, failing_estimator):
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[("boom", failing_estimator)], cv=None, on_error="raise"
    )
    with pytest.raises(RuntimeError, match="boom"):
        clf.fit(X, y)


def test_all_models_failing_is_an_error_not_an_empty_leaderboard(
    binary_data, failing_estimator
):
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[("boom", failing_estimator)], cv=3, on_error="ignore"
    )
    with pytest.raises(RuntimeError, match="Every candidate model failed"):
        clf.fit(X, y)


def test_invalid_on_error_is_rejected_before_any_fitting(binary_data):
    X, y = binary_data
    with pytest.raises(ValueError, match="Invalid on_error"):
        EazyClassifier(estimators=PAIR, on_error="explode").fit(X, y)


def test_an_explicit_test_set_bypasses_the_internal_split(binary_data):
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(
        X[:60], y[:60], X_test=X[60:], y_test=y[60:]
    )
    assert len(clf.predictions_["LogisticRegression"]) == 20


def test_x_test_without_y_test_is_an_error(binary_data):
    X, y = binary_data
    with pytest.raises(ValueError, match="y_test must be provided"):
        EazyClassifier(estimators=PAIR, cv=None).fit(X[:60], y[:60], X_test=X[60:])


def test_refitting_the_same_instance_is_idempotent(binary_data):
    """0.2.x consumed `self.classifiers` during the first fit, so the second produced
    an empty leaderboard."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=3, random_state=0)
    first = clf.fit(X, y).results_["accuracy"].copy()
    second = clf.fit(X, y).results_["accuracy"]
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())
    assert len(second) == len(PAIR)
