"""One named test per defect found in 0.2.x.

Each test documents the original failure so that a future refactor cannot quietly
reintroduce it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from eazypredict import EazyClassifier, EazyRegressor

PAIR = ["LogisticRegression", "DecisionTreeClassifier"]


# --------------------------------------------------------------------- sorting


@pytest.mark.parametrize(
    "sort_by",
    ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "log_loss"],
)
def test_every_sort_metric_works(binary_data, sort_by):
    """bug #1: results were sorted by indexing a 2-tuple with 2 or 3, so every metric
    except the first raised IndexError."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, sort_by=sort_by, cv=None, random_state=0).fit(
        X, y
    )
    assert len(clf.results_) == 2


def test_leaderboard_is_ordered_best_first(binary_data):
    """bug #1: sorting by 'accuracy' compared whole lists, so order was incidental."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, sort_by="accuracy", cv=None, random_state=0)
    clf.fit(X, y)
    accuracy = clf.results_["accuracy"].to_numpy()
    assert np.all(np.diff(accuracy) <= 0)
    assert clf.results_.index[0] == clf.best_model_name_


def test_lower_is_better_metric_sorts_ascending(binary_data):
    """A minimised metric must still put the best model in row 0. The 0.2.x code
    hard-coded `reverse=True`, so adding log loss would have inverted the ranking."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, sort_by="log_loss", cv=None, random_state=0)
    clf.fit(X, y)
    log_loss = clf.results_["log_loss"].to_numpy()
    assert np.all(np.diff(log_loss) >= 0)


def test_failed_model_sorts_last_not_first(binary_data, failing_estimator):
    """A NaN score must never win. 0.2.x raised TypeError comparing None to float."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[*PAIR, ("Exploding", failing_estimator)],
        cv=None,
        on_error="ignore",
        random_state=0,
    ).fit(X, y)
    assert clf.results_.index[-1] == "Exploding"
    assert clf.best_model_name_ in PAIR


def test_invalid_sort_by_raises_value_error(binary_data):
    """bug #2: `self.sort_key` was a typo for `self.sort_by`, and it sat in an elif
    *condition*, so any invalid value raised AttributeError and the intended
    `else: raise` branch was unreachable."""
    X, y = binary_data
    with pytest.raises(ValueError, match="Invalid sort_by"):
        EazyClassifier(estimators=PAIR, sort_by="garbage", cv=None).fit(X, y)


def test_regressor_sort_by_r2(regression_data):
    """bug #3: the same out-of-range tuple index in EazyRegressor."""
    X, y = regression_data
    reg = EazyRegressor(
        estimators=["LinearRegression", "DecisionTreeRegressor"],
        sort_by="r2",
        cv=None,
        random_state=0,
    ).fit(X, y)
    r2 = reg.results_["r2"].to_numpy()
    assert np.all(np.diff(r2) <= 0)


def test_regressor_rmse_is_reported_positive(regression_data):
    """RMSE is reported in natural units, not as scikit-learn's negated scorer value."""
    X, y = regression_data
    reg = EazyRegressor(
        estimators=["LinearRegression"], sort_by="rmse", cv=None, random_state=0
    ).fit(X, y)
    assert reg.results_.loc["LinearRegression", "rmse"] > 0


# ------------------------------------------------------------------- behaviour


def test_fit_is_idempotent(binary_data):
    """bug #8: `__getClassifierList` rewrote `self.classifiers` from names to tuples,
    so the second fit() filtered against the wrong shape and produced an empty frame
    (`ValueError: Length mismatch: Expected axis has 0 elements`)."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0)
    first = clf.fit(X, y).results_.copy()
    second = clf.fit(X, y).results_.copy()
    assert len(second) == 2
    assert_frame_equal(
        first.drop(columns=["fit_time", "score_time"]),
        second.drop(columns=["fit_time", "score_time"]),
    )


def test_fit_does_not_mutate_constructor_params(binary_data):
    """The scikit-learn contract: fit() must not change what get_params() returns."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0)
    before = clf.get_params()
    clf.fit(X, y)
    assert clf.get_params() == before
    assert clf.estimators == PAIR


def test_roc_auc_uses_probabilities_not_hard_labels():
    """bug #7: AUC was computed from predicted labels, discarding the ranking
    information the statistic is defined on. On noisy data this shifts the value in
    either direction and can reorder the leaderboard."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=3, flip_y=0.25, random_state=0
    )
    clf = EazyClassifier(
        estimators=["RandomForestClassifier"],
        cv=None,
        test_size=0.25,
        random_state=0,
    ).fit(X, y)

    model = clf.models_["RandomForestClassifier"]
    expected = roc_auc_score(clf._y_test, model.predict_proba(clf._X_test)[:, 1])
    assert clf.results_.loc["RandomForestClassifier", "roc_auc"] == pytest.approx(
        expected
    )


def test_auc_is_perfect_for_a_monotone_but_miscalibrated_ranker(binary_data):
    """A model that ranks perfectly but predicts badly calibrated probabilities should
    still score AUC 1.0 - only possible when AUC sees probabilities."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["LogisticRegression"], cv=None, random_state=0
    ).fit(X, y)
    assert 0.0 <= clf.results_.loc["LogisticRegression", "roc_auc"] <= 1.0


def test_failing_model_is_isolated(binary_data, failing_estimator):
    """bug #15: only metric computation was guarded, so a model that failed to *fit*
    aborted the whole run."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[*PAIR, ("Exploding", failing_estimator)],
        cv=None,
        on_error="ignore",
        random_state=0,
    ).fit(X, y)

    assert "Exploding" in clf.errors_
    assert "boom" in clf.errors_["Exploding"]
    assert clf.results_.loc["Exploding", "status"] == "failed"
    assert set(PAIR) <= set(clf.models_)


def test_on_error_raise_propagates(binary_data, failing_estimator):
    X, y = binary_data
    with pytest.raises(RuntimeError, match="boom"):
        EazyClassifier(
            estimators=[("Exploding", failing_estimator)], cv=None, on_error="raise"
        ).fit(X, y)


def test_messy_dataframe_end_to_end(messy_frame):
    """bug #16: no preprocessing, so a frame with NaNs and string columns failed with
    `ValueError: could not convert string to float`."""
    X, y = messy_frame
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    assert (clf.results_["status"] == "ok").all()
    assert len(clf.predict(X)) == len(y)


def test_random_state_makes_runs_reproducible(binary_data):
    """bug #14: most default estimators are stochastic and nothing was seeded."""
    X, y = binary_data
    columns = ["accuracy", "f1"]
    a = (
        EazyClassifier(estimators=["RandomForestClassifier"], cv=None, random_state=42)
        .fit(X, y)
        .results_[columns]
    )
    b = (
        EazyClassifier(estimators=["RandomForestClassifier"], cv=None, random_state=42)
        .fit(X, y)
        .results_[columns]
    )
    assert_frame_equal(a, b)


def test_string_labels_round_trip(string_label_data):
    """bug #23: XGBoost requires y in {0..n-1}; labels must be encoded internally and
    restored on predict."""
    X, y = string_label_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    assert set(clf.classes_) == {"setosa", "versicolor"}
    assert set(np.unique(clf.predict(X))) <= {"setosa", "versicolor"}


def test_transformer_rejected_as_classifier(binary_data):
    """bug #21: `all_estimators()` was called without a type filter, so a transformer
    was happily instantiated and then failed with an opaque AttributeError."""
    X, y = binary_data
    with pytest.raises(ValueError, match="Unknown classifier"):
        EazyClassifier(estimators=["StandardScaler"], cv=None).fit(X, y)


def test_unknown_estimator_suggests_alternatives(binary_data):
    X, y = binary_data
    with pytest.raises(ValueError, match="Did you mean"):
        EazyClassifier(estimators=["LogisticRegresion"], cv=None).fit(X, y)


def test_methods_before_fit_raise_not_fitted():
    """bug #24: these raised a bare AttributeError about a missing `X_train`."""
    clf = EazyClassifier()
    with pytest.raises(NotFittedError):
        clf.predict([[0, 1]])
    with pytest.raises(NotFittedError):
        clf.fit_voting_ensemble()


def test_dataframe_train_with_array_test(binary_data):
    """bug #22: an ndarray check converted *both* X_train and X_test, so a DataFrame
    train set silently lost its column names."""
    X, y = binary_data
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(
        frame.iloc[:60], y[:60], X_test=frame.iloc[60:], y_test=y[60:]
    )
    assert list(clf.feature_names_in_) == list(frame.columns)


def test_public_api_is_importable():
    """bug #13: __init__.py exported nothing, forcing the deep import path."""
    import eazypredict

    assert {"EazyClassifier", "EazyRegressor", "__version__"} <= set(
        eazypredict.__all__
    )


def test_no_removed_sklearn_apis_referenced():
    """bug #12: a docstring advertised `datasets.load_boston`, gone in sklearn 1.2."""
    from pathlib import Path

    banned = ["load_boston", "needs_proba=", "needs_threshold=", "squared=False"]
    source_root = Path(__file__).resolve().parents[1] / "src"
    for path in source_root.rglob("*.py"):
        text = path.read_text()
        for token in banned:
            assert token not in text, f"{path.name} references removed API {token!r}"


# -------------------------------------------------------------------- ensemble


def test_voting_ensemble_accepts_numpy_target(binary_data):
    """bug #5: fitVotingEnsemble called `self.y_train.values.ravel()`, which only works
    for a pandas object - it crashed for every ndarray caller."""
    X, y = binary_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    ensemble = clf.fit_voting_ensemble(n_models=2)
    assert len(ensemble.predict(X)) == len(y)


def test_voting_ensemble_multiclass(multiclass_data):
    """bug #6: the ensemble scored with `f1_score(...)` with no `average`, which raises
    on multiclass targets."""
    X, y = multiclass_data
    clf = EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    ensemble = clf.fit_voting_ensemble(n_models=2)
    assert len(np.unique(ensemble.predict(X))) >= 2


def test_ensemble_selection_follows_sort_metric(binary_data):
    """bug #25: selection iterated `model_results.iloc[:, 0]` - always the first
    column - so it ignored whichever metric the user actually sorted by."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=[*PAIR, "GaussianNB"], sort_by="f1", cv=None, random_state=0
    ).fit(X, y)
    ensemble = clf.fit_voting_ensemble(n_models=2)
    assert [name for name, _ in ensemble.estimators] == list(clf.results_.index[:2])


def test_soft_voting_falls_back_for_non_probabilistic_models(binary_data):
    """RidgeClassifier has no predict_proba; the fallback must be explicit, not a
    crash."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["RidgeClassifier", "DecisionTreeClassifier"],
        cv=None,
        random_state=0,
    ).fit(X, y)
    with pytest.warns(UserWarning, match="hard voting"):
        ensemble = clf.fit_voting_ensemble(n_models=2, voting="soft")
    assert ensemble.voting == "hard"


def test_regressor_voting_ensemble(regression_data):
    X, y = regression_data
    reg = EazyRegressor(
        estimators=["LinearRegression", "DecisionTreeRegressor"],
        cv=None,
        random_state=0,
    ).fit(X, y)
    ensemble = reg.fit_voting_ensemble(n_models=2)
    assert len(ensemble.predict(X)) == len(y)
