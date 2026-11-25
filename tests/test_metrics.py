"""The metric registry.

Every metric is a scikit-learn *scorer*, which is what makes ROC AUC correct: the
scorer asks the fitted estimator for probabilities itself, rather than being handed
hard class labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import get_scorer, roc_auc_score
from sklearn.model_selection import train_test_split

from eazypredict import EazyClassifier, EazyRegressor
from eazypredict.metrics import (
    MetricSpec,
    classification_metrics,
    regression_metrics,
    resolve_metrics,
)


def test_every_default_scorer_name_exists():
    """A typo in a scorer name would only surface mid-run, after minutes of fitting."""
    for registry in (classification_metrics(2), classification_metrics(3)):
        assert registry
    assert regression_metrics()


def test_binary_and_multiclass_use_different_auc():
    """`roc_auc` is undefined for >2 classes; the one-vs-rest variant is required."""
    binary = classification_metrics(2)["roc_auc"].scorer
    multi = classification_metrics(3)["roc_auc"].scorer
    # `get_scorer` hands back a fresh copy each call, so compare by repr.
    assert repr(binary) != repr(multi)
    assert repr(binary) == repr(get_scorer("roc_auc"))
    assert repr(multi) == repr(get_scorer("roc_auc_ovr_weighted"))


@pytest.mark.parametrize("name", ["log_loss"])
def test_minimised_classification_metrics_are_displayed_positive(name):
    spec = classification_metrics(2)[name]
    assert not spec.greater_is_better
    # sklearn returns these negated; display flips them back.
    assert spec.display(-0.25) == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["rmse", "mae", "mape"])
def test_minimised_regression_metrics_are_displayed_positive(name):
    spec = regression_metrics()[name]
    assert not spec.greater_is_better
    assert spec.display(-3.0) == pytest.approx(3.0)


@pytest.mark.parametrize("name", ["accuracy", "balanced_accuracy", "f1", "roc_auc"])
def test_maximised_metrics_pass_through_unchanged(name):
    spec = classification_metrics(2)[name]
    assert spec.greater_is_better
    assert spec.display(0.8) == pytest.approx(0.8)


def test_rmse_uses_the_surviving_sklearn_api():
    """`mean_squared_error(squared=False)` was removed in sklearn 1.6."""
    scorer = regression_metrics()["rmse"].scorer
    assert "root_mean_squared_error" in repr(scorer)
    assert repr(scorer) == repr(get_scorer("neg_root_mean_squared_error"))


# ----------------------------------------------------------------- resolve_metrics


def test_none_returns_a_copy_of_the_defaults():
    defaults = classification_metrics(2)
    resolved = resolve_metrics(None, defaults)
    assert resolved == defaults
    resolved.pop("accuracy")
    assert "accuracy" in defaults


def test_a_list_selects_and_orders_a_subset():
    resolved = resolve_metrics(["f1", "accuracy"], classification_metrics(2))
    assert list(resolved) == ["f1", "accuracy"]


def test_unknown_metric_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_metrics(["not_a_metric"], classification_metrics(2))


def test_a_mapping_accepts_scorer_names_and_callables():
    resolved = resolve_metrics(
        {"acc": "accuracy", "custom": get_scorer("f1")}, classification_metrics(2)
    )
    assert set(resolved) == {"acc", "custom"}
    assert all(isinstance(spec, MetricSpec) for spec in resolved.values())


# -------------------------------------------------------------------- integration


def test_requested_metrics_become_the_leaderboard_columns(binary_data):
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["LogisticRegression"],
        metrics=["accuracy", "f1"],
        cv=None,
        random_state=0,
    ).fit(X, y)
    bookkeeping = {"fit_time", "score_time", "status"}
    assert [c for c in clf.results_.columns if c not in bookkeeping] == [
        "accuracy",
        "f1",
    ]


def test_sort_by_must_name_a_selected_metric(binary_data):
    """Sorting by a metric that was never computed produced a KeyError deep inside
    the 0.2.x sort; it must be an actionable error instead."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["LogisticRegression"], metrics=["accuracy"], sort_by="f1", cv=None
    )
    with pytest.raises(ValueError, match="Invalid sort_by"):
        clf.fit(X, y)


def test_roc_auc_is_computed_from_probabilities_not_labels(binary_data):
    """The headline metric bug. On noisy data the label-based figure differs from the
    probability-based one in *both* directions, so it can reorder the leaderboard."""
    X, y = binary_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = EazyClassifier(
        estimators=["RandomForestClassifier"], cv=None, random_state=0
    ).fit(X_train, y_train, X_test=X_test, y_test=y_test)

    model = clf.models_["RandomForestClassifier"]
    expected = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert clf.results_.loc["RandomForestClassifier", "roc_auc"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize("cv", [None, 3])
@pytest.mark.parametrize("name", ["RidgeClassifier", "SVC", "SGDClassifier"])
def test_an_inapplicable_metric_costs_one_cell_not_the_whole_model(name, cv):
    """`cross_validate(error_score=...)` only covers exceptions from *fit*; one raised
    while scoring still propagates. These three have no `predict_proba`, so asking for
    log loss used to discard their accuracy too and drop them from the leaderboard -
    under `cv=5`, which is the default.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    clf = EazyClassifier(estimators=[name], cv=cv, random_state=0).fit(X, y)

    assert clf.results_.loc[name, "status"] == "ok"
    assert np.isfinite(clf.results_.loc[name, "accuracy"])
    # Ranking metrics still work: these expose `decision_function`.
    assert np.isfinite(clf.results_.loc[name, "roc_auc"])
    # Only the genuinely unavailable metric is missing.
    assert np.isnan(clf.results_.loc[name, "log_loss"])


def test_auc_is_nan_not_an_exception_for_an_unscorable_model(binary_data, caplog):
    """`RidgeClassifier` has no `predict_proba`. Its decision_function is unnormalised,
    so a genuine AUC is unavailable - but that must not fail the whole run."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["RidgeClassifier", "LogisticRegression"], cv=None, random_state=0
    ).fit(X, y)
    assert clf.results_.loc["RidgeClassifier", "status"] == "ok"
    assert np.isfinite(clf.results_.loc["RidgeClassifier", "accuracy"])


def test_log_loss_ranks_the_right_way_round(binary_data):
    """`greater_is_better=False` is the whole reason a minimised metric can be used
    as `sort_by` at all."""
    X, y = binary_data
    clf = EazyClassifier(
        estimators=["LogisticRegression", "DecisionTreeClassifier"],
        sort_by="log_loss",
        cv=None,
        random_state=0,
    ).fit(X, y)
    losses = clf.results_["log_loss"].to_numpy()
    assert (losses >= 0).all()
    assert losses[0] == pytest.approx(np.nanmin(losses))


def test_regressor_rmse_is_positive_and_sorted_ascending(regression_data):
    X, y = regression_data
    reg = EazyRegressor(
        estimators=["LinearRegression", "DecisionTreeRegressor"],
        sort_by="rmse",
        cv=None,
        random_state=0,
    ).fit(X, y)
    rmse = reg.results_["rmse"].to_numpy()
    assert (rmse > 0).all()
    assert rmse[0] == pytest.approx(np.nanmin(rmse))


def test_multiclass_metrics_are_finite(multiclass_data):
    X, y = multiclass_data
    clf = EazyClassifier(
        estimators=["LogisticRegression"], cv=None, random_state=0
    ).fit(X, y)
    row = clf.results_.loc["LogisticRegression"]
    for name in classification_metrics(3):
        assert np.isfinite(row[name]), f"{name} is not finite"
