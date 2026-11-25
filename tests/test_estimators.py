"""Estimator discovery and resolution."""

from __future__ import annotations

import pytest
from sklearn.base import is_classifier, is_regressor

from eazypredict._compat import BOOSTER_ESTIMATORS, has_booster, load_optional_estimator
from eazypredict.estimators import (
    DEFAULT_CLASSIFIERS,
    DEFAULT_REGRESSORS,
    available_estimators,
    resolve_estimators,
)


def test_resolve_is_pure():
    """The 0.2.x helper rewrote `self.classifiers` in place, so the second call saw a
    list of tuples where it expected names and silently resolved to nothing."""
    spec = ["LogisticRegression", "DecisionTreeClassifier"]
    original = list(spec)
    first = resolve_estimators(spec, "classifier")
    second = resolve_estimators(spec, "classifier")
    assert spec == original
    assert [n for n, _ in first] == [n for n, _ in second] == original


def test_resolve_returns_fresh_instances():
    a = resolve_estimators(["DecisionTreeClassifier"], "classifier")[0][1]
    b = resolve_estimators(["DecisionTreeClassifier"], "classifier")[0][1]
    assert a is not b


@pytest.mark.parametrize(
    ("kind", "checker"), [("classifier", is_classifier), ("regressor", is_regressor)]
)
def test_only_the_right_kind_is_discovered(kind, checker):
    """Without `type_filter`, `all_estimators()` also returns transformers, so
    `StandardScaler` could be instantiated as a classifier."""
    for name, cls in available_estimators(kind).items():
        if name in BOOSTER_ESTIMATORS:
            continue
        # `is_classifier`/`is_regressor` read estimator tags, which sklearn >=1.6
        # only exposes on instances.
        assert checker(cls()), f"{name} is not a {kind}"


def test_transformer_is_rejected():
    with pytest.raises(ValueError, match="Unknown classifier"):
        resolve_estimators(["StandardScaler"], "classifier")


def test_typo_suggests_the_intended_name():
    with pytest.raises(ValueError, match=r"Did you mean.*LogisticRegression"):
        resolve_estimators(["LogisticRegresion"], "classifier")


def test_empty_specification_is_an_error():
    with pytest.raises(ValueError, match="No estimators requested"):
        resolve_estimators([], "classifier")


def test_unknown_string_specification():
    with pytest.raises(ValueError, match="Unknown estimator specification"):
        resolve_estimators("everything", "classifier")


def test_pre_configured_instances_pass_through():
    from sklearn.ensemble import RandomForestClassifier

    configured = RandomForestClassifier(n_estimators=7)
    resolved = resolve_estimators([("custom", configured)], "classifier")
    assert resolved == [("custom", configured)]
    assert resolved[0][1].n_estimators == 7


def test_bare_instance_is_named_after_its_class():
    from sklearn.tree import DecisionTreeClassifier

    name, estimator = resolve_estimators([DecisionTreeClassifier()], "classifier")[0]
    assert name == "DecisionTreeClassifier"
    assert isinstance(estimator, DecisionTreeClassifier)


def test_defaults_resolve():
    resolved = resolve_estimators("default", "classifier")
    assert len(resolved) >= len(DEFAULT_CLASSIFIERS) - len(BOOSTER_ESTIMATORS)


def test_defaults_include_a_pure_sklearn_booster():
    """So that the core install, without the optional [boosters] extra, still has a
    competitive gradient-boosting model on the leaderboard."""
    assert "HistGradientBoostingClassifier" in DEFAULT_CLASSIFIERS
    assert "HistGradientBoostingRegressor" in DEFAULT_REGRESSORS


def test_gaussian_process_is_not_a_default_regressor():
    """It is O(n^3) in time and O(n^2) in memory and exhausts RAM on datasets of only
    moderate size, which is a poor default for a tool that runs unattended."""
    assert "GaussianProcessRegressor" not in DEFAULT_REGRESSORS


def test_explicitly_requesting_a_missing_booster_is_an_error(monkeypatch):
    """Silently returning a shorter leaderboard than the user asked for is worse than
    failing: they would never know the model was skipped."""
    monkeypatch.setattr("eazypredict.estimators.has_booster", lambda name: False)
    monkeypatch.setattr("eazypredict.estimators.available_estimators", lambda kind: {})
    with pytest.raises(ImportError, match="eazypredict\\[boosters\\]"):
        resolve_estimators(["XGBClassifier"], "classifier")


def test_missing_booster_in_defaults_warns_but_continues(monkeypatch):
    monkeypatch.setattr("eazypredict.estimators.has_booster", lambda name: False)
    with pytest.warns(UserWarning, match="boosters"):
        resolved = resolve_estimators("default", "classifier")
    assert not {n for n, _ in resolved} & set(BOOSTER_ESTIMATORS)


def test_has_booster_rejects_unknown_names():
    assert not has_booster("NotAnEstimator")


def test_load_optional_estimator_rejects_unknown_names():
    with pytest.raises(KeyError):
        load_optional_estimator("NotAnEstimator")


@pytest.mark.slow
def test_boosters_resolve_when_installed():
    for name in BOOSTER_ESTIMATORS:
        if not has_booster(name):
            pytest.skip(f"{name} not installed")
        assert load_optional_estimator(name) is not None
