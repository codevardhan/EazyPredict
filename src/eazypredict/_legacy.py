"""Backwards-compatible shims for the 0.2.x API.

Importing ``eazypredict.EazyClassifier`` as a *module* is the 0.2.x usage that every
README, blog post and notebook copied, so it keeps working for now. These shims are
scheduled for removal in 0.5.0.

The legacy ``fit`` signature is irreparably ambiguous: the implementation was
``fit(X_train, X_test, y_train, y_test)`` while every published example called
``fit(X_train, y_train, X_test, y_test)``. Silently reinterpreting the arguments would
train half of all existing callers on their test set with no error raised, so the shim
preserves the *implemented* order exactly and points users at the new API instead.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .classification import EazyClassifier as _EazyClassifier
from .regression import EazyRegressor as _EazyRegressor

_MIGRATION = (
    "Use `from eazypredict import EazyClassifier` and the scikit-learn style "
    "`fit(X, y)` instead. Note that the old positional order was "
    "(X_train, X_test, y_train, y_test), which did not match the documented "
    "examples. This shim will be removed in 0.5.0."
)

_LEGACY_CLASSIFIER_COLUMNS = {
    "accuracy": "Accuracy",
    "f1": "f1 score",
    "roc_auc": "ROC AUC score",
}
_LEGACY_REGRESSOR_COLUMNS = {"rmse": "RMSE", "r2": "R Squared"}


def _legacy_frame(results: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    present = [c for c in columns if c in results.columns]
    frame = results[present].copy()
    frame.columns = [columns[c] for c in present]
    frame.index.name = None
    return frame


class LegacyEazyClassifier(_EazyClassifier):
    """Deprecated 0.2.x-compatible classifier."""

    def __init__(self, classififers="all", save_dir=None, sort_by="accuracy"):
        # The 0.2.x parameter name was misspelled; it is reproduced verbatim here
        # because that is what existing callers pass.
        super().__init__(
            estimators="default" if classififers == "all" else classififers,
            sort_by="accuracy" if sort_by == "accuracy" else _legacy_sort(sort_by),
            save_dir=save_dir,
            cv=None,
        )
        self.classififers = classififers

    def fit(self, X_train, X_test, y_train, y_test):
        """Deprecated. Returns ``(models, predictions, results)`` as 0.2.x did."""
        warnings.warn(_MIGRATION, DeprecationWarning, stacklevel=2)
        super().fit(X_train, y_train, X_test=X_test, y_test=y_test)
        return (
            self.models_,
            self.predictions_,
            _legacy_frame(self.results_, _LEGACY_CLASSIFIER_COLUMNS),
        )

    def fitVotingEnsemble(self, model_dict=None, model_results=None, num_models=5):
        """Deprecated alias for :meth:`fit_voting_ensemble`."""
        warnings.warn(
            "fitVotingEnsemble is deprecated; use fit_voting_ensemble(n_models=...). "
            "The model_dict and model_results arguments are ignored: the ensemble is "
            "built from the fitted leaderboard. This shim will be removed in 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        ensemble = self.fit_voting_ensemble(n_models=num_models, voting="hard")
        return ensemble, _ensemble_frame(self, ensemble, _LEGACY_CLASSIFIER_COLUMNS)


class LegacyEazyRegressor(_EazyRegressor):
    """Deprecated 0.2.x-compatible regressor."""

    def __init__(self, regressors="all", save_dir=None, sort_by="rmse"):
        super().__init__(
            estimators="default" if regressors == "all" else regressors,
            sort_by=_legacy_sort(sort_by),
            save_dir=save_dir or None,
            cv=None,
        )
        self.regressors = regressors

    def fit(self, X_train, X_test, y_train, y_test):
        """Deprecated. Returns ``(models, predictions, results)`` as 0.2.x did."""
        warnings.warn(_MIGRATION, DeprecationWarning, stacklevel=2)
        super().fit(X_train, y_train, X_test=X_test, y_test=y_test)
        return (
            self.models_,
            self.predictions_,
            _legacy_frame(self.results_, _LEGACY_REGRESSOR_COLUMNS),
        )

    def fitVotingEnsemble(self, model_dict=None, model_results=None, num_models=5):
        """Deprecated alias for :meth:`fit_voting_ensemble`."""
        warnings.warn(
            "fitVotingEnsemble is deprecated; use fit_voting_ensemble(n_models=...). "
            "This shim will be removed in 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        ensemble = self.fit_voting_ensemble(n_models=num_models)
        return ensemble, _ensemble_frame(self, ensemble, _LEGACY_REGRESSOR_COLUMNS)


def _legacy_sort(sort_by: str) -> str:
    """Map 0.2.x metric keys onto the current names."""
    return {
        "accuracy": "accuracy",
        "f1_score": "f1",
        "roc_score": "roc_auc",
        "rmse": "rmse",
        "r_squared": "r2",
    }.get(sort_by, sort_by)


def _ensemble_frame(parent, ensemble, columns: dict[str, str]) -> pd.DataFrame:
    scores = {}
    for key, label in columns.items():
        spec = parent.metrics_.get(key)
        if spec is None:
            continue
        try:
            scores[label] = spec.display(
                spec.scorer(ensemble, parent._X_train, parent._y_train)
            )
        except Exception:
            scores[label] = float("nan")
    row = {"Models": " ".join(name for name, _ in ensemble.estimators), **scores}
    return pd.DataFrame(row, index=[0])
