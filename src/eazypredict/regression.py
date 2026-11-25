"""The regression front end."""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold, train_test_split

from .base import _as_1d_target, _BaseEazy, _n_features
from .metrics import regression_metrics, resolve_metrics

__all__ = ["EazyRegressor"]


class EazyRegressor(RegressorMixin, _BaseEazy):
    """Fit many regressors and compare them on a single leaderboard.

    Args:
        estimators: ``"default"``, ``"all"``, a list of scikit-learn estimator names,
            or a list of ``(name, estimator)`` pairs.
        metrics: ``None`` for the defaults, a list of default metric names to keep, or
            a mapping of name to scorer.
        sort_by: Metric used to order the leaderboard. Row 0 is always the best model,
            whether the metric is maximised (``r2``) or minimised (``rmse``).
        preprocess: ``True`` to build an automatic pipeline, ``False`` to pass the data
            through untouched, or a transformer to use instead.
        cv: Number of cross-validation folds, a splitter, or ``None`` for a single
            holdout split.
        test_size: Holdout fraction, used when ``cv`` is ``None``.
        random_state: Seed threaded into every estimator that accepts one.
        verbose: ``0`` silent, ``1`` progress bar.
        on_error: ``"warn"``, ``"raise"`` or ``"ignore"`` when a model fails to fit.
        save_dir: Directory to write fitted models, results and provenance into.

    Attributes:
        results_ (pandas.DataFrame): Leaderboard, best model first.
        models_ (dict): Fitted pipelines by name.
        predictions_ (dict): Test-set predictions by name.
        errors_ (dict): Error strings for models that failed.
        best_estimator_: The highest-scoring fitted pipeline.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> from eazypredict import EazyRegressor
        >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
        >>> reg = EazyRegressor(
        ...     estimators=["LinearRegression", "DecisionTreeRegressor"],
        ...     random_state=0,
        ... ).fit(X, y)
        >>> reg.results_.index[0] == reg.best_model_name_
        True
    """

    _kind = "regressor"

    def __init__(
        self,
        estimators="default",
        *,
        metrics=None,
        sort_by="rmse",
        preprocess=True,
        cv=5,
        test_size=0.25,
        random_state=None,
        verbose=0,
        on_error="warn",
        save_dir=None,
    ):
        self.estimators = estimators
        self.metrics = metrics
        self.sort_by = sort_by
        self.preprocess = preprocess
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.on_error = on_error
        self.save_dir = save_dir

    # ----------------------------------------------------------------- internals

    def _default_metrics(self, y):
        return resolve_metrics(self.metrics, regression_metrics())

    def _cv_splitter(self, y):
        if not isinstance(self.cv, int):
            return self.cv
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _split(self, X, y):
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    # ---------------------------------------------------------------------- API

    def fit(self, X, y, *, X_test=None, y_test=None):
        """Fit and score every candidate regressor.

        Args:
            X: Training features.
            y: Training target.
            X_test: Optional explicit test features.
            y_test: Optional explicit test target.

        Returns:
            self
        """
        y = _as_1d_target(y)
        if y_test is not None:
            y_test = _as_1d_target(y_test)

        self.n_features_in_ = _n_features(X)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self._run(X, y, X_test, y_test)
        return self

    def fit_voting_ensemble(self, n_models=5, *, weights=None):
        """Average the predictions of the top models.

        Args:
            n_models: How many of the leading models to include.
            weights: Optional per-model weights.

        Returns:
            The fitted :class:`~sklearn.ensemble.VotingRegressor`.
        """
        self._check_fitted()
        selected = [
            (name, clone(self.models_[name]))
            for name in self.results_.index[:n_models]
            if name in self.models_
        ]
        if not selected:
            raise RuntimeError("No fitted models available to build an ensemble.")

        ensemble = VotingRegressor(selected, weights=weights)
        ensemble.fit(self._X_train, self._y_train)
        self.ensemble_ = ensemble
        return ensemble
