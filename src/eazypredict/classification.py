"""The classification front end."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets

from .base import _as_1d_target, _BaseEazy, _n_features
from .metrics import classification_metrics, resolve_metrics

__all__ = ["EazyClassifier"]


class EazyClassifier(ClassifierMixin, _BaseEazy):
    """Fit many classifiers and compare them on a single leaderboard.

    Every candidate is wrapped in its own preprocessing pipeline, so mixed-dtype
    DataFrames with missing values work without manual cleaning.

    Args:
        estimators: ``"default"``, ``"all"``, a list of scikit-learn estimator names,
            or a list of ``(name, estimator)`` pairs.
        metrics: ``None`` for the defaults, a list of default metric names to keep, or
            a mapping of name to scorer.
        sort_by: Metric used to order the leaderboard. Row 0 is always the best model.
        preprocess: ``True`` to build an automatic pipeline, ``False`` to pass the data
            through untouched, or a transformer to use instead.
        cv: Number of cross-validation folds, a splitter, or ``None`` for a single
            holdout split. Defaults to 5: a single split on a few hundred rows has
            enough variance that the resulting ranking is largely noise.
        test_size: Holdout fraction, used when ``cv`` is ``None`` and no explicit test
            set is supplied.
        stratify: Whether to preserve class proportions when splitting.
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
        classes_ (numpy.ndarray): The original class labels.

    Example:
        >>> from sklearn.datasets import make_classification
        >>> from eazypredict import EazyClassifier
        >>> X, y = make_classification(n_samples=60, n_features=5, random_state=0)
        >>> clf = EazyClassifier(
        ...     estimators=["LogisticRegression", "DecisionTreeClassifier"],
        ...     random_state=0,
        ... ).fit(X, y)
        >>> list(clf.results_.columns)[:2]
        ['accuracy', 'balanced_accuracy']
    """

    _kind = "classifier"

    def __init__(
        self,
        estimators="default",
        *,
        metrics=None,
        sort_by="accuracy",
        preprocess=True,
        cv=5,
        test_size=0.25,
        stratify=True,
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
        self.stratify = stratify
        self.random_state = random_state
        self.verbose = verbose
        self.on_error = on_error
        self.save_dir = save_dir

    # ----------------------------------------------------------------- internals

    def _default_metrics(self, y):
        return resolve_metrics(self.metrics, classification_metrics(len(self.classes_)))

    def _cv_splitter(self, y):
        if not isinstance(self.cv, int):
            return self.cv
        if self.stratify:
            return StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

    # ---------------------------------------------------------------------- API

    def fit(self, X, y, *, X_test=None, y_test=None):
        """Fit and score every candidate classifier.

        Args:
            X: Training features.
            y: Training target. Labels of any type are accepted and preserved.
            X_test: Optional explicit test features.
            y_test: Optional explicit test target.

        Returns:
            self
        """
        y = _as_1d_target(y)

        # Without this, a continuous target is silently label-encoded into one class
        # per distinct float, yielding a nonsense leaderboard instead of telling the
        # caller they wanted EazyRegressor.
        check_classification_targets(y)

        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        y_encoded = self._label_encoder.transform(y)

        if y_test is not None:
            y_test = self._label_encoder.transform(_as_1d_target(y_test))

        self.n_features_in_ = _n_features(X)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self._run(X, y_encoded, X_test, y_test)
        return self

    def predict(self, X):
        """Predict labels with the best model, in the caller's original label space."""
        self._check_fitted()
        return self._label_encoder.inverse_transform(self.best_estimator_.predict(X))

    def predict_proba(self, X):
        """Class probabilities from the best model."""
        self._check_fitted()
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(
                f"{self.best_model_name_} does not support predict_proba."
            )
        return self.best_estimator_.predict_proba(X)

    def fit_voting_ensemble(self, n_models=5, *, voting="soft", weights=None):
        """Combine the top models into a voting ensemble.

        Args:
            n_models: How many of the leading models to include.
            voting: ``"soft"`` to average probabilities, ``"hard"`` to count votes.
                Soft voting usually wins, and is used by default; it falls back to hard
                voting if any selected model cannot produce probabilities.
            weights: Optional per-model weights.

        Returns:
            The fitted :class:`~sklearn.ensemble.VotingClassifier`.
        """
        self._check_fitted()
        # Drive selection from the leaderboard order so the ensemble honours sort_by.
        selected = [
            (name, clone(self.models_[name]))
            for name in self.results_.index[:n_models]
            if name in self.models_
        ]
        if not selected:
            raise RuntimeError("No fitted models available to build an ensemble.")

        if voting == "soft":
            lacking = [
                name for name, model in selected if not hasattr(model, "predict_proba")
            ]
            if lacking:
                warnings.warn(
                    f"Falling back to hard voting: {', '.join(lacking)} cannot produce "
                    "probabilities.",
                    UserWarning,
                    stacklevel=2,
                )
                voting = "hard"

        ensemble = VotingClassifier(selected, voting=voting, weights=weights)
        ensemble.fit(self._X_train, self._y_train)
        self.ensemble_ = ensemble
        return ensemble
