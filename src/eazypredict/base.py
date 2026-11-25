"""Shared machinery for the classifier and regressor front ends."""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from .estimators import resolve_estimators
from .metrics import MetricSpec
from .preprocessing import make_preprocessor

__all__ = ["ModelOutcome", "set_random_state"]

logger = logging.getLogger("eazypredict")


def _shape_of(X) -> tuple[int, ...]:
    """The shape of ``X`` without routing through ``np.shape``.

    ``np.shape`` and ``np.ndim`` dispatch via ``__array_function__``, which a strict
    array-like duck type is entitled to refuse - and some do, with a ``TypeError``
    that has nothing to do with the caller's data. Reading ``.shape`` directly, and
    converting only when there is none, keeps those inputs working.
    """
    shape = getattr(X, "shape", None)
    if shape is not None:
        return tuple(shape)
    return np.asarray(X).shape


def _n_features(X) -> int:
    shape = _shape_of(X)
    return int(shape[1]) if len(shape) > 1 else 1


def _tolerant_scorer(scorer):
    """Wrap a scorer so an inapplicable metric yields NaN instead of killing the model.

    ``cross_validate(error_score=...)`` only covers exceptions raised during *fit*; one
    raised while scoring still propagates. Without this, asking ``RidgeClassifier`` for
    ``neg_log_loss`` - which needs a ``predict_proba`` it does not have - throws away
    its accuracy, F1 and everything else, and drops the model from the leaderboard
    entirely. A metric a model cannot provide should cost one cell, not one row.
    """

    def score(estimator, X, y, **kwargs) -> float:
        try:
            return scorer(estimator, X, y, **kwargs)
        except Exception:
            return np.nan

    return score


def _as_1d_target(y) -> np.ndarray:
    """Flatten a single-column target to 1-D, warning as scikit-learn does.

    ``df[["target"]]`` and ``y.reshape(-1, 1)`` are common enough that rejecting them
    would be hostile, but silently accepting them hides the case where the caller
    meant a multi-output target and will now get a single-output model instead.

    Raises:
        ValueError: if ``y`` is ``None``. Caught here because ``np.asarray(None)``
            produces a 0-d object array, and the resulting complaint about an
            "unknown label type" gives no hint that the target was simply omitted.
    """
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None."
        )
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        warnings.warn(
            # Wording matches scikit-learn's verbatim: its conformance checks match
            # on this exact phrase.
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,), for example using ravel().",
            DataConversionWarning,
            stacklevel=3,
        )
        y = y.ravel()
    return y


def set_random_state(estimator: BaseEstimator, random_state) -> BaseEstimator:
    """Seed every ``random_state`` parameter reachable from ``estimator``.

    Without this, the majority of the default estimators are stochastic and a
    leaderboard cannot be reproduced - which quietly undermines any comparison drawn
    from it.

    Args:
        estimator: Estimator to seed, modified in place.
        random_state: Seed, ``RandomState``, or ``None`` to leave the estimator alone.

    Returns:
        The same estimator, for chaining.
    """
    if random_state is None:
        return estimator

    rng = check_random_state(random_state)
    to_set = {
        key: rng.randint(np.iinfo(np.int32).max)
        for key in estimator.get_params(deep=True)
        if key == "random_state" or key.endswith("__random_state")
    }
    if to_set:
        estimator.set_params(**to_set)
    return estimator


@dataclass
class ModelOutcome:
    """The result of evaluating one candidate model."""

    name: str
    estimator: Pipeline | None = None
    scores: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    fit_time: float = 0.0
    score_time: float = 0.0
    status: str = "ok"
    error: str | None = None
    warnings: tuple[str, ...] = ()


class _BaseEazy(BaseEstimator):
    """Common implementation shared by :class:`EazyClassifier` and
    :class:`EazyRegressor`.

    Subclasses supply the task kind, the default metric registry and the default
    ``sort_by``; everything else - resolution, preprocessing, evaluation, error
    isolation, persistence and leaderboard assembly - lives here.
    """

    _kind: str = ""
    _estimator_param: str = "estimators"

    # ------------------------------------------------------------------ helpers

    def _default_metrics(self, y) -> dict[str, MetricSpec]:
        raise NotImplementedError

    def _cv_splitter(self, y):
        raise NotImplementedError

    def _build_pipeline(self, estimator: BaseEstimator, X) -> Pipeline:
        """Wrap an estimator in its own cloned preprocessor.

        Each model gets a *separate* preprocessor instance so that it is refitted
        inside every cross-validation fold. Sharing one fitted preprocessor across
        folds would leak test-fold statistics into training.
        """
        steps: list[tuple[str, BaseEstimator]] = []
        if self.preprocess is not False:
            if self.preprocess is True:
                preprocessor = make_preprocessor(X)
            else:
                preprocessor = clone(self.preprocess)
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def _sorted_names(
        self, scores: dict[str, dict[str, float]], metrics: dict[str, MetricSpec]
    ) -> list[str]:
        """Order model names best-first by ``self.sort_by``.

        Honouring ``greater_is_better`` is what lets accuracy and log loss coexist;
        0.2.x hard-coded the direction and indexed a tuple out of range for every
        metric except the first.
        """
        if self.sort_by not in metrics:
            raise ValueError(
                f"Invalid sort_by {self.sort_by!r}. "
                f"Available metrics: {', '.join(metrics)}."
            )
        spec = metrics[self.sort_by]

        def key(name: str) -> tuple[int, float]:
            value = scores.get(name, {}).get(self.sort_by, np.nan)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                # Failed or unscorable models sort last whatever the direction.
                return (1, 0.0)
            return (0, -value if spec.greater_is_better else value)

        return sorted(scores, key=key)

    # -------------------------------------------------------------- evaluation

    def _evaluate(
        self,
        name: str,
        pipeline: Pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        metrics: dict[str, MetricSpec],
    ) -> ModelOutcome:
        """Fit and score a single candidate, isolating any failure.

        The *fit* is what fails in practice - a model that cannot handle the data,
        or runs out of memory. 0.2.x wrapped only the metric computation, so one bad
        estimator aborted the entire run.
        """
        outcome = ModelOutcome(name=name)
        scoring = {key: _tolerant_scorer(spec.scorer) for key, spec in metrics.items()}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if self.cv is None:
                    started = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    outcome.fit_time = time.perf_counter() - started

                    started = time.perf_counter()
                    outcome.predictions = pipeline.predict(X_test)
                    for key, spec in metrics.items():
                        try:
                            outcome.scores[key] = spec.display(
                                spec.scorer(pipeline, X_test, y_test)
                            )
                        except Exception:
                            # e.g. OVR AUC from an unnormalised decision_function.
                            outcome.scores[key] = np.nan
                    outcome.score_time = time.perf_counter() - started
                    outcome.estimator = pipeline
                else:
                    report = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv=self._cv_splitter(y_train),
                        error_score=np.nan,
                        return_train_score=False,
                    )
                    outcome.fit_time = float(np.sum(report["fit_time"]))
                    outcome.score_time = float(np.sum(report["score_time"]))
                    for key, spec in metrics.items():
                        fold_scores = report[f"test_{key}"]
                        outcome.scores[key] = spec.display(np.nanmean(fold_scores))
                        outcome.scores[f"{key}_std"] = float(np.nanstd(fold_scores))

                    if np.all(np.isnan(report[f"test_{self.sort_by}"])):
                        raise RuntimeError(
                            "every cross-validation fold failed for this estimator"
                        )
                    # Refit on the full training set so the model is usable.
                    pipeline.fit(X_train, y_train)
                    outcome.estimator = pipeline
                    outcome.predictions = pipeline.predict(X_test)
            except Exception as exc:
                outcome.status = "failed"
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.scores = dict.fromkeys(metrics, np.nan)
                if self.on_error == "raise":
                    raise
                if self.on_error == "warn":
                    logger.warning("%s failed: %s", name, outcome.error)

            outcome.warnings = tuple(
                sorted({f"{w.category.__name__}: {w.message}" for w in caught})
            )
        return outcome

    # --------------------------------------------------------------------- fit

    def _run(self, X, y, X_test, y_test):
        """Shared ``fit`` body. Returns nothing; populates fitted attributes."""
        for attribute in ("sort_by", "on_error"):
            if getattr(self, attribute) is None:
                raise ValueError(f"{attribute} must not be None")
        if self.on_error not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"Invalid on_error {self.on_error!r}; "
                "expected 'warn', 'raise' or 'ignore'."
            )

        metrics = self._default_metrics(y)
        candidates = resolve_estimators(
            getattr(self, self._estimator_param), self._kind
        )

        if X_test is None:
            X_train, X_test, y_train, y_test = self._split(X, y)
        else:
            X_train, y_train = X, y
            if y_test is None:
                raise ValueError("y_test must be provided when X_test is given.")

        # Retained so ensembles can be built, and predictions reproduced, after the
        # fact without refitting.
        self._X_train, self._y_train = X_train, y_train
        self._X_test, self._y_test = X_test, y_test

        outcomes: dict[str, ModelOutcome] = {}
        iterator = candidates
        if self.verbose:
            from tqdm.auto import tqdm

            iterator = tqdm(candidates, desc=self._kind)

        for name, estimator in iterator:
            set_random_state(estimator, self.random_state)
            pipeline = self._build_pipeline(estimator, X_train)
            set_random_state(pipeline, self.random_state)
            outcomes[name] = self._evaluate(
                name, pipeline, X_train, y_train, X_test, y_test, metrics
            )
            logger.info(
                "%s: %s in %.3fs", name, outcomes[name].status, outcomes[name].fit_time
            )

        scores = {name: outcome.scores for name, outcome in outcomes.items()}
        order = self._sorted_names(scores, metrics)

        self.metrics_ = metrics
        self.outcomes_ = {name: outcomes[name] for name in order}
        self.results_ = self._build_results(order, outcomes, metrics)
        self.models_ = {
            name: outcomes[name].estimator
            for name in order
            if outcomes[name].estimator is not None
        }
        self.predictions_ = {
            name: outcomes[name].predictions
            for name in order
            if outcomes[name].predictions is not None
        }
        self.errors_ = {
            name: outcomes[name].error
            for name in order
            if outcomes[name].error is not None
        }

        if not self.models_:
            raise RuntimeError(
                "Every candidate model failed. Inspect `errors_` for details, or pass "
                "on_error='raise' to surface the first exception."
            )

        self.best_model_name_ = next(iter(self.models_))
        self.best_estimator_ = self.models_[self.best_model_name_]

        if self.save_dir is not None:
            self._persist(X_train, y_train)

    def _build_results(
        self,
        order: list[str],
        outcomes: dict[str, ModelOutcome],
        metrics: dict[str, MetricSpec],
    ) -> pd.DataFrame:
        columns = list(metrics)
        if self.cv is not None:
            columns += [f"{name}_std" for name in metrics]
        rows = {
            name: {
                **{
                    column: outcomes[name].scores.get(column, np.nan)
                    for column in columns
                },
                "fit_time": outcomes[name].fit_time,
                "score_time": outcomes[name].score_time,
                "status": outcomes[name].status,
            }
            for name in order
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "model"
        return frame

    def _persist(self, X_train, y_train) -> None:
        """Write fitted models plus provenance.

        joblib replaces pickle, and the sidecar metadata records the library versions
        and seed - without them a saved leaderboard is not reproducible six months on.
        """
        import joblib
        import sklearn

        directory = Path(self.save_dir)
        directory.mkdir(parents=True, exist_ok=True)

        for name, model in self.models_.items():
            joblib.dump(model, directory / f"{name}.joblib")

        self.results_.to_csv(directory / "results.csv")
        metadata: dict[str, Any] = {
            "created": datetime.now(timezone.utc).isoformat(),
            "estimator": type(self).__name__,
            "random_state": self.random_state,
            "cv": self.cv if isinstance(self.cv, (int, type(None))) else str(self.cv),
            "sort_by": self.sort_by,
            "best_model": self.best_model_name_,
            "n_samples": int(_shape_of(X_train)[0]),
            "n_features": _n_features(X_train),
            "versions": {
                "eazypredict": _package_version(),
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ------------------------------------------------------------------ public

    def _check_fitted(self) -> None:
        if not hasattr(self, "results_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet. "
                "Call fit(X, y) before using this method."
            )

    def predict(self, X):
        """Predict with the best model on the leaderboard."""
        self._check_fitted()
        return self.best_estimator_.predict(X)

    def to_markdown(self, **kwargs) -> str:
        """Render the leaderboard as a Markdown table."""
        self._check_fitted()
        return self.results_.to_markdown(**kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render the leaderboard as a LaTeX table."""
        self._check_fitted()
        return self.results_.to_latex(**kwargs)


def _package_version() -> str:
    from . import __version__

    return __version__
