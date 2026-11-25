"""Metric registry.

Metrics are built on scikit-learn *scorer* objects rather than bare metric functions.
This matters: a scorer receives the fitted estimator and decides for itself how to
obtain a prediction, so ``roc_auc`` automatically uses ``predict_proba`` (falling back
to ``decision_function``). Computing ROC AUC from hard class labels - as 0.2.x did -
discards the ranking information the statistic is defined on, and produces a number
that is both wrong and capable of reordering the leaderboard.

scikit-learn scorers follow a uniform "greater is better" convention, so metrics that
are naturally minimised (log loss, RMSE) are returned negated. We store that convention
in `MetricSpec.greater_is_better` and flip the sign for display, so users see a
positive RMSE while sorting stays uniform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.metrics import get_scorer

__all__ = [
    "MetricSpec",
    "classification_metrics",
    "regression_metrics",
    "resolve_metrics",
]


@dataclass(frozen=True)
class MetricSpec:
    """A named metric and the direction that counts as an improvement."""

    name: str
    scorer: Callable
    greater_is_better: bool = True

    def display(self, raw: float) -> float:
        """Convert a raw scorer value into the number a user expects to read."""
        return raw if self.greater_is_better else -raw


def _spec(name: str, scorer_name: str, greater_is_better: bool = True) -> MetricSpec:
    return MetricSpec(name, get_scorer(scorer_name), greater_is_better)


def classification_metrics(n_classes: int) -> dict[str, MetricSpec]:
    """Default classification metrics.

    ``balanced_accuracy`` and ``matthews_corrcoef`` are included because plain accuracy
    is misleading on imbalanced targets, which is exactly when a model-comparison tool
    is most likely to be trusted blindly.

    Args:
        n_classes: Number of classes in the training target. Determines whether ROC AUC
            is computed directly or one-vs-rest.
    """
    multiclass = n_classes > 2
    metrics = [
        _spec("accuracy", "accuracy"),
        _spec("balanced_accuracy", "balanced_accuracy"),
        _spec("f1", "f1_weighted"),
        _spec("precision", "precision_weighted"),
        _spec("recall", "recall_weighted"),
        _spec("roc_auc", "roc_auc_ovr_weighted" if multiclass else "roc_auc"),
        _spec("log_loss", "neg_log_loss", greater_is_better=False),
        _spec("matthews_corrcoef", "matthews_corrcoef"),
    ]
    return {m.name: m for m in metrics}


def regression_metrics() -> dict[str, MetricSpec]:
    """Default regression metrics."""
    metrics = [
        _spec("rmse", "neg_root_mean_squared_error", greater_is_better=False),
        _spec("mae", "neg_mean_absolute_error", greater_is_better=False),
        _spec("r2", "r2"),
        _spec("mape", "neg_mean_absolute_percentage_error", greater_is_better=False),
        _spec("explained_variance", "explained_variance"),
    ]
    return {m.name: m for m in metrics}


def resolve_metrics(
    requested: list | dict | None, defaults: dict[str, MetricSpec]
) -> dict[str, MetricSpec]:
    """Combine user-supplied metrics with the defaults.

    Args:
        requested: ``None`` for the defaults, a list of default metric names to keep,
            or a mapping of name to ``MetricSpec`` / scorer callable.
        defaults: The default registry for the task.

    Raises:
        ValueError: if a requested name is not a known default.
    """
    if requested is None:
        return dict(defaults)

    if isinstance(requested, dict):
        resolved = {}
        for name, value in requested.items():
            if isinstance(value, MetricSpec):
                resolved[name] = value
            elif isinstance(value, str):
                resolved[name] = _spec(name, value)
            else:
                resolved[name] = MetricSpec(name, value)
        return resolved

    unknown = [name for name in requested if name not in defaults]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(defaults)}."
        )
    return {name: defaults[name] for name in requested}
