"""Automatic preprocessing.

Real datasets arrive with missing values and categorical columns, which crash almost
every scikit-learn estimator. Building a sensible default pipeline is the difference
between a library that works on ``load_breast_cancer`` and one that works on a CSV.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = ["HIGH_CARDINALITY_THRESHOLD", "make_preprocessor"]

#: Above this many distinct values, one-hot encoding a column is usually a mistake.
HIGH_CARDINALITY_THRESHOLD = 50

_CATEGORICAL_DTYPES = ["object", "category", "string", "bool"]


def _warn_on_high_cardinality(X: pd.DataFrame, max_categories: int | None) -> None:
    if max_categories is not None:
        return
    offenders = {
        column: int(X[column].nunique(dropna=True))
        for column in X.select_dtypes(include=_CATEGORICAL_DTYPES).columns
        if X[column].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    }
    if offenders:
        detail = ", ".join(f"{c} ({n} values)" for c, n in offenders.items())
        warnings.warn(
            f"High-cardinality categorical column(s) will be one-hot encoded into "
            f"many features: {detail}. Pass max_categories=... to cap this.",
            UserWarning,
            stacklevel=3,
        )


def make_preprocessor(
    X,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scale: bool = True,
    max_categories: int | None = None,
    min_frequency: int | float | None = None,
) -> BaseEstimator:
    """Build an unfitted preprocessing transformer appropriate for ``X``.

    Numeric columns are imputed and (optionally) standardised. Categorical columns are
    imputed with the most frequent value and one-hot encoded, tolerating categories
    unseen at training time.

    Output is always dense: ``GaussianNB``, ``HistGradientBoosting*`` and the
    discriminant-analysis estimators all reject sparse input.

    Args:
        X: Sample of the training features, used only to detect column types.
        numeric_strategy: ``SimpleImputer`` strategy for numeric columns.
        categorical_strategy: ``SimpleImputer`` strategy for categorical columns.
        scale: Whether to standardise numeric columns.
        max_categories: Cap on one-hot categories per column.
        min_frequency: Minimum count or fraction for a category to get its own column.

    Returns:
        An unfitted transformer. Callers must clone it per model so that nothing is
        shared across cross-validation folds.
    """
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy=numeric_strategy))
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # A plain ndarray cannot carry mixed dtypes, so there is nothing to split on, and
    # make_column_selector requires a DataFrame and raises on arrays.
    if not isinstance(X, pd.DataFrame):
        return numeric_pipeline

    _warn_on_high_cardinality(X, max_categories)

    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy=categorical_strategy, fill_value="missing"),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                    max_categories=max_categories,
                    min_frequency=min_frequency,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=_CATEGORICAL_DTYPES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
