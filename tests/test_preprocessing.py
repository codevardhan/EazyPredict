"""Automatic preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from eazypredict.preprocessing import HIGH_CARDINALITY_THRESHOLD, make_preprocessor


def test_ndarray_gets_a_plain_pipeline(binary_data):
    """`make_column_selector` requires a DataFrame and raises on arrays, and an
    ndarray cannot hold mixed dtypes anyway, so there is nothing to split on."""
    X, _ = binary_data
    assert isinstance(make_preprocessor(X), Pipeline)


def test_dataframe_gets_a_column_transformer(messy_frame):
    X, _ = messy_frame
    assert isinstance(make_preprocessor(X), ColumnTransformer)


def test_messy_frame_becomes_dense_finite_numeric(messy_frame):
    X, _ = messy_frame
    transformed = make_preprocessor(X).fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert np.isfinite(transformed).all()
    assert transformed.shape[0] == len(X)


def test_output_is_never_sparse(messy_frame):
    """GaussianNB, HistGradientBoosting and the discriminant analyses all reject
    sparse input, so one-hot encoding must stay dense."""
    import scipy.sparse as sp

    X, _ = messy_frame
    assert not sp.issparse(make_preprocessor(X).fit_transform(X))


def test_unseen_category_at_transform_time(messy_frame):
    """Real test sets contain categories the training set never saw."""
    X, _ = messy_frame
    preprocessor = make_preprocessor(X).fit(X)
    unseen = X.copy()
    unseen.loc[unseen.index[0], "cat_object"] = "a_brand_new_value"
    assert np.isfinite(preprocessor.transform(unseen)).all()


def test_high_cardinality_column_warns():
    """A Titanic 'Cabin'-like column silently explodes into hundreds of features."""
    frame = pd.DataFrame(
        {
            "numeric": np.arange(120, dtype=float),
            "identifier": [f"id_{i}" for i in range(120)],
        }
    )
    with pytest.warns(UserWarning, match="High-cardinality"):
        make_preprocessor(frame)


def test_max_categories_suppresses_the_warning_and_caps_width():
    frame = pd.DataFrame(
        {
            "numeric": np.arange(120, dtype=float),
            "identifier": [f"id_{i}" for i in range(120)],
        }
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        preprocessor = make_preprocessor(frame, max_categories=5)
    assert preprocessor.fit_transform(frame).shape[1] <= 1 + 5


def test_cardinality_threshold_is_not_tripped_by_ordinary_columns(messy_frame):
    X, _ = messy_frame
    assert X["cat_object"].nunique() < HIGH_CARDINALITY_THRESHOLD


def test_scaling_can_be_disabled(binary_data):
    X, _ = binary_data
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    unscaled = make_preprocessor(frame, scale=False).fit_transform(frame)
    assert unscaled.std(axis=0).max() == pytest.approx(frame.std(ddof=0).max())


def test_preprocessor_is_unfitted(messy_frame):
    """Callers clone this per model; returning something already fitted would leak
    training statistics across cross-validation folds."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    X, _ = messy_frame
    with pytest.raises(NotFittedError):
        check_is_fitted(make_preprocessor(X))
