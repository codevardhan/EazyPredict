"""Saving fitted models and their provenance.

0.2.x opened the pickle file without a context manager and never recorded what
produced the models. A leaderboard you cannot reproduce six months later is not
evidence of anything, so the sidecar metadata is part of the feature.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from eazypredict import EazyClassifier, EazyRegressor

PAIR = ["LogisticRegression", "DecisionTreeClassifier"]


def _fit(binary_data, save_dir, estimators=PAIR, **kwargs):
    X, y = binary_data
    return EazyClassifier(
        estimators=estimators, cv=None, random_state=0, save_dir=save_dir, **kwargs
    ).fit(X, y)


def test_nothing_is_written_without_save_dir(binary_data, tmp_path):
    X, y = binary_data
    EazyClassifier(estimators=PAIR, cv=None, random_state=0).fit(X, y)
    assert not list(tmp_path.iterdir())


def test_one_file_per_fitted_model(binary_data, tmp_save_dir):
    _fit(binary_data, tmp_save_dir)
    for name in PAIR:
        assert (tmp_save_dir / f"{name}.joblib").is_file()


def test_the_save_directory_is_created(binary_data, tmp_path):
    nested = tmp_path / "a" / "b" / "models"
    _fit(binary_data, nested)
    assert nested.is_dir()


def test_a_saved_model_round_trips_and_still_predicts(binary_data, tmp_save_dir):
    """joblib instead of bare pickle, and the file must actually reload - the 0.2.x
    unclosed writer could leave a truncated file on some platforms."""
    X, _ = binary_data
    clf = _fit(binary_data, tmp_save_dir)
    reloaded = joblib.load(tmp_save_dir / "LogisticRegression.joblib")
    np.testing.assert_array_equal(
        reloaded.predict(X), clf.models_["LogisticRegression"].predict(X)
    )


def test_a_saved_model_is_the_whole_pipeline(messy_frame, tmp_save_dir):
    """Saving the bare estimator would be useless: the caller has no way to rebuild
    the fitted preprocessor that made its inputs numeric."""
    X, y = messy_frame
    EazyClassifier(
        estimators=["LogisticRegression"],
        cv=None,
        random_state=0,
        save_dir=tmp_save_dir,
    ).fit(X, y)
    reloaded = joblib.load(tmp_save_dir / "LogisticRegression.joblib")
    assert "preprocessor" in reloaded.named_steps
    # Raw messy frame in, predictions out - no manual preprocessing required.
    assert len(reloaded.predict(X)) == len(y)


def test_the_leaderboard_is_written_as_csv(binary_data, tmp_save_dir):
    clf = _fit(binary_data, tmp_save_dir)
    frame = pd.read_csv(tmp_save_dir / "results.csv", index_col=0)
    assert list(frame.index) == list(clf.results_.index)
    assert frame.loc[clf.best_model_name_, "accuracy"] > 0


def test_metadata_records_enough_to_reproduce_the_run(binary_data, tmp_save_dir):
    clf = _fit(binary_data, tmp_save_dir)
    metadata = json.loads((tmp_save_dir / "metadata.json").read_text())

    assert metadata["estimator"] == "EazyClassifier"
    assert metadata["random_state"] == 0
    assert metadata["sort_by"] == clf.sort_by
    assert metadata["best_model"] == clf.best_model_name_
    assert metadata["n_features"] == clf._X_train.shape[1]
    assert metadata["created"]


def test_metadata_pins_the_library_versions(binary_data, tmp_save_dir):
    """Scores drift between scikit-learn releases. Without the versions, a saved
    leaderboard cannot be checked against a later rerun."""
    import sklearn

    _fit(binary_data, tmp_save_dir)
    versions = json.loads((tmp_save_dir / "metadata.json").read_text())["versions"]
    assert versions["scikit-learn"] == sklearn.__version__
    assert versions["numpy"] == np.__version__
    assert versions["pandas"] == pd.__version__
    assert versions["eazypredict"]


def test_metadata_is_json_serialisable_with_a_splitter_object(
    binary_data, tmp_save_dir
):
    """`cv` may be a splitter instance, which `json.dumps` cannot encode."""
    from sklearn.model_selection import KFold

    X, y = binary_data
    EazyClassifier(
        estimators=["LogisticRegression"],
        cv=KFold(n_splits=2),
        random_state=0,
        save_dir=tmp_save_dir,
    ).fit(X, y)
    metadata = json.loads((tmp_save_dir / "metadata.json").read_text())
    assert "KFold" in metadata["cv"]


def test_failed_models_are_not_saved(binary_data, tmp_save_dir, failing_estimator):
    _fit(
        binary_data,
        tmp_save_dir,
        estimators=[("boom", failing_estimator), *PAIR],
        on_error="ignore",
    )
    assert not (tmp_save_dir / "boom.joblib").exists()
    assert (tmp_save_dir / "LogisticRegression.joblib").is_file()


def test_the_regressor_persists_too(regression_data, tmp_save_dir):
    X, y = regression_data
    EazyRegressor(
        estimators=["LinearRegression"],
        cv=None,
        random_state=0,
        save_dir=tmp_save_dir,
    ).fit(X, y)
    assert (tmp_save_dir / "LinearRegression.joblib").is_file()
    metadata = json.loads((tmp_save_dir / "metadata.json").read_text())
    assert metadata["estimator"] == "EazyRegressor"
