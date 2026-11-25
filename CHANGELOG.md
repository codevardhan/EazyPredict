# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.3.0

### Added

- Automatic preprocessing: mixed-dtype DataFrames with missing values and categorical
  columns now work without manual cleaning. Each candidate model gets its own cloned
  preprocessor inside a `Pipeline`, so cross-validation does not leak.
- Cross-validated leaderboards (`cv=5` by default) reporting a mean and standard
  deviation per metric. A single split on a few hundred rows makes the ranking mostly
  noise.
- Per-model error isolation: a model that fails costs one row, not the whole run.
  Inspect `errors_`, or pass `on_error="raise"`.
- `random_state` seeds every candidate estimator, making a leaderboard reproducible.
- `save_dir` writes each fitted pipeline via `joblib` plus a `metadata.json` recording
  the seed, data shape and library versions.
- `py.typed`, a test suite, CI, and `ruff`/`mypy` configuration.

### Fixed

- **`fit` had the argument order `(X_train, X_test, y_train, y_test)` while every
  published example called `(X_train, y_train, X_test, y_test)`**, so half of all users
  silently trained on their test set with no error. The new API is sklearn-compliant
  `fit(X, y)`. The legacy entry point is preserved, with its *implemented* order
  intact — reinterpreting those arguments would have silently changed what existing
  callers trained on.
- **ROC AUC was computed from hard class labels**, discarding the ranking information
  the statistic is defined on. Metrics are now scikit-learn scorers, which obtain
  probabilities themselves. The reported figure moved in both directions on noisy
  data, so this could reorder the leaderboard.
- Sorting indexed a 2-tuple by metric position: `sort_by="f1_score"` raised
  `IndexError`, `"accuracy"` worked only by accident, and any `None` metric raised
  `TypeError`. A `self.sort_key` typo inside an `elif` condition also made every
  invalid `sort_by` raise `AttributeError` instead of the intended error.
- Estimator resolution mutated `self.classifiers` in place, so a second `fit()` on the
  same instance produced an empty leaderboard. Resolution is now a pure function.
- Only metric computation was wrapped in `try`/`except`; `model.fit()` was unguarded,
  so one bad estimator aborted an entire multi-minute run.
- A metric a model cannot provide no longer discards the model. `RidgeClassifier`,
  `SGDClassifier` and `SVC` have no `predict_proba`, and because
  `cross_validate(error_score=...)` covers only failures in `fit`, requesting log loss
  removed them from the leaderboard entirely under the default `cv`.
- Estimator discovery had no `type_filter`, so transformers such as `StandardScaler`
  could be instantiated as classifiers. Meta-estimators needing a base estimator and
  deprecated classes are now also excluded from `"all"`.
- Passing a continuous target to `EazyClassifier` label-encoded every distinct float
  into its own class, yielding a plausible-looking leaderboard built on nonsense. It
  now raises.
- `fit(X, None)` reported an "unknown label type" rather than saying `y` was omitted.
- A column-vector `y` is flattened with a `DataConversionWarning`, as scikit-learn
  does, instead of silently.
- Shape inspection used `np.shape`/`np.ndim`, which dispatch through
  `__array_function__` and fail on strict array-like inputs.
- Models were saved through an unclosed `pickle` writer; `joblib` and a context manager
  are now used.
- **The published package declared no dependencies at all.** `pip install eazypredict`
  installed nothing else, so the first `import eazypredict` failed with
  `ModuleNotFoundError: No module named 'sklearn'`. Dependencies are now declared in
  `pyproject.toml`.
- `requirements.txt` listed `pickle`, a standard-library module with no PyPI
  distribution, which made `pip install -r requirements.txt` fail outright.
- Removed a `.pyc` file that was tracked in version control, and normalised the file
  mode of every tracked file (the whole tree had flipped `644` → `755`).

### Changed

- Packaging migrated from `setup.cfg` to PEP 621 `pyproject.toml`. The version is now
  single-sourced from `pyproject.toml` and read at runtime via `importlib.metadata`;
  previously `setup.cfg` (0.2.0), `__init__.py` (0.1.2) and the published sdist (0.2.1)
  all disagreed.
- `xgboost` and `lightgbm` are now an optional `[boosters]` extra rather than hard
  imports, and `matplotlib` is an optional `[plots]` extra. The core install is light
  and works on Python versions where booster wheels are not yet published.
- Minimum supported Python is now 3.10 (was nominally 3.6).
- `GaussianProcessRegressor` is no longer a default: it is O(n³) in time and O(n²) in
  memory, and exhausts RAM on datasets of only moderate size. `HistGradientBoosting*`
  was added so the core install still has a competitive gradient booster without the
  `[boosters]` extra.

### Deprecated

- `from eazypredict.EazyClassifier import EazyClassifier` and the `fitVotingEnsemble`
  /`classififers` spellings still work but emit a `DeprecationWarning`. They will be
  removed in 0.5.0. Use `from eazypredict import EazyClassifier`.

## [0.1.2] — 2023-01-24

- Released on PyPI.
