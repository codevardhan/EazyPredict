# EazyPredict

Fit a dozen scikit-learn models on your data and compare them on a single leaderboard.
Useful when you want to know which family of model is worth your time before you commit
to tuning one.

Inspired by [LazyPredict](https://github.com/shankarpandala/lazypredict), with a
narrower model list and a few things it does differently:

- **Cross-validated by default** (`cv=5`), reporting a mean and a standard deviation
  per metric. A single split on a few hundred rows makes the ranking mostly noise.
- **Automatic preprocessing.** Mixed-dtype frames with missing values and categorical
  columns work as-is. Each model gets its own cloned preprocessor inside a `Pipeline`,
  so cross-validation does not leak.
- **A model that fails costs one row, not the run.** Check `errors_`.
- **Reproducible.** `random_state` is threaded into every candidate estimator.
- **Light core.** xgboost, lightgbm and matplotlib are optional extras.

## Installation

```bash
pip install eazypredict                 # core: scikit-learn, pandas, numpy
pip install 'eazypredict[boosters]'     # + xgboost, lightgbm
pip install 'eazypredict[plots]'        # + matplotlib
pip install 'eazypredict[all]'
```

Requires Python 3.10+.

## Classification

```python
from sklearn.datasets import load_breast_cancer
from eazypredict import EazyClassifier

X, y = load_breast_cancer(return_X_y=True)

clf = EazyClassifier(random_state=0).fit(X, y)
print(clf.results_[["accuracy", "f1", "roc_auc", "accuracy_std", "status"]].round(3))
```

```
                                accuracy     f1  roc_auc  accuracy_std status
model
LogisticRegression                 0.981  0.981    0.995         0.012     ok
SVC                                0.981  0.981    0.997         0.012     ok
HistGradientBoostingClassifier     0.979  0.979    0.994         0.017     ok
MLPClassifier                      0.976  0.976    0.994         0.011     ok
LGBMClassifier                     0.974  0.974    0.996         0.023     ok
XGBClassifier                      0.972  0.972    0.993         0.022     ok
SGDClassifier                      0.969  0.969    0.988         0.024     ok
KNeighborsClassifier               0.967  0.967    0.989         0.023     ok
RidgeClassifier                    0.962  0.962    0.993         0.020     ok
RandomForestClassifier             0.962  0.962    0.996         0.009     ok
DecisionTreeClassifier             0.937  0.937    0.932         0.018     ok
GaussianNB                         0.929  0.929    0.988         0.031     ok
```

Row 0 is always the best model. It is refit on the full training set and used for
prediction:

```python
clf.best_model_name_  # 'LogisticRegression'
clf.best_estimator_  # the fitted Pipeline, preprocessing included
clf.models_  # every fitted pipeline, by name

clf.predict(X[:5])
clf.predict_proba(X[:5])
```

ROC AUC is computed from probabilities — or from `decision_function` where that is all
a model offers — not from hard class labels. A model that can supply neither reports
`NaN` for that one metric and keeps its place on the leaderboard.

## Regression

```python
from sklearn.datasets import load_diabetes
from eazypredict import EazyRegressor

X_reg, y_reg = load_diabetes(return_X_y=True)

reg = EazyRegressor(random_state=0).fit(X_reg, y_reg)
print(reg.results_[["rmse", "mae", "r2", "rmse_std", "status"]].round(3))
```

```
                                  rmse     mae     r2  rmse_std status
model
Ridge                           54.084  43.928  0.523     3.867     ok
LinearRegression                54.093  43.925  0.522     3.864     ok
ElasticNet                      55.462  46.502  0.500     3.995     ok
RandomForestRegressor           56.913  46.204  0.470     5.752     ok
KNeighborsRegressor             57.994  46.009  0.455     4.990     ok
HistGradientBoostingRegressor   59.251  46.638  0.425     6.642     ok
LGBMRegressor                   59.717  47.324  0.418     5.565     ok
XGBRegressor                    63.174  50.158  0.343     5.253     ok
NuSVR                           73.395  63.407  0.129     3.616     ok
DecisionTreeRegressor           76.806  60.101  0.031     7.736     ok
MLPRegressor                   106.945  86.631 -0.862     7.265     ok
```

Minimised metrics (`rmse`, `mae`, `mape`, `log_loss`) are reported as positive numbers
and sorted ascending, so row 0 is the best model whichever metric you sort by.

## Messy data

The case most of these tools fall over on: floats with gaps, string columns, nothing
cleaned.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame(
    {
        "age": rng.normal(40, 12, n),
        "city": rng.choice(["London", "Paris", "Tokyo"], n),
        "score": rng.normal(0, 1, n),
    }
)
df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan  # missing numbers
df.loc[rng.choice(n, 15, replace=False), "city"] = None  # missing categories
target = (df["score"] + rng.normal(0, 0.5, n) > 0).astype(int)

messy = EazyClassifier(
    estimators=[
        "LogisticRegression",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
    ],
    random_state=0,
).fit(df, target)
print(messy.results_[["accuracy", "f1", "roc_auc", "status"]].round(3))
```

```
                                accuracy     f1  roc_auc status
model
LogisticRegression                 0.873  0.873    0.953     ok
RandomForestClassifier             0.853  0.853    0.930     ok
HistGradientBoostingClassifier     0.847  0.846    0.943     ok
```

Numeric columns are median-imputed and scaled; categorical columns are
most-frequent-imputed and one-hot encoded. Pass `preprocess=False` to turn this off, or
pass your own transformer to use instead.

## Choosing models

```python
custom = EazyClassifier(
    estimators=["LinearSVC", "NearestCentroid", "AdaBoostClassifier"],
    sort_by="f1",
    random_state=0,
).fit(X, y)
```

`estimators` accepts `"default"`, `"all"` (every applicable scikit-learn estimator), a
list of names, bare instances, or `(name, estimator)` pairs — so a model can be
pre-configured:

```python
from sklearn.ensemble import RandomForestClassifier

EazyClassifier(estimators=[("rf_500", RandomForestClassifier(n_estimators=500))])
```

A misspelled name fails immediately with a suggestion, rather than after a long run.

## Choosing metrics

Defaults are `accuracy, balanced_accuracy, f1, precision, recall, roc_auc, log_loss,
matthews_corrcoef` for classification and `rmse, mae, r2, mape, explained_variance` for
regression. Pass a list to keep a subset, in your own order:

```python
EazyClassifier(metrics=["accuracy", "f1"], sort_by="f1")
```

Or a mapping of name to any scikit-learn scorer:

```python
from sklearn.metrics import get_scorer, make_scorer, fbeta_score

EazyClassifier(
    metrics={
        "accuracy": "accuracy",
        "f2": make_scorer(fbeta_score, beta=2),
        "auc": get_scorer("roc_auc"),
    }
)
```

## Cross-validation and splits

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

EazyClassifier(cv=10)  # more folds
EazyClassifier(cv=StratifiedKFold(3))  # any splitter
EazyClassifier(cv=None, test_size=0.25)  # a single holdout split, faster
```

With an explicit test set, every model is scored on exactly that set:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

holdout = EazyClassifier(
    estimators=["LogisticRegression", "RandomForestClassifier"],
    cv=None,
    random_state=0,
).fit(X_train, y_train, X_test=X_test, y_test=y_test)
```

## Ensembles

```python
top = EazyClassifier(
    estimators=[
        "LogisticRegression",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "GaussianNB",
    ],
    random_state=0,
).fit(X_train, y_train)

ensemble = top.fit_voting_ensemble(n_models=3)
print(ensemble.score(X_test, y_test))
```

The leading `n_models` from the leaderboard are combined, so the ensemble honours
`sort_by`. Soft voting is used by default, falling back to hard voting with a warning
if a selected model has no `predict_proba`. `EazyRegressor.fit_voting_ensemble` is the
same, minus the voting argument.

## Saving a run

```python
saved = EazyClassifier(
    estimators=["LogisticRegression", "DecisionTreeClassifier"],
    save_dir="runs/experiment-1",
    random_state=0,
).fit(X, y)
```

```
runs/experiment-1/
├── LogisticRegression.joblib        # the whole pipeline, preprocessing included
├── DecisionTreeClassifier.joblib
├── results.csv
└── metadata.json
```

`metadata.json` records the seed, CV setting, data shape, best model and the versions of
every library involved, so a leaderboard can be reproduced months later.

## When a model fails

It gets a row with `status="failed"` and an entry in `errors_`; the rest of the run
continues.

```python
clf.errors_  # {'SomeModel': 'ValueError: ...'}
EazyClassifier(on_error="raise")  # stop at the first failure instead
```

## Upgrading to 0.3.0

0.3.0 is a correctness release, and it changes the API:

```
before:  clf.fit(X_train, X_test, y_train, y_test)
now:     clf.fit(X, y)
```

The old signature took `(X_train, X_test, y_train, y_test)` while every published
example called it `(X_train, y_train, X_test, y_test)`, so a great many leaderboards
were quietly computed on the training set. The new `fit(X, y)` follows the scikit-learn
convention and returns `self`.

The old entry points still work and emit a `DeprecationWarning`; they will be removed in
0.5.0. They keep their original *implemented* argument order — reinterpreting those
arguments would silently change what existing code trains on.

```python
from eazypredict import EazyClassifier, EazyRegressor  # the import path to use
```

See [CHANGELOG.md](CHANGELOG.md) for the full list of fixes.

## Roadmap

Hyperparameter tuning, parallel fitting, permutation feature importance, stacking
ensembles, probability calibration, and leaderboard export to Markdown and LaTeX.

## License

MIT
