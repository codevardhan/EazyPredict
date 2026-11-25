"""EazyPredict: run and compare many scikit-learn models in a few lines.

>>> from sklearn.datasets import make_classification
>>> from eazypredict import EazyClassifier
>>> X, y = make_classification(n_samples=60, n_features=5, random_state=0)
>>> clf = EazyClassifier(estimators=["LogisticRegression"], random_state=0)
>>> clf.fit(X, y).best_model_name_
'LogisticRegression'
"""

from __future__ import annotations

import sys as _sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import _legacy_classifier, _legacy_regressor
from .classification import EazyClassifier
from .estimators import DEFAULT_CLASSIFIERS, DEFAULT_REGRESSORS, available_estimators
from .regression import EazyRegressor

__author__ = """Harshavardhan M"""
__email__ = "harsha7vardhan@gmail.com"

try:
    __version__ = _version("eazypredict")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw tree
    __version__ = "0.0.0.dev0"

# The 0.2.x layout put each class in a module of the same name, so the class
# `eazypredict.EazyClassifier` and the module `eazypredict.EazyClassifier` collide.
# Importing the submodule normally would make Python rebind the attribute on this
# package to the *module*, shadowing the class for everyone who imported it first.
# Pre-seeding sys.modules means the legacy import path resolves to the shim without
# the import system ever touching this namespace.
_sys.modules[__name__ + ".EazyClassifier"] = _legacy_classifier
_sys.modules[__name__ + ".EazyRegressor"] = _legacy_regressor

__all__ = [
    "DEFAULT_CLASSIFIERS",
    "DEFAULT_REGRESSORS",
    "EazyClassifier",
    "EazyRegressor",
    "__version__",
    "available_estimators",
]
