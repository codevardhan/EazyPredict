"""Estimator discovery and resolution."""

from __future__ import annotations

import difflib
import warnings
from functools import cache
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators, get_tags

from ._compat import (
    BOOSTER_ESTIMATORS,
    has_booster,
    load_optional_estimator,
    missing_booster_error,
)

__all__ = [
    "DEFAULT_CLASSIFIERS",
    "DEFAULT_REGRESSORS",
    "available_estimators",
    "resolve_estimators",
]

Kind = Literal["classifier", "regressor"]

#: Fast, well-behaved defaults spanning linear, tree, kernel, neighbour and boosting
#: families. ``HistGradientBoosting*`` is included so that the leaderboard still has a
#: competitive gradient-booster when the optional ``[boosters]`` extra is absent.
DEFAULT_CLASSIFIERS: tuple[str, ...] = (
    "LogisticRegression",
    "RidgeClassifier",
    "SGDClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "SVC",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "XGBClassifier",
    "LGBMClassifier",
)

#: ``GaussianProcessRegressor`` is deliberately excluded: it is O(n^3) in time and
#: O(n^2) in memory, and exhausts RAM on datasets of only a few tens of thousands of
#: rows. It remains available by name for anyone who wants it.
DEFAULT_REGRESSORS: tuple[str, ...] = (
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "KNeighborsRegressor",
    "NuSVR",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
    "MLPRegressor",
    "XGBRegressor",
    "LGBMRegressor",
)

# Construction-time parameters that silence estimators which otherwise write directly
# to stdout. Applied only where the estimator actually accepts the parameter.
#
# Deliberately narrow. Blanket-setting something like `n_jobs=1` looks harmless but
# scikit-learn deprecated that parameter on several estimators in 1.8, so forcing it
# emits a FutureWarning on everyone's behalf. Inner estimators already default to a
# single worker, so there is nothing to gain.
_PER_ESTIMATOR_PARAMS: dict[str, dict[str, Any]] = {
    # lightgbm treats verbose=0 as "warnings still on"; -1 is the silent level.
    "LGBMClassifier": {"verbose": -1},
    "LGBMRegressor": {"verbose": -1},
    "XGBClassifier": {"verbosity": 0},
    "XGBRegressor": {"verbosity": 0},
}


def _is_usable_by_name(cls: type) -> bool:
    """Whether ``cls`` can go on a leaderboard identified by name alone.

    Two groups are excluded:

    * **Meta-estimators** (``VotingClassifier``, ``MultiOutputRegressor``, ...) require
      a base estimator, so ``cls()`` raises ``TypeError``. ``SelfTrainingClassifier``
      is the subtle case: it constructs with ``estimator=None`` and only fails later,
      when its tags are read. They remain usable by passing a configured instance.
    * **Deprecated estimators** (``PassiveAggressiveRegressor`` in sklearn 1.8, ...)
      construct fine but emit a ``FutureWarning``. Including them in ``"all"`` would
      make eazypredict warn on the user's behalf about a class they never named.

    Both are detected by attempting the construction and tag read rather than by
    introspecting signatures or docstrings, so the check keeps working as
    scikit-learn evolves.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            get_tags(cls())
        except Exception:
            return False
    return not any(
        issubclass(w.category, (DeprecationWarning, FutureWarning)) for w in caught
    )


@cache
def _sklearn_estimators(kind: Kind) -> dict[str, type]:
    """Named scikit-learn estimators of the requested kind.

    The ``type_filter`` is what stops a transformer such as ``StandardScaler`` being
    accepted as a classifier and failing later with an opaque ``AttributeError``.

    Cached because the usability probe constructs every candidate class, and the
    catalogue cannot change within a process.
    """
    return {
        name: cls
        for name, cls in all_estimators(type_filter=kind)
        if _is_usable_by_name(cls)
    }


def available_estimators(kind: Kind) -> dict[str, type]:
    """All estimator classes of ``kind`` that can be used right now.

    Includes the optional boosters only when their backing package is installed.
    """
    found = dict(_sklearn_estimators(kind))  # copy: the cached dict must stay pristine
    suffix = "Classifier" if kind == "classifier" else "Regressor"
    for name in BOOSTER_ESTIMATORS:
        if name.endswith(suffix) and has_booster(name):
            found[name] = load_optional_estimator(name)
    return found


def _instantiate(name: str, cls: type) -> BaseEstimator:
    """Construct an estimator, applying only the quiet params it actually accepts."""
    overrides = _PER_ESTIMATOR_PARAMS.get(name)
    if not overrides:
        return cls()
    try:
        accepted = set(cls().get_params())
    except Exception:  # pragma: no cover - estimators with required ctor args
        return cls()

    params = {k: v for k, v in overrides.items() if k in accepted}
    try:
        return cls(**params)
    except TypeError:  # pragma: no cover - defensive
        return cls()


def resolve_estimators(spec: str | list, kind: Kind) -> list[tuple[str, BaseEstimator]]:
    """Turn an ``estimators`` argument into concrete ``(name, estimator)`` pairs.

    This function is deliberately pure: it reads ``spec`` and returns a new list,
    never mutating its input or any caller state. The 0.2.x implementation rewrote
    ``self.classifiers`` in place, so a second call to ``fit()`` on the same instance
    silently produced an empty leaderboard.

    Args:
        spec: ``"default"``, ``"all"``, a list of estimator names, or a list of
            ``(name, estimator_instance)`` pairs. Bare estimator instances are also
            accepted and named after their class.
        kind: ``"classifier"`` or ``"regressor"``.

    Returns:
        A list of ``(name, unfitted estimator instance)`` pairs.

    Raises:
        ValueError: if ``spec`` is empty or names an unknown estimator.
        ImportError: if a named optional estimator's package is not installed.
    """
    catalogue = available_estimators(kind)

    if isinstance(spec, str):
        if spec == "all":
            return [(name, _instantiate(name, cls)) for name, cls in catalogue.items()]
        if spec != "default":
            raise ValueError(
                f"Unknown estimator specification {spec!r}. "
                "Use 'default', 'all', or a list of estimator names."
            )
        defaults = DEFAULT_CLASSIFIERS if kind == "classifier" else DEFAULT_REGRESSORS
        requested: list = list(defaults)
        implicit = True
    else:
        requested = list(spec)
        implicit = False

    if not requested:
        raise ValueError("No estimators requested.")

    resolved: list[tuple[str, BaseEstimator]] = []
    unavailable: list[str] = []

    for item in requested:
        if isinstance(item, tuple):
            name, estimator = item
            resolved.append((str(name), estimator))
            continue
        if not isinstance(item, str):
            resolved.append((type(item).__name__, item))
            continue

        if item in BOOSTER_ESTIMATORS and not has_booster(item):
            # Part of the default list: skip quietly and report once. Explicitly
            # requested: fail loudly, because silently returning a shorter
            # leaderboard than asked for is worse than an error.
            if implicit:
                unavailable.append(item)
                continue
            raise missing_booster_error(item)

        cls = catalogue.get(item)
        if cls is None:
            close = difflib.get_close_matches(item, catalogue, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"Unknown {kind} {item!r}.{hint}")
        resolved.append((item, _instantiate(item, cls)))

    if unavailable:
        warnings.warn(
            f"Skipping {', '.join(sorted(unavailable))}: the required package is not "
            "installed. Install it with: pip install 'eazypredict[boosters]'",
            UserWarning,
            stacklevel=2,
        )

    if not resolved:
        raise ValueError("No usable estimators remain after resolution.")
    return resolved
