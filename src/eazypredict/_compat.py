"""Optional third-party integrations.

xgboost and lightgbm are heavy compiled dependencies whose wheels lag new CPython
releases by months. They are an optional ``[boosters]`` extra, imported lazily so that
``import eazypredict`` works with the core dependencies alone.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__all__ = ["BOOSTER_ESTIMATORS", "has_booster", "load_optional_estimator"]

# Public estimator name -> (distribution name, module, attribute, extra).
BOOSTER_ESTIMATORS: dict[str, tuple[str, str, str]] = {
    "XGBClassifier": ("xgboost", "xgboost", "XGBClassifier"),
    "XGBRegressor": ("xgboost", "xgboost", "XGBRegressor"),
    "LGBMClassifier": ("lightgbm", "lightgbm", "LGBMClassifier"),
    "LGBMRegressor": ("lightgbm", "lightgbm", "LGBMRegressor"),
}


def has_booster(name: str) -> bool:
    """Whether the optional estimator ``name`` can be imported.

    Uses ``find_spec`` so that merely asking the question does not pay the (substantial)
    cost of importing xgboost or lightgbm.
    """
    entry = BOOSTER_ESTIMATORS.get(name)
    if entry is None:
        return False
    module = entry[1]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def missing_booster_error(name: str) -> ImportError:
    """The error raised when an unavailable optional estimator is explicitly named."""
    distribution = BOOSTER_ESTIMATORS[name][0]
    return ImportError(
        f"{name!r} requires the {distribution!r} package, which is not installed. "
        f"Install it with: pip install 'eazypredict[boosters]'"
    )


def load_optional_estimator(name: str) -> Any:
    """Import and return an optional estimator class.

    Raises:
        ImportError: if the backing distribution is not installed. Explicitly asking
            for an estimator that cannot be provided is an error; silently dropping it
            would leave the user wondering why their leaderboard is short.
    """
    try:
        _, module, attribute = BOOSTER_ESTIMATORS[name]
    except KeyError:
        raise KeyError(f"{name!r} is not an optional estimator") from None

    try:
        return getattr(importlib.import_module(module), attribute)
    except ImportError as exc:
        raise missing_booster_error(name) from exc
