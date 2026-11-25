"""Stands in for the 0.2.x ``eazypredict.EazyClassifier`` module.

Registered in ``sys.modules`` by :mod:`eazypredict` so that the legacy
``from eazypredict.EazyClassifier import EazyClassifier`` keeps working.
"""

from __future__ import annotations

from ._legacy import LegacyEazyClassifier as EazyClassifier

__all__ = ["EazyClassifier"]
