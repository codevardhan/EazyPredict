"""Stands in for the 0.2.x ``eazypredict.EazyRegressor`` module.

Registered in ``sys.modules`` by :mod:`eazypredict` so that the legacy
``from eazypredict.EazyRegressor import EazyRegressor`` keeps working.
"""

from __future__ import annotations

from ._legacy import LegacyEazyRegressor as EazyRegressor

__all__ = ["EazyRegressor"]
