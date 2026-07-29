"""Score-network parameterizations.

Currently ships :class:`ScoreMLP1D` only (1D, spectral-normalized option).
"""
from .score_mlp_1d import ScoreMLP1D

__all__ = ["ScoreMLP1D"]
