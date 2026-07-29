"""Forward operators G in the inverse problem y = G(u) + sigma * eta."""
from .deconv_1d import GaussianBlur1D, Identity1D

__all__ = ["GaussianBlur1D", "Identity1D"]
