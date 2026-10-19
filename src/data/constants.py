"""Configuration constants for synthetic plot generation."""

from typing import Final

X_RANGE: Final[tuple[float, float]] = (-5.0, 10.0)
X_MIN: Final[float]
X_MAX: Final[float]
X_MIN, X_MAX = X_RANGE

Y_RANGE: Final[tuple[float, float]] = (0.0, 10.0)
Y_MIN: Final[float]
Y_MAX: Final[float]
Y_MIN, Y_MAX = Y_RANGE

ORIGIN: Final[tuple[float, float]] = (0.0, 0.0)

PLOT_WIDTH: Final[int] = 5
PLOT_HEIGHT: Final[int] = 5
PLOT_DPI: Final[int] = 100

NOISE_STD: Final[float] = 1.0
RANDOM_SEED: Final[int] = 42
