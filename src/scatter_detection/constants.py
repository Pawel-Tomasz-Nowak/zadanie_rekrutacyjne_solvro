"""Wspólne stałe wykorzystywane przez generator danych i pipeline detekcji."""

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

PLOT_DPI: Final[int] = 125
IMAGE_WIDTH_MIN: Final[int] = 500
IMAGE_WIDTH_MAX: Final[int] = 1000
IMAGE_HEIGHT_MIN: Final[int] = 500
IMAGE_HEIGHT_MAX: Final[int] = 1000
MARKER_SIZE_MIN: Final[int] = 6
MARKER_SIZE_MAX: Final[int] = 30
MARKER_COLORS: Final[tuple[str, ...]] = (
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "magenta",
    "cyan",
)

NOISE_STD: Final[float] = 1.0
RANDOM_SEED: Final[int] = 42
