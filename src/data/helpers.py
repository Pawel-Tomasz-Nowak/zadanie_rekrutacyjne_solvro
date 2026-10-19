"""Helper functions used by the synthetic data generator."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def generate_x_values(
    x_range: tuple[float, float],
    n_points: int,
    power: float = 2.0,
) -> NDArray[np.float64]:
    """Generate non-linearly spaced x values within ``x_range``."""
    x_min, x_max = x_range

    if n_points < 2:
        raise ValueError("n_points must be greater than or equal to 2.")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if power <= 0:
        raise ValueError("power must be greater than 0.")

    normalized_values = np.linspace(0.0, 1.0, n_points)
    return x_min + (x_max - x_min) * normalized_values**power


def create_polynomial(coefficients: Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a polynomial function from coefficients in ascending order."""
    polynomial_coefficients = np.asarray(coefficients).ravel()

    def polynomial(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.polynomial.polynomial.polyval(x_values, polynomial_coefficients)

    return polynomial


def create_plot_functions(
    random_generator: np.random.Generator,
) -> list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """Create the polynomial, exponential, and trigonometric functions to plot."""
    coefficients = [
        random_generator.uniform(0.0, 1.0, size=i + 1)
        for i in range(1, 5)
    ]
    polynomials = [create_polynomial(values) for values in coefficients]
    exponentials = [lambda x_values: 2**x_values]
    trigonometric = [
        lambda x_values: np.tan(x_values),
        lambda x_values: np.sin(x_values),
        lambda x_values: np.cos(x_values),
    ]

    return polynomials + exponentials + trigonometric
