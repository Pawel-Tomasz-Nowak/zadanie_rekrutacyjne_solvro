"""Helper functions used by the synthetic data generator."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def generate_x_values(
    x_range: tuple[float, float],
    n_points: int,
    random_generator: np.random.Generator | None = None,
    strategy: str = "power",
    power: float = 2.0,
) -> NDArray[np.float64]:
    """Generate sorted x coordinates using a configurable sampling strategy."""
    x_min, x_max = x_range

    if n_points < 2:
        raise ValueError("n_points must be greater than or equal to 2.")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if power <= 0:
        raise ValueError("power must be greater than 0.")

    if strategy == "uniform":
        normalized_values = np.linspace(0.0, 1.0, n_points)
    elif strategy == "power":
        normalized_values = np.linspace(0.0, 1.0, n_points) ** power
    elif strategy == "random":
        if random_generator is None:
            raise ValueError("random_generator is required for random sampling.")
        normalized_values = np.sort(random_generator.uniform(0.0, 1.0, n_points))
    elif strategy == "clustered":
        if random_generator is None:
            raise ValueError("random_generator is required for clustered sampling.")
        center = random_generator.uniform(0.2, 0.8)
        spread = random_generator.uniform(0.08, 0.2)
        normalized_values = np.clip(
            random_generator.normal(center, spread, n_points),
            0.0,
            1.0,
        )
        normalized_values.sort()
    elif strategy == "mixed":
        if random_generator is None:
            raise ValueError("random_generator is required for mixed sampling.")
        n_clustered = max(2, n_points // 2)
        n_uniform = n_points - n_clustered
        clustered = random_generator.beta(2.0, 8.0, n_clustered)
        uniform = random_generator.uniform(0.0, 1.0, n_uniform)
        normalized_values = np.sort(np.concatenate((clustered, uniform)))
    else:
        raise ValueError(f"Unsupported x sampling strategy: {strategy}")

    return x_min + (x_max - x_min) * normalized_values


def create_polynomial(coefficients: Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a polynomial function from coefficients in ascending order."""
    polynomial_coefficients = np.asarray(coefficients).ravel()

    def polynomial(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.polynomial.polynomial.polyval(x_values, polynomial_coefficients)

    return polynomial


def create_plot_functions(
    random_generator: np.random.Generator,
) -> dict[str, list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
    """Create benchmark function families with deterministic random parameters."""
    coefficients = [
        random_generator.uniform(0.0, 1.0, size=i + 1)
        for i in range(1, 5)
    ]
    polynomials = [create_polynomial(values) for values in coefficients]
    exponentials = [lambda x_values: 2 ** (0.2 * x_values)]
    trigonometric = [
        lambda x_values: np.tan(x_values),
        lambda x_values: np.sin(x_values),
        lambda x_values: np.cos(x_values),
    ]
    damped_oscillations = [
        lambda x_values: 4.0 * np.exp(-0.18 * np.abs(x_values))
        * np.sin(2.2 * x_values),
    ]
    sigmoids = [
        lambda x_values: 8.0 / (1.0 + np.exp(-1.1 * (x_values - 1.5))) + 0.5,
    ]
    gaussian_mixtures = [
        lambda x_values: (
            2.0
            + 4.0 * np.exp(-((x_values + 2.0) ** 2) / 2.0)
            - 3.0 * np.exp(-((x_values - 2.5) ** 2) / 0.9)
            + 2.0 * np.exp(-((x_values - 5.5) ** 2) / 1.5)
        ),
    ]
    piecewise_functions = [
        lambda x_values: np.select(
            [
                x_values < -1.0,
                x_values < 4.0,
            ],
            [
                2.0 + np.sin(x_values),
                3.0 + 0.6 * x_values,
            ],
            default=6.0 - np.exp(-0.5 * (x_values - 4.0)),
        ),
    ]
    composite_functions = [
        lambda x_values: (
            3.0
            + 0.12 * x_values**2
            + 1.5 * np.sin(1.4 * x_values)
            + 2.0 * np.exp(-((x_values - 2.0) ** 2) / 1.2)
        ),
    ]

    functions = {
        "polynomials": polynomials,
        "exponentials": exponentials,
        "trigonometric": trigonometric,
        "damped_oscillations": damped_oscillations,
        "sigmoids": sigmoids,
        "gaussian_mixtures": gaussian_mixtures,
        "piecewise": piecewise_functions,
        "composite": composite_functions,
    }

    return functions
