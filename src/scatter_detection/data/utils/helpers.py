"""Funkcje pomocnicze do generowania zsyntetyzowanych danych benchmarkowych.

Wartości i nazwy błędów są po polsku, żeby kod był czytelniejszy dla osoby
pracującej nad pipeline'em CV. Dodatkowo, funkcje były lekko uporządkowane tak,
żeby łatwiej utrzymać je w przyszłości.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

SUPPORTED_X_SAMPLING_STRATEGIES = ("uniform", "power", "random", "clustered", "mixed")


def generate_x_values(
    x_range: tuple[float, float],
    n_points: int,
    random_generator: np.random.Generator | None = None,
    strategy: str = "power",
    power: float = 2.0,
) -> NDArray[np.float64]:
    """Generuje posortowane współrzędne x zgodnie z wybraną strategią próbkowania."""
    x_min, x_max = x_range

    if n_points < 2:
        raise ValueError("Liczba punktów musi wynosić co najmniej 2.")
    if x_min >= x_max:
        raise ValueError("Prawa granica zakresu musi być większa od lewej.")
    if power <= 0:
        raise ValueError("Współczynnik potęgi musi być dodatni.")
    if strategy not in SUPPORTED_X_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Nieobsługiwana strategia próbkowania osi X: {strategy}. "
            f"Dostępne opcje: {SUPPORTED_X_SAMPLING_STRATEGIES}."
        )

    if strategy == "uniform":
        normalized_values = np.linspace(0.0, 1.0, n_points)
    elif strategy == "power":
        normalized_values = np.linspace(0.0, 1.0, n_points) ** power
    elif strategy == "random":
        if random_generator is None:
            raise ValueError("Do próbkowania losowego wymagany jest random_generator.")
        normalized_values = np.sort(random_generator.uniform(0.0, 1.0, n_points))
    elif strategy == "clustered":
        if random_generator is None:
            raise ValueError("Do próbkowania skupionego wymagany jest random_generator.")
        center = random_generator.uniform(0.2, 0.8)
        spread = random_generator.uniform(0.08, 0.2)
        normalized_values = np.clip(random_generator.normal(center, spread, n_points), 0.0, 1.0)
        normalized_values.sort()
    elif strategy == "mixed":
        if random_generator is None:
            raise ValueError("Do próbkowania mieszanego wymagany jest random_generator.")
        clustered_count = max(2, n_points // 2)
        uniform_count = n_points - clustered_count
        clustered_values = random_generator.beta(2.0, 8.0, clustered_count)
        uniform_values = random_generator.uniform(0.0, 1.0, uniform_count)
        normalized_values = np.sort(np.concatenate((clustered_values, uniform_values)))
    else:
        raise ValueError(f"Nieprzewidziana strategia: {strategy}")

    return x_min + (x_max - x_min) * normalized_values


def create_polynomial(coefficients: Sequence[float]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Tworzy funkcję wielomianową z współczynnikami od najmniejszych potęg do najwyższych."""
    polynomial_coefficients = np.asarray(coefficients).ravel()

    def polynomial_function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.polynomial.polynomial.polyval(x_values, polynomial_coefficients)

    return polynomial_function


def _make_exponential() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2.0 ** (0.2 * x_values)

    return function


def _make_trigonometric_functions() -> list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    return [
        lambda x_values: np.tan(x_values),
        lambda x_values: np.sin(x_values),
        lambda x_values: np.cos(x_values),
    ]


def _make_damped_oscillation() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 4.0 * np.exp(-0.18 * np.abs(x_values)) * np.sin(2.2 * x_values)

    return function


def _make_sigmoid() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 8.0 / (1.0 + np.exp(-1.1 * (x_values - 1.5))) + 0.5

    return function


def _make_gaussian_mixture() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            2.0
            + 4.0 * np.exp(-((x_values + 2.0) ** 2) / 2.0)
            - 3.0 * np.exp(-((x_values - 2.5) ** 2) / 0.9)
            + 2.0 * np.exp(-((x_values - 5.5) ** 2) / 1.5)
        )

    return function


def _make_piecewise_function() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.select(
            [x_values < -1.0, x_values < 4.0],
            [2.0 + np.sin(x_values), 3.0 + 0.6 * x_values],
            default=6.0 - np.exp(-0.5 * (x_values - 4.0)),
        )

    return function


def _make_composite_function() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def function(x_values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            3.0
            + 0.12 * x_values ** 2
            + 1.5 * np.sin(1.4 * x_values)
            + 2.0 * np.exp(-((x_values - 2.0) ** 2) / 1.2)
        )

    return function


def create_plot_functions(
    random_generator: np.random.Generator,
) -> dict[str, list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
    """Tworzy zestaw benchmarkowych rodzin funkcji dla generatora wykresów."""
    polynomial_coefficients = [
        random_generator.uniform(0.0, 1.0, size=degree)
        for degree in range(1, 5)
    ]
    polynomials = [create_polynomial(values) for values in polynomial_coefficients]

    functions: dict[str, list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = {
        "polynomials": polynomials,
        "exponentials": [_make_exponential()],
        "trigonometric": _make_trigonometric_functions(),
        "damped_oscillations": [_make_damped_oscillation()],
        "sigmoids": [_make_sigmoid()],
        "gaussian_mixtures": [_make_gaussian_mixture()],
        "piecewise": [_make_piecewise_function()],
        "composite": [_make_composite_function()],
    }

    return functions
