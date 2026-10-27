"""Generator syntetycznych wykresów rozrzutowych i ich metadanych GT.

Moduł odpowiada za produkcję obrazów PNG oraz plików JSON z prawdziwymi
współrzędnymi punktów. Wspólny zestaw stałych jest importowany z
``src.scatter_detection.constants``.
"""

import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scatter_detection.constants import (
    IMAGE_HEIGHT_MAX,
    IMAGE_HEIGHT_MIN,
    IMAGE_WIDTH_MAX,
    IMAGE_WIDTH_MIN,
    MARKER_SIZE_MAX,
    MARKER_SIZE_MIN,
    N,
    NOISE_STD,
    PLOT_DPI,
    RANDOM_SEED,
    X_MAX,
    X_MIN,
    X_RANGE,
    Y_MAX,
    Y_MIN,
    X_SAMPLING_STRATEGIES,
    PLOT_STYLES
)
from scatter_detection.data.utils.helpers import create_plot_functions, generate_x_values


def _display_progress(completed: int, total: int) -> None:
    """Wyświetla postęp generowania obrazów w jednej linii terminala."""
    bar_width = 40
    filled_width = int(bar_width * completed / total)
    progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)
    print(f"\rGenerowanie obrazów: [{progress_bar}] {completed}/{total}", end="", flush=True)


def generate_plots(output_dir: Path | str = Path(__file__).parent / "generated_data") -> None:
    """Generuje obrazy i metadane GT dla wszystkich funkcji benchmarkowych."""
    output_root = Path(output_dir)
    image_dir = output_root / "images"
    json_dir = output_root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(RANDOM_SEED)
    benchmark_functions = create_plot_functions(random_generator)
    total_images = (
        sum(len(functions) for functions in benchmark_functions.values())
        * len(PLOT_STYLES)
        * 2
        * len(X_SAMPLING_STRATEGIES)
        * N
    )
    generated_images = 0
    image_index = 1

    for function_family_name, functions in benchmark_functions.items():
        for function in functions:
            for plot_style, use_grid, x_sampling_strategy in product(
                PLOT_STYLES,
                (False, True),
                X_SAMPLING_STRATEGIES,
            ):
                for _ in range(N):
                    image_width = int(random_generator.integers(IMAGE_WIDTH_MIN, IMAGE_WIDTH_MAX + 1))
                    image_height = int(random_generator.integers(IMAGE_HEIGHT_MIN, IMAGE_HEIGHT_MAX + 1))
                    marker_size = int(random_generator.integers(MARKER_SIZE_MIN, MARKER_SIZE_MAX + 1))
                    n_points = int(random_generator.integers(30, 100))

                    x_values = generate_x_values(
                        X_RANGE,
                        n_points,
                        random_generator=random_generator,
                        strategy=x_sampling_strategy,
                        power=float(random_generator.uniform(1.5, 3.0)),
                    )
                    noise = random_generator.normal(scale=NOISE_STD, size=n_points)
                    y_values = function(x_values) + noise

                    visible_mask = (
                        (x_values >= X_MIN)
                        & (x_values <= X_MAX)
                        & (y_values >= Y_MIN)
                        & (y_values <= Y_MAX)
                    )

                    figure, axes = plt.subplots(
                        figsize=(image_width / PLOT_DPI, image_height / PLOT_DPI),
                        dpi=PLOT_DPI,
                    )

                    if plot_style == "line_scatter_plot":
                        axes.plot(
                            x_values,
                            y_values,
                            color="black",
                            linewidth=1.0,
                            zorder=1,
                        )
                    axes.scatter(x_values, y_values, color="black", s=marker_size, zorder=2)
                    axes.set_xlim(X_MIN, X_MAX)
                    axes.set_ylim(Y_MIN, Y_MAX)

                    if use_grid:
                        axes.grid(True, which="both", linestyle="--", linewidth=1)

                    image_path = image_dir / f"plot_{image_index:03d}.png"
                    json_path = json_dir / f"plot_{image_index:03d}_gt.json"
                    figure.savefig(image_path, dpi=PLOT_DPI, bbox_inches=None)

                    ground_truth = {
                        "function_type": function_family_name,
                        "width": image_width,
                        "height": image_height,
                        "grid_lines": bool(use_grid),
                        "x_sampling_strategy": x_sampling_strategy,
                        "marker_size": marker_size,
                        "plot_style": plot_style,
                        "n_generated": n_points,
                        "n_visible": int(visible_mask.sum()),
                        "x_range": [X_MIN, X_MAX],
                        "y_range": [Y_MIN, Y_MAX],
                        "points": [
                            {"x": float(x_value), "y": float(y_value)}
                            for x_value, y_value in zip(x_values[visible_mask], y_values[visible_mask])
                        ],
                    }

                    with json_path.open("w", encoding="utf-8") as file:
                        json.dump(ground_truth, file, indent=2)

                    plt.close(figure)
                    image_index += 1
                    generated_images += 1
                    _display_progress(generated_images, total_images)

    print()


if __name__ == "__main__":
    generate_plots()
