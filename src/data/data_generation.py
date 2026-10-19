"""Generate plot images and their ground-truth point data."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .constants import (
        NOISE_STD,
        PLOT_DPI,
        PLOT_HEIGHT,
        PLOT_WIDTH,
        RANDOM_SEED,
        X_MAX,
        X_MIN,
        X_RANGE,
        Y_MAX,
        Y_MIN,
    )
    from .helpers import create_plot_functions, generate_x_values
else:
    sys.path.insert(0, str(Path(__file__).parent))
    from constants import (  
        NOISE_STD,
        PLOT_DPI,
        PLOT_HEIGHT,
        PLOT_WIDTH,
        RANDOM_SEED,
        X_MAX,
        X_MIN,
        X_RANGE,
        Y_MAX,
        Y_MIN,
    )
    from helpers import create_plot_functions, generate_x_values  # type: ignore[no-redef]


def generate_plots(
    output_dir: Path | str = Path(__file__).parent / "generated_data",
) -> None:
    """Generate plot images and JSON ground truth for every configured function."""
    output_root = Path(output_dir)
    image_dir = output_root / "images"
    json_dir = output_root / "jsons"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(RANDOM_SEED)
    functions = create_plot_functions(random_generator)
    image_index = 1

    for function in functions:
        repetitions = random_generator.integers(1, 6)

        for _ in range(repetitions):
            n_points = int(random_generator.integers(10, 50))
            x_values = generate_x_values(X_RANGE, n_points)
            noise = random_generator.normal(scale=NOISE_STD, size=n_points)
            y_values = function(x_values) + noise

            visible_mask = (
                (x_values >= X_MIN)
                & (x_values <= X_MAX)
                & (y_values >= Y_MIN)
                & (y_values <= Y_MAX)
            )

            figure, axes = plt.subplots(
                figsize=(PLOT_WIDTH, PLOT_HEIGHT),
                dpi=PLOT_DPI,
            )
            axes.scatter(x_values, y_values, color="blue", s=10)
            axes.set_xlim(X_MIN, X_MAX)
            axes.set_ylim(Y_MIN, Y_MAX)

            image_path = image_dir / f"plot_{image_index:03d}.png"
            json_path = json_dir / f"plot_{image_index:03d}_gt.json"
            figure.savefig(image_path, dpi=PLOT_DPI, bbox_inches=None)

            ground_truth = {
                "function": getattr(function, "__name__", str(function)),
                "n_generated": n_points,
                "n_visible": int(visible_mask.sum()),
                "x_range": [X_MIN, X_MAX],
                "y_range": [Y_MIN, Y_MAX],
                "points": [
                    {"x": float(x_value), "y": float(y_value)}
                    for x_value, y_value in zip(
                        x_values[visible_mask],
                        y_values[visible_mask],
                    )
                ],
            }
            with json_path.open("w", encoding="utf-8") as file:
                json.dump(ground_truth, file, indent=2)

            plt.close(figure)
            image_index += 1


if __name__ == "__main__":
    generate_plots()
    
