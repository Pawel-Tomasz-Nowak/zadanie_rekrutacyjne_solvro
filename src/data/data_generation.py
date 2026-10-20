"""Generate plot images and their ground-truth point data."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
if __package__:
    from .utils.constants import (
        NOISE_STD,
        PLOT_DPI,
        RANDOM_SEED,
        IMAGE_HEIGHT_MAX,
        IMAGE_HEIGHT_MIN,
        IMAGE_WIDTH_MAX,
        IMAGE_WIDTH_MIN,
        MARKER_SIZE_MAX,
        MARKER_SIZE_MIN,
        MARKER_COLORS,
        X_MAX,
        X_MIN,
        X_RANGE,
        Y_MAX,
        Y_MIN,
    )
    from .utils.helpers import create_plot_functions, generate_x_values
else:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.data.utils.constants import (  
        NOISE_STD,
        PLOT_DPI,
        RANDOM_SEED,
        IMAGE_HEIGHT_MAX,
        IMAGE_HEIGHT_MIN,
        IMAGE_WIDTH_MAX,
        IMAGE_WIDTH_MIN,
        MARKER_SIZE_MAX,
        MARKER_SIZE_MIN,
        MARKER_COLORS,
        X_MAX,
        X_MIN,
        X_RANGE,
        Y_MAX,
        Y_MIN,
    )
    from src.data.utils.helpers import create_plot_functions, generate_x_values  # type: ignore[no-redef]


X_SAMPLING_STRATEGIES = ("uniform", "power", "random", "clustered", "mixed")


def generate_plots(
    output_dir: Path | str = Path(__file__).parent / "generated_data",
) -> None:
    """Generate plot images and JSON ground truth for every configured function."""
    output_root = Path(output_dir)
    image_dir = output_root / "images"
    json_dir = output_root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(RANDOM_SEED)
    functions = create_plot_functions(random_generator)
    image_index = 1

    for func_type, functions in functions.items():
        for function in functions:
            repetitions = random_generator.integers(1, 6)

            for _ in range(repetitions):
                image_width = int(
                    random_generator.integers(IMAGE_WIDTH_MIN, IMAGE_WIDTH_MAX + 1)
                )
                image_height = int(
                    random_generator.integers(IMAGE_HEIGHT_MIN, IMAGE_HEIGHT_MAX + 1)
                )
                use_grid = bool(random_generator.binomial(1, p=0.5))
                marker_size = int(
                    random_generator.integers(MARKER_SIZE_MIN, MARKER_SIZE_MAX + 1)
                )
                marker_color = str(random_generator.choice(MARKER_COLORS))
                n_points = int(random_generator.integers(30, 100))
                x_sampling_strategy = random_generator.choice(X_SAMPLING_STRATEGIES)
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
                
                axes.scatter(
                    x_values,
                    y_values,
                    color=marker_color,
                    s=marker_size,
                    zorder=2,
                )
                axes.set_xlim(X_MIN, X_MAX)
                axes.set_ylim(Y_MIN, Y_MAX)

                if use_grid:
                    axes.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

                image_path = image_dir / f"plot_{image_index:03d}.png"
                json_path = json_dir / f"plot_{image_index:03d}_gt.json"
                figure.savefig(image_path, dpi=PLOT_DPI, bbox_inches=None)

                ground_truth = {
                    "function_type": func_type,
                    "width": image_width,
                    "height": image_height,
                    "grid_lines": bool(use_grid),
                    "x_sampling_strategy": x_sampling_strategy,
                    "marker_size": marker_size,
                    "marker_color": marker_color,
                    "plot_style": "line_scatter",
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
    
