"""Punkt wejścia do pipeline'u detekcji punktów w projekcie scatter_detection."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatter_detection.utils.configuration import create_configuration
from scatter_detection.utils.detection import (
    detect_points,
    detect_points_line_aware,
    isolate_plot_area,
    pixel_to_value,
)


def _display_progress(completed: int, total: int) -> None:
    """Wyświetla postęp przetwarzania obrazów w jednej linii terminala."""
    bar_width = 40
    filled_width = int(bar_width * completed / total) if total else 0
    progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)
    print(f"\rPrzetwarzanie obrazów: [{progress_bar}] {completed}/{total}", end="", flush=True)


def run_pipeline(project_root: str | Path | None = None) -> None:
    """Przetwarza wszystkie obrazy w katalogu generated_data i zapisuje wykryte punkty."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent.parent
    data_dir = root / "src" / "scatter_detection" / "data" / "generated_data"
    images_path = data_dir / "images"
    metadata_dir = data_dir / "metadata"
    config_dir = data_dir / "configs"

    config_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_path.glob("*.png"))
    total_images = len(image_paths)
    processed_images = 0

    for image_path in image_paths:
        image_id = image_path.stem
        metadata_path = metadata_dir / f"{image_id}_gt.json"
        config_path = config_dir / f"{image_id}_config.json"
        detected_points_path = data_dir / "detected_points" / f"{image_id}_detected_points.json"

        image, config = create_configuration(image_id, image_path, metadata_path, config_path)
        plot_image = isolate_plot_area(image, config)
        baseline_points = pixel_to_value(
            detect_points(plot_image, config),
            config["calibration"],
        )
        line_aware_points = pixel_to_value(
            detect_points_line_aware(plot_image, config),
            config["calibration"],
        )

        detected_points_path.parent.mkdir(parents=True, exist_ok=True)
        with detected_points_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "baseline": baseline_points,
                    "line_aware": line_aware_points,
                },
                file,
                indent=2,
            )

        processed_images += 1
        _display_progress(processed_images, total_images)

    print()


if __name__ == "__main__":
    run_pipeline()
