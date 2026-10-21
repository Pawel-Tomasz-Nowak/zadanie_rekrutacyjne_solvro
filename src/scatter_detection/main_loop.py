"""Punkt wejścia do pipeline'u detekcji punktów w projekcie scatter_detection."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatter_detection.main.utils.configuration import create_configuration
from scatter_detection.main.utils.detection import detect_points, isolate_plot_area, pixel_to_value


def run_pipeline(project_root: str | Path | None = None) -> None:
    """Przetwarza wszystkie obrazy w katalogu generated_data i zapisuje wykryte punkty."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent.parent
    data_dir = root / "src" / "scatter_detection" / "data" / "generated_data"
    images_path = data_dir / "images"
    metadata_dir = data_dir / "metadata"
    config_dir = data_dir / "configs"

    config_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_path.glob("*.png")):
        image_id = image_path.stem
        metadata_path = metadata_dir / f"{image_id}_gt.json"
        config_path = config_dir / f"{image_id}_config.json"
        detected_points_path = data_dir / "detected_points" / f"{image_id}_detected_points.json"

        print(f"Przetwarzanie obrazu: {image_id}")
        image, config = create_configuration(image_id, image_path, metadata_path, config_path)
        plot_image = isolate_plot_area(image, config)
        detected_points = detect_points(plot_image, config)
        converted_points = pixel_to_value(detected_points, config["calibration"])

        detected_points_path.parent.mkdir(parents=True, exist_ok=True)
        with detected_points_path.open("w", encoding="utf-8") as file:
            json.dump(converted_points, file, indent=2)


if __name__ == "__main__":
    run_pipeline()
