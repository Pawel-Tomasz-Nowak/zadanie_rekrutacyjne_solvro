"""Funkcje konfiguracyjne odpowiedzialne za geometrię obrazu i walidację wejścia."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from scatter_detection.constants import X_MAX, X_MIN, Y_MAX, Y_MIN
from math import floor, ceil

def compute_image_boundaries(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Oblicza granice obszaru wykresu w pikselach zgodnie z ustawieniami Matplotlib."""
    subplot_left = plt.rcParams["figure.subplot.left"]
    subplot_right = plt.rcParams["figure.subplot.right"]
    subplot_bottom = plt.rcParams["figure.subplot.bottom"]
    subplot_top = plt.rcParams["figure.subplot.top"]

    px1 = int(floor(subplot_left * image_width))
    px2 = int(ceil(subplot_right * image_width))
    py1 = int(ceil((1.0 - subplot_bottom) * image_height))
    py2 = int(floor((1.0 - subplot_top) * image_height))

    return px1, px2, py1, py2


def validate_configuration(config: dict, image) -> tuple[int, int]:
    """Sprawdza, czy konfiguracja opisuje poprawny obszar wykresu dla danego obrazu."""
    if image is None:
        raise FileNotFoundError(
            f"Nie można odczytać obrazu dla image_id: {config.get('image_id', 'nieznany')}"
        )

    image_height, image_width = image.shape[:2]
    xmin, ymin, xmax, ymax = config["plot_bbox"]

    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
        raise ValueError(
            "plot_bbox ma nieprawidłowe granice albo wychodzi poza rozmiar obrazu."
        )

    x_axis = config["calibration"]["x_axis"]
    y_axis = config["calibration"]["y_axis"]

    if x_axis["px1"] == x_axis["px2"] or x_axis["val1"] == x_axis["val2"]:
        raise ValueError("Kalibracja osi X musi zawierać dwa różne piksele i dwie różne wartości.")
    if y_axis["py1"] == y_axis["py2"] or y_axis["val1"] == y_axis["val2"]:
        raise ValueError("Kalibracja osi Y musi zawierać dwa różne piksele i dwie różne wartości.")

    return image_width, image_height


def create_configuration(image_id: str, image_path: str | Path, metadata_path: str | Path, config_path: str | Path):
    """Tworzy konfigurację dla jednego obrazu na podstawie metadanych i zapisuje ją do JSON."""
    metadata_path = Path(metadata_path)
    image_path = Path(image_path)
    config_path = Path(config_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_width = int(metadata["width"])
    image_height = int(metadata["height"])

    px1, px2, py1, py2 = compute_image_boundaries(image_width, image_height)

    config = {
        "image_id": image_id,
        "plot_bbox": [px1, py2, px2, py1],
        "calibration": {
            "x_axis": {"px1": px1, "val1": X_MIN, "px2": px2, "val2": X_MAX},
            "y_axis": {"py1": py1, "val1": Y_MIN, "py2": py2, "val2": Y_MAX},
        },
    }

    image = cv2.imread(str(image_path))
    validate_configuration(config, image)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    return image, config
