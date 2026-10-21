"""Warstwa kompatybilności dla starych importów w module main."""

from scatter_detection.main.utils.configuration import (
    compute_image_boundaries,
    create_configuration,
    validate_configuration,
)
from scatter_detection.main.utils.detection import (
    create_image_mask,
    detect_points,
    isolate_plot_area,
    pixel_to_value,
)

__all__ = [
    "compute_image_boundaries",
    "create_configuration",
    "validate_configuration",
    "isolate_plot_area",
    "create_image_mask",
    "detect_points",
    "pixel_to_value",
]
