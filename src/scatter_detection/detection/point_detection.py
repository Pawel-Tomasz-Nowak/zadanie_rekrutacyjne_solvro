"""Operacje przygotowujące kadr i wydzielające markery z obrazu wykresu."""

import cv2


def isolate_plot_area(image, config: dict):
    """Ogranicza analizę do obszaru, w którym mogą znajdować się dane."""
    if image is None:
        raise ValueError("Nie da się wyizolować wykresu z pustego obrazu.")

    xmin, ymin, xmax, ymax = config["plot_bbox"]
    image_height, image_width = image.shape[:2]

    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
        raise ValueError("plot_bbox wychodzi poza rozmiar obrazu albo ma nieprawidłowe wartości.")

    plot_image = image[ymin:ymax, xmin:xmax].copy()
    expected_height = ymax - ymin
    expected_width = xmax - xmin

    if plot_image.shape[:2] != (expected_height, expected_width):
        raise RuntimeError("Wyizolowany fragment obrazu ma nieoczekiwany kształt.")

    return plot_image


def create_image_mask(plot_image, background_threshold: int = 245):
    """Oddziela ciemne elementy wykresu od jasnego tła przed analizą komponentów."""
    if plot_image is None or plot_image.size == 0:
        raise ValueError("Nie można stworzyć maski z pustego obrazu wykresu.")

    grayscale = cv2.cvtColor(plot_image, cv2.COLOR_BGR2GRAY)
    _, foreground_mask = cv2.threshold(
        grayscale,
        background_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)


def detect_points(plot_image, config: dict, min_area: int = 8, max_area: int = 500):
    """Wybiera zwarte komponenty, aby odrzucić osie i przypadkowe artefakty obrazu."""
    mask = create_image_mask(plot_image)
    number_of_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    xmin, ymin, _, _ = config["plot_bbox"]
    detected_points = []

    for label in range(1, number_of_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if not (min_area <= area <= max_area):
            continue
        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if abs(aspect_ratio - 1.0) > 0.5:
            continue

        x_local, y_local = centroids[label]
        detected_points.append(
            {
                "x_px": float(x_local + xmin),
                "y_px": float(y_local + ymin),
            }
        )

    return detected_points


def remove_thin_structures(
    foreground_mask,
    distance_threshold: float = 2.0,
):
    """Usuwa cienkie połączenia, których obecność zniekształcałaby detekcję markerów."""
    if foreground_mask is None or foreground_mask.size == 0:
        raise ValueError("Nie można usunąć linii z pustej maski.")
    if distance_threshold <= 0:
        raise ValueError("Próg odległości musi być większy od zera.")

    distance_map = cv2.distanceTransform(
        foreground_mask,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    marker_core_mask = cv2.inRange(
        distance_map,
        distance_threshold,
        float(distance_map.max()),
    )
    return marker_core_mask


def detect_points_line_aware(
    plot_image,
    config: dict,
    min_area: int = 2,
    max_area: int = 500,
    distance_threshold: float = 2.0,
):
    """Wykorzystuje rdzenie markerów, żeby ograniczyć wpływ linii w wykresach łączonych."""
    foreground_mask = create_image_mask(plot_image)
    marker_core_mask = remove_thin_structures(
        foreground_mask,
        distance_threshold=distance_threshold,
    )
    number_of_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        marker_core_mask,
        connectivity=8,
    )

    xmin, ymin, _, _ = config["plot_bbox"]
    detected_points = []

    for label in range(1, number_of_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if not (min_area <= area <= max_area):
            continue
        if width == 0 or height == 0:
            continue

        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        x_local, y_local = centroids[label]
        detected_points.append(
            {
                "x_px": float(x_local + xmin),
                "y_px": float(y_local + ymin),
            }
        )

    return detected_points


def pixel_to_value(points: list[dict[str, float]], calibration: dict) -> list[dict[str, float]]:
    """Przenosi wyniki detekcji do skali danych użytej w metadanych GT."""
    x_axis = calibration["x_axis"]
    y_axis = calibration["y_axis"]

    converted_points: list[dict[str, float]] = []

    for point in points:
        x_px = point["x_px"]
        y_px = point["y_px"]

        x_value = x_axis["val1"] + (
            (x_px - x_axis["px1"]) / (x_axis["px2"] - x_axis["px1"])
            * (x_axis["val2"] - x_axis["val1"])
        )
        y_value = y_axis["val1"] + (
            (y_axis["py1"] - y_px) / (y_axis["py1"] - y_axis["py2"])
            * (y_axis["val2"] - y_axis["val1"])
        )

        converted_point = dict(point)
        converted_point["x"] = float(x_value)
        converted_point["y"] = float(y_value)
        converted_points.append(converted_point)

    return converted_points
