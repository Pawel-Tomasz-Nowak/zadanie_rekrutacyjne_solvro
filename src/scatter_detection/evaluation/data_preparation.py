"""Przygotowanie tabeli wyników obu metod detekcji."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from scatter_detection.config.constants import X_RANGE, Y_RANGE


DETECTION_METHODS = ("baseline", "line_aware")
METADATA_COLUMNS = (
    "function_type",
    "grid_lines",
    "x_sampling_strategy",
    "marker_size",
    "plot_style",
    "width",
    "height",
    "n_generated",
    "n_visible",
)


def compute_matching_tolerance(relative_fraction: float = 0.01) -> float:
    """Ustala skalę porównania niezależną od rozmiaru wygenerowanego obrazu."""
    if relative_fraction <= 0:
        raise ValueError("Ułamek tolerancji musi być większy od zera.")

    x_span = X_RANGE[1] - X_RANGE[0]
    y_span = Y_RANGE[1] - Y_RANGE[0]
    return max(x_span, y_span) * relative_fraction


def _match_points(
    ground_truth: list[dict[str, float]],
    predicted: list[dict[str, float]],
    tolerance: float,
) -> tuple[int, list[float]]:
    """Zapobiega wielokrotnemu zaliczeniu jednego punktu podczas porównania metod."""
    candidates = []
    for gt_index, gt_point in enumerate(ground_truth):
        for predicted_index, predicted_point in enumerate(predicted):
            distance = math.hypot(
                gt_point["x"] - predicted_point["x"],
                gt_point["y"] - predicted_point["y"],
            )
            if distance <= tolerance:
                candidates.append((distance, gt_index, predicted_index))

    candidates.sort()
    matched_ground_truth = set()
    matched_predicted = set()
    distances = []

    for distance, gt_index, predicted_index in candidates:
        if gt_index in matched_ground_truth or predicted_index in matched_predicted:
            continue
        matched_ground_truth.add(gt_index)
        matched_predicted.add(predicted_index)
        distances.append(distance)

    return len(distances), distances


def _read_json(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_row(
    image_id: str,
    metadata: dict[str, Any],
    ground_truth: list[dict[str, float]],
    predicted: list[dict[str, float]],
    detection_method: str,
    tolerance: float,
) -> dict[str, Any]:
    true_positives, distances = _match_points(ground_truth, predicted, tolerance)
    total_ground_truth = len(ground_truth)
    detected_points = len(predicted)
    false_positives = detected_points - true_positives
    false_negatives = total_ground_truth - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    mean_distance = sum(distances) / len(distances) if distances else None
    rmse_distance = (
        math.sqrt(sum(distance**2 for distance in distances) / len(distances))
        if distances
        else None
    )

    row: dict[str, Any] = {
        "image_id": image_id,
        "detection_method": detection_method,
        "tolerance": tolerance,
        "total_ground_truth": total_ground_truth,
        "detected_points": detected_points,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance": mean_distance,
        "rmse_distance": rmse_distance,
    }
    row.update({column: metadata.get(column) for column in METADATA_COLUMNS})
    return row


def prepare_evaluation_dataframe(
    project_root: str | Path | None = None,
    relative_tolerance: float = 0.01,
) -> pl.DataFrame:
    """Tworzy tabelę pozwalającą porównywać metody przy tych samych obrazach."""
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    generated_data = root / "src" / "scatter_detection" / "data" / "generated_data"
    metadata_dir = generated_data / "metadata"
    detected_points_dir = generated_data / "detected_points"
    tolerance = compute_matching_tolerance(relative_tolerance)

    rows = []
    for metadata_path in sorted(metadata_dir.glob("*_gt.json")):
        image_id = metadata_path.stem.removesuffix("_gt")
        metadata = _read_json(metadata_path)
        detected_payload = _read_json(
            detected_points_dir / f"{image_id}_detected_points.json"
        )
        ground_truth = metadata.get("points", [])

        for detection_method in DETECTION_METHODS:
            if detection_method not in detected_payload:
                raise KeyError(
                    f"Plik {image_id}_detected_points.json nie zawiera metody "
                    f"„{detection_method}”."
                )
            rows.append(
                _build_row(
                    image_id=image_id,
                    metadata=metadata,
                    ground_truth=ground_truth,
                    predicted=detected_payload[detection_method],
                    detection_method=detection_method,
                    tolerance=tolerance,
                )
            )

    if not rows:
        raise FileNotFoundError(f"Nie znaleziono metadanych w katalogu: {metadata_dir}")

    return pl.DataFrame(rows).sort(["image_id", "detection_method"])


def save_evaluation_dataframe(
    dataframe: pl.DataFrame,
    output_dir: str | Path,
) -> None:
    """Zachowuje wyniki w formatach wygodnych do analizy tabelarycznej."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataframe.write_parquet(output_path / "detection_evaluation.parquet")
    dataframe.write_csv(output_path / "detection_evaluation.csv")


if __name__ == "__main__":
    evaluation_df = prepare_evaluation_dataframe()
    output_directory = Path(__file__).resolve().parents[1] / "data" / "evaluation"
    save_evaluation_dataframe(evaluation_df, output_directory)
    print(f"Przygotowano {evaluation_df.height} wierszy danych ewaluacyjnych.")
    print(f"Zapisano wyniki w: {output_directory}")
