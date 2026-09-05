"""Generate auditable, read-only evidence for the SSDC-UAV YOLO12s review.

This script never changes source datasets or experiment directories.  It reads:
  * SSDC-UAV YOLO labels and image sizes;
  * the COCO-pretrained YOLO12s baseline's predictions.json/result.log;
  * the maintained test-result summary CSV; and
  * the supplied source-statistics workbook.

It writes compact derived evidence files below .codex/ssdc_uav_yolo12s_evidence/.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".codex" / "ssdc_uav_yolo12s_evidence"
DATASET = ROOT / "datasets" / "SSDC-UAV_yolo"
TEST_LABELS = DATASET / "labels" / "test"
TEST_IMAGES = DATASET / "images" / "test"
BASELINE_DIR = (
    ROOT
    / "runs"
    / "ssdc_uav_test"
    / "size-s-base"
    / "yolo12s-baseline_ssdc_uav_test_exp01"
)
SUMMARY_CSV = ROOT / "runs" / "test_result" / "SSDC-UAV_Test_Result.csv"
STATS_XLSX = ROOT / "datasets" / "SSDC-UAV-info" / "SSDC-UAV-info.xlsx"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj)!r}")


def write_json(name: str, value: Any) -> Path:
    destination = OUT / name
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def coco_size_bin(area: float) -> str:
    if area < 32**2:
        return "small(<32² px)"
    if area < 96**2:
        return "medium(32²–<96² px)"
    return "large(≥96² px)"


def side_bin(area: float) -> str:
    side = math.sqrt(area)
    if side < 16:
        return "side<16 px"
    if side < 32:
        return "16≤side<32 px"
    if side < 48:
        return "32≤side<48 px"
    if side < 64:
        return "48≤side<64 px"
    if side < 96:
        return "64≤side<96 px"
    return "side≥96 px"


def find_image(stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = TEST_IMAGES / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No test image found for label stem {stem!r}")


def read_ground_truth() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    dimension_counter: Counter[tuple[int, int]] = Counter()
    class_counter: Counter[str] = Counter()
    all_objects: list[dict[str, Any]] = []

    for label_path in sorted(TEST_LABELS.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image(stem)
        with Image.open(image_path) as image:
            width, height = image.size
        dimension_counter[(width, height)] += 1
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected label format in {label_path}:{line_number + 1}: {line!r}")
            category, x_center, y_center, box_width, box_height = map(float, parts)
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            box = (
                x_center - box_width / 2,
                y_center - box_height / 2,
                x_center + box_width / 2,
                y_center + box_height / 2,
            )
            area = box_width * box_height
            record = {
                "key": f"{stem}:{len(records)}",
                "image_id": stem,
                "class_id": int(category),
                "box": box,
                "width_px": box_width,
                "height_px": box_height,
                "area_px2": area,
                "coco_size": coco_size_bin(area),
                "side_size": side_bin(area),
            }
            records.append(record)
            all_objects.append(record)
            class_counter[str(int(category))] += 1
        gt_by_image[stem] = records

    width_values = [item["width_px"] for item in all_objects]
    height_values = [item["height_px"] for item in all_objects]
    area_values = [item["area_px2"] for item in all_objects]
    side_values = [math.sqrt(value) for value in area_values]
    stats = {
        "images": len(gt_by_image),
        "instances": len(all_objects),
        "classes": dict(class_counter),
        "image_dimensions": [
            {"width": width, "height": height, "images": count}
            for (width, height), count in sorted(dimension_counter.items())
        ],
        "width_px": quantile_summary(width_values),
        "height_px": quantile_summary(height_values),
        "area_px2": quantile_summary(area_values),
        "sqrt_area_px": quantile_summary(side_values),
        "coco_size_counts": dict(Counter(item["coco_size"] for item in all_objects)),
        "side_size_counts": dict(Counter(item["side_size"] for item in all_objects)),
    }
    return gt_by_image, {"objects": all_objects, "stats": stats}


def quantile_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def q(p: float) -> float:
        position = (len(ordered) - 1) * p
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return round(ordered[low], 4)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 4)

    return {
        "min": round(ordered[0], 4),
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def read_predictions() -> list[dict[str, Any]]:
    raw = json.loads((BASELINE_DIR / "predictions.json").read_text(encoding="utf-8"))
    predictions = []
    for index, record in enumerate(raw):
        x, y, width, height = (float(v) for v in record["bbox"])
        predictions.append(
            {
                "index": index,
                "image_id": str(record["image_id"]),
                "score": float(record["score"]),
                "class_id": int(record["category_id"]) - 1,
                "box": (x, y, x + width, y + height),
            }
        )
    return predictions


def score_stream(
    predictions: list[dict[str, Any]],
    gt_by_image: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Greedy confidence-ordered one-to-one matching used by detector AP evaluation."""
    matched: dict[str, set[int]] = defaultdict(set)
    stream: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        candidates = gt_by_image.get(prediction["image_id"], [])
        best_index = -1
        best_iou = 0.0
        for gt_index, target in enumerate(candidates):
            if gt_index in matched[prediction["image_id"]] or target["class_id"] != prediction["class_id"]:
                continue
            overlap = iou_xyxy(prediction["box"], target["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        is_tp = best_index >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched[prediction["image_id"]].add(best_index)
        stream.append(
            {
                "score": prediction["score"],
                "is_tp": is_tp,
                "image_id": prediction["image_id"],
                "gt_index": best_index if is_tp else None,
                "match_iou": best_iou if is_tp else 0.0,
            }
        )
    return stream


def best_f1_operating_point(stream: list[dict[str, Any]], total_gt: int) -> dict[str, Any]:
    tp = 0
    fp = 0
    best: dict[str, Any] | None = None
    for index, item in enumerate(stream):
        if item["is_tp"]:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_gt
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        current = {
            "stream_index": index,
            "confidence_threshold": item["score"],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": total_gt - tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if best is None or current["f1"] > best["f1"]:
            best = current
    if best is None:
        raise ValueError("No predictions are available")
    return best


def matched_at_threshold(
    stream: list[dict[str, Any]], threshold: float
) -> dict[str, dict[int, float]]:
    matched: dict[str, dict[int, float]] = defaultdict(dict)
    for item in stream:
        if item["score"] < threshold:
            break
        if item["is_tp"]:
            matched[item["image_id"]][int(item["gt_index"])] = item["match_iou"]
    return matched


def aggregate_by_size(
    gt_by_image: dict[str, list[dict[str, Any]]],
    matched_50: dict[str, dict[int, float]],
    matched_75: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("coco_size", "side_size"):
        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for image_id, objects in gt_by_image.items():
            for index, item in enumerate(objects):
                grouped[str(item[field])].append((image_id, index))
        for size_name, keys in grouped.items():
            count = len(keys)
            hits_50 = sum(index in matched_50.get(image_id, {}) for image_id, index in keys)
            hits_75 = sum(index in matched_75.get(image_id, {}) for image_id, index in keys)
            result.append(
                {
                    "bin_type": field,
                    "bin": size_name,
                    "gt": count,
                    "matched_iou50": hits_50,
                    "recall_iou50": round(hits_50 / count, 6),
                    "matched_iou75": hits_75,
                    "recall_iou75": round(hits_75 / count, 6),
                    "recall_gap_50_to_75": round((hits_50 - hits_75) / count, 6),
                }
            )
    return result


def error_profile(
    predictions: list[dict[str, Any]],
    gt_by_image: dict[str, list[dict[str, Any]]],
    confidence_threshold: float,
    matched_50: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    """Classify unmatched GTs by whether an accepted prediction gets near the box."""
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= confidence_threshold:
            accepted[prediction["image_id"]].append(prediction)

    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for image_id, objects in gt_by_image.items():
        for index, target in enumerate(objects):
            size_name = str(target["coco_size"])
            if index in matched_50.get(image_id, {}):
                counters[("all", size_name)]["matched_iou50"] += 1
                continue
            best_overlap = 0.0
            for prediction in accepted.get(image_id, []):
                if prediction["class_id"] == target["class_id"]:
                    best_overlap = max(best_overlap, iou_xyxy(prediction["box"], target["box"]))
            if best_overlap < 0.1:
                counters[("all", size_name)]["miss_no_overlap_lt0.1"] += 1
            else:
                counters[("all", size_name)]["miss_partial_overlap_0.1_to_lt0.5"] += 1

    rows: list[dict[str, Any]] = []
    for (_, size_name), counts in sorted(counters.items()):
        total = sum(counts.values())
        row = {"coco_size": size_name, "gt": total}
        row.update(counts)
        for key in ("matched_iou50", "miss_no_overlap_lt0.1", "miss_partial_overlap_0.1_to_lt0.5"):
            row[f"{key}_rate"] = round(counts[key] / total, 6) if total else None
        rows.append(row)
    return rows


def source_statistics_workbook() -> dict[str, Any]:
    workbook = pd.ExcelFile(STATS_XLSX)
    sheets = {}
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(STATS_XLSX, sheet_name=sheet_name, header=None)
        frame = frame.where(pd.notna(frame), None)
        sheets[sheet_name] = frame.values.tolist()
    return {"sheet_names": workbook.sheet_names, "sheets": sheets}


def read_summary_metrics() -> list[dict[str, Any]]:
    with SUMMARY_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_columns = {
        "Epoch",
        "Parameters (M)",
        "GFLOPS (imgsz=640)",
        "Precision",
        "Recall",
        "F1-Score",
        "mAP50 (IoU=0.50)",
        "mAP75 (IoU=0.75)",
        "mAP50-95 (IoU=0.50:0.95)",
    }
    baseline = next(row for row in rows if row["Model"] == "YOLOv12s-baseline")
    normalized = []
    for row in rows:
        output = dict(row)
        for field in numeric_columns:
            output[field] = float(row[field])
        output["delta_mAP50_95_pp"] = round(
            output["mAP50-95 (IoU=0.50:0.95)"] - float(baseline["mAP50-95 (IoU=0.50:0.95)"]),
            3,
        )
        output["delta_mAP75_pp"] = round(
            output["mAP75 (IoU=0.75)"] - float(baseline["mAP75 (IoU=0.75)"]),
            3,
        )
        output["delta_recall_pp"] = round(output["Recall"] - float(baseline["Recall"]), 3)
        output["delta_precision_pp"] = round(output["Precision"] - float(baseline["Precision"]), 3)
        normalized.append(output)
    return sorted(normalized, key=lambda row: row["mAP50-95 (IoU=0.50:0.95)"], reverse=True)


def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    destination = OUT / name
    keys = list(rows[0]) if rows else []
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def test_run_manifest() -> list[dict[str, Any]]:
    test_root = ROOT / "runs" / "ssdc_uav_test"
    rows = []
    for log_path in sorted(test_root.rglob("result.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")

        def metric(label: str) -> float | None:
            match = re.search(re.escape(label) + r"\s*\|\s*([0-9.]+)", text)
            return float(match.group(1)) if match else None

        rows.append(
            {
                "run_folder": str(log_path.parent.relative_to(ROOT)),
                "parameters_m": metric("Parameters (M)"),
                "gflops": metric("GFLOPS (imgsz=640)"),
                "precision": metric("Precision"),
                "recall": metric("Recall"),
                "map50": metric("mAP50 (IoU=0.50)"),
                "map75": metric("mAP75 (IoU=0.75)"),
                "map50_95": metric("mAP50-95 (IoU=0.50:0.95)"),
                "has_predictions_json": (log_path.parent / "predictions.json").exists(),
            }
        )
    return rows


def main(output_dir: Path | None = None) -> None:
    global OUT
    if output_dir is not None:
        OUT = output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    gt_by_image, gt = read_ground_truth()
    predictions = read_predictions()
    total_gt = gt["stats"]["instances"]

    streams = {threshold: score_stream(predictions, gt_by_image, threshold) for threshold in (0.5, 0.75)}
    operating = best_f1_operating_point(streams[0.5], total_gt)
    confidence = operating["confidence_threshold"]
    matched_50 = matched_at_threshold(streams[0.5], confidence)
    matched_75 = matched_at_threshold(streams[0.75], confidence)
    matched_50_count = sum(len(value) for value in matched_50.values())
    matched_75_count = sum(len(value) for value in matched_75.values())

    size_rows = aggregate_by_size(gt_by_image, matched_50, matched_75)
    error_rows = error_profile(predictions, gt_by_image, confidence, matched_50)
    metrics = read_summary_metrics()
    run_manifest = test_run_manifest()

    write_json("test_gt_statistics.json", gt["stats"])
    write_json(
        "baseline_operating_point.json",
        {
            "matching_method": (
                "single-class, confidence-descending greedy one-to-one matching; "
                "the best global F1 threshold is reconstructed from saved predictions.json"
            ),
            "iou50_best_f1": operating,
            "at_same_confidence_iou75": {
                "confidence_threshold": confidence,
                "matched": matched_75_count,
                "total_gt": total_gt,
                "recall": round(matched_75_count / total_gt, 6),
            },
            "at_same_confidence_iou50": {
                "matched": matched_50_count,
                "total_gt": total_gt,
                "recall": round(matched_50_count / total_gt, 6),
            },
            "saved_log_reference": {
                "precision": 0.84911,
                "recall": 0.81341,
                "f1": 0.83088,
                "map50": 0.88267,
                "map75": 0.60853,
                "map50_95": 0.55639,
            },
        },
    )
    write_csv("baseline_size_recall_profile.csv", size_rows)
    write_csv("baseline_error_profile.csv", error_rows)
    write_csv("experiment_comparison.csv", metrics)
    write_csv("test_run_manifest.csv", run_manifest)
    write_json("source_statistics_workbook.json", source_statistics_workbook())
    write_json(
        "evidence_manifest.json",
        {
            "root": str(ROOT),
            "sources": [
                {
                    "path": str(DATASET / "ssdc-uav.yaml"),
                    "sha256": sha256(DATASET / "ssdc-uav.yaml"),
                },
                {
                    "path": str(BASELINE_DIR / "result.log"),
                    "sha256": sha256(BASELINE_DIR / "result.log"),
                },
                {
                    "path": str(BASELINE_DIR / "predictions.json"),
                    "sha256": sha256(BASELINE_DIR / "predictions.json"),
                },
                {"path": str(SUMMARY_CSV), "sha256": sha256(SUMMARY_CSV)},
                {"path": str(STATS_XLSX), "sha256": sha256(STATS_XLSX)},
            ],
            "derived_files": [
                "test_gt_statistics.json",
                "baseline_operating_point.json",
                "baseline_size_recall_profile.csv",
                "baseline_error_profile.csv",
                "experiment_comparison.csv",
                "test_run_manifest.csv",
                "source_statistics_workbook.json",
            ],
        },
    )
    print(f"Wrote evidence to {OUT}")
    print(f"Test GT: {total_gt} instances across {len(gt_by_image)} images")
    print(f"Baseline saved predictions: {len(predictions)}")
    print(
        "Best reconstructed IoU=0.50 operating point: "
        f"conf={confidence:.6f}, P={operating['precision']:.6f}, "
        f"R={operating['recall']:.6f}, F1={operating['f1']:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional writable directory for derived evidence. Defaults to .codex/ssdc_uav_yolo12s_evidence.",
    )
    arguments = parser.parse_args()
    main(arguments.output_dir)
