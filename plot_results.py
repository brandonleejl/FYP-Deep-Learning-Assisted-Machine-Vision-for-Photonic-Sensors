#!/usr/bin/env python3
"""Generate report-ready performance plots from predictions CSV."""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


REQUIRED_COLUMNS = ["actual_ph", "pred_ph_expected", "abs_error_expected"]
COLUMN_ALIASES = {
    "pred_ph_expected": ["predicted_ph_expected"],
    "abs_error_expected": ["abs_error"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prediction performance figures and export summary metrics."
    )
    parser.add_argument("--csv_path", default="", help="Path to predictions CSV")
    parser.add_argument("--out_dir", default="", help="Directory to save outputs")
    return parser.parse_args()


def resolve_defaults(csv_path: str, out_dir: str) -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")

    resolved_csv = csv_path.strip()
    if not resolved_csv:
        candidates = sorted(tf for tf in os.listdir(results_dir) if tf.endswith("_predictions.csv")) if os.path.isdir(results_dir) else []
        if not candidates:
            raise FileNotFoundError(
                "No --csv_path provided and no '*_predictions.csv' files found in ./results"
            )
        resolved_csv = os.path.join(results_dir, candidates[-1])

    resolved_out = out_dir.strip() if out_dir.strip() else os.path.join(results_dir, "plots")
    return {"csv_path": resolved_csv, "out_dir": resolved_out}


def load_csv_data(csv_path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if pd is not None:
        df = pd.read_csv(csv_path)
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[canonical] = df[alias]
                        break

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Keep only numeric rows for required columns.
        numeric_df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
        clean = numeric_df.dropna(axis=0, how="any")

        if clean.empty:
            raise ValueError("No valid numeric rows left after removing NaNs.")

        return {
            "actual_ph": clean["actual_ph"].to_numpy(dtype=np.float64),
            "pred_ph_expected": clean["pred_ph_expected"].to_numpy(dtype=np.float64),
            "abs_error_expected": clean["abs_error_expected"].to_numpy(dtype=np.float64),
        }

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or missing header row.")

        fieldnames = list(reader.fieldnames)
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical not in fieldnames:
                for alias in aliases:
                    if alias in fieldnames:
                        fieldnames.append(canonical)
                        break

        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        actual_vals: List[float] = []
        pred_vals: List[float] = []
        abs_err_vals: List[float] = []

        for row in reader:
            pred_col = "pred_ph_expected" if row.get("pred_ph_expected") not in (None, "") else "predicted_ph_expected"
            abs_col = "abs_error_expected" if row.get("abs_error_expected") not in (None, "") else "abs_error"
            try:
                actual = float((row.get("actual_ph") or "").strip())
                pred = float((row.get(pred_col) or "").strip())
                abs_err = float((row.get(abs_col) or "").strip())
            except ValueError:
                continue

            if any(math.isnan(x) for x in (actual, pred, abs_err)):
                continue

            actual_vals.append(actual)
            pred_vals.append(pred)
            abs_err_vals.append(abs_err)

    if not actual_vals:
        raise ValueError("No valid numeric rows found after filtering invalid values/NaNs.")

    return {
        "actual_ph": np.asarray(actual_vals, dtype=np.float64),
        "pred_ph_expected": np.asarray(pred_vals, dtype=np.float64),
        "abs_error_expected": np.asarray(abs_err_vals, dtype=np.float64),
    }


def compute_metrics(actual: np.ndarray, pred: np.ndarray, abs_err: np.ndarray) -> Dict[str, float]:
    residual = pred - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))

    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    metrics = {
        "num_samples": float(actual.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc_tol_0.1": float(np.mean(abs_err <= 0.1)),
        "acc_tol_0.3": float(np.mean(abs_err <= 0.3)),
        "acc_tol_0.5": float(np.mean(abs_err <= 0.5)),
    }
    return metrics


def save_metrics(metrics: Dict[str, float], out_dir: str) -> None:
    metrics_path = os.path.join(out_dir, "metrics.txt")
    lines = [
        f"Samples: {int(metrics['num_samples'])}",
        f"MAE: {metrics['mae']:.6f}",
        f"RMSE: {metrics['rmse']:.6f}",
        f"R^2: {metrics['r2']:.6f}",
        f"Accuracy within +/-0.1: {metrics['acc_tol_0.1']:.4f}",
        f"Accuracy within +/-0.3: {metrics['acc_tol_0.3']:.4f}",
        f"Accuracy within +/-0.5: {metrics['acc_tol_0.5']:.4f}",
    ]

    text = "\n".join(lines)
    print(text)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def save_figures(
    actual: np.ndarray,
    pred: np.ndarray,
    abs_err: np.ndarray,
    metrics: Dict[str, float],
    out_dir: str,
) -> None:
    plt.rcParams.update({"font.size": 11})

    # FIG1: actual vs predicted scatter with y=x line and metrics box.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual, pred, alpha=0.8)
    lo = float(min(np.min(actual), np.min(pred)))
    hi = float(max(np.max(actual), np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("Actual vs Predicted pH")
    ax.set_xlabel("Actual pH")
    ax.set_ylabel("Predicted pH (expected)")
    summary_text = (
        f"MAE: {metrics['mae']:.3f}\n"
        f"RMSE: {metrics['rmse']:.3f}\n"
        f"R^2: {metrics['r2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        summary_text,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_actual_vs_pred.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig1_actual_vs_pred.pdf"))
    plt.close(fig)

    # FIG2: residual plot.
    residual = pred - actual
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual, residual, alpha=0.8)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Residuals vs Actual pH")
    ax.set_xlabel("Actual pH")
    ax.set_ylabel("Residual (Predicted - Actual)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_residuals.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig2_residuals.pdf"))
    plt.close(fig)

    # FIG3: absolute error histogram.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(abs_err, bins=20)
    ax.axvline(0.1, linestyle="--")
    ax.axvline(0.3, linestyle="--")
    ax.axvline(0.5, linestyle="--")
    ax.set_title("Histogram of Absolute Error")
    ax.set_xlabel("Absolute Error |Predicted - Actual|")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_abs_error_hist.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig3_abs_error_hist.pdf"))
    plt.close(fig)

    # FIG4: tolerance accuracy curve.
    tol = np.linspace(0.0, 1.0, 101)
    acc = np.array([np.mean(abs_err <= t) for t in tol], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tol, acc)
    ax.set_title("Tolerance Accuracy Curve")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Accuracy: P(|error| <= tolerance)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_tolerance_curve.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig4_tolerance_curve.pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        resolved = resolve_defaults(args.csv_path, args.out_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        data = load_csv_data(resolved["csv_path"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    os.makedirs(resolved["out_dir"], exist_ok=True)

    actual = data["actual_ph"]
    pred = data["pred_ph_expected"]
    abs_err = data["abs_error_expected"]

    metrics = compute_metrics(actual, pred, abs_err)
    save_metrics(metrics, resolved["out_dir"])
    save_figures(actual, pred, abs_err, metrics, resolved["out_dir"])

    print(f"Saved metrics and figures to: {os.path.abspath(resolved['out_dir'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
