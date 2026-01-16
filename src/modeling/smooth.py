from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - runtime dependency check
    savgol_filter = None


TIME_COL = "second_elapsed"


def _parse_prefixes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_quaternion_orientation(columns: list[str]) -> bool:
    colset = set(columns)
    quat_w = {"ori_w", "ori_qw"}
    quat_xyz = {"ori_x", "ori_y", "ori_z", "ori_qx", "ori_qy", "ori_qz"}
    has_w = any(name in colset for name in quat_w)
    has_xyz = any(name in colset for name in quat_xyz)
    return has_w and has_xyz


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def _smooth_columns(df: pd.DataFrame,columns: list[str],window_length: int,polyorder: int,mode: str,) -> pd.DataFrame:
    for col in columns:
        series = df[col].to_numpy(dtype=float)
        df[col] = savgol_filter(series,window_length=window_length,polyorder=polyorder,deriv=0,mode=mode,)
    return df


def _smooth_orientation(
    df: pd.DataFrame,
    columns: list[str],
    window_length: int,
    polyorder: int,
    mode: str,
    ori_units: str,
) -> pd.DataFrame:
    for col in columns:
        series = df[col].to_numpy(dtype=float)
        if ori_units == "deg":
            series = np.deg2rad(series)
        series = np.unwrap(series)
        series = savgol_filter(series,window_length=window_length,polyorder=polyorder,deriv=0,mode=mode,)
        series = _wrap_pi(series)
        if ori_units == "deg":
            series = np.rad2deg(series)
        df[col] = series
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Savitzky-Golay smoothing to CSV recordings.")
    parser.add_argument(
        "--input_dir",
        default="Data/processed/trimmed",
        help="Input directory containing trimmed CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default="Data/processed/smoothed",
        help="Output directory for smoothed CSV files.",
    )
    parser.add_argument("--window_length", type=int, default=21, help="Savgol window length (odd).")
    parser.add_argument("--polyorder", type=int, default=3, help="Savgol polynomial order.")
    parser.add_argument("--mode", default="interp", help="Savgol mode.")
    parser.add_argument(
        "--include_prefixes",
        default="acc_,gyr_",
        help="Comma-separated list of prefixes to smooth.",
    )
    parser.add_argument(
        "--smooth_orientation",
        action="store_true",
        default=False,
        help="Enable orientation smoothing for Euler angles.",
    )
    parser.add_argument(
        "--ori_units",
        choices=("rad", "deg"),
        default="deg",
        help="Orientation units (only used with --smooth_orientation).",
    )
    args = parser.parse_args()

    if savgol_filter is None:
        logging.error("SciPy is required. Install with: pip install scipy")
        return 1

    if args.window_length % 2 == 0 or args.window_length < args.polyorder + 2:
        logging.error("window_length must be odd and >= polyorder + 2.")
        return 1
    if args.polyorder >= args.window_length:
        logging.error("polyorder must be < window_length.")
        return 1

    project_root = Path(__file__).resolve().parents[2]
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("*.csv"))
    logging.info("Found %d CSV files in %s", len(csv_paths), input_dir)

    prefixes = _parse_prefixes(args.include_prefixes)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        output_path = output_dir / csv_path.name

        if len(df) < args.window_length:
            logging.warning(
                "Skipping smoothing for %s (rows=%d < window_length=%d).",
                csv_path.name,
                len(df),
                args.window_length,
            )
            df.to_csv(output_path, index=False)
            logging.info("Wrote %s", output_path)
            continue

        data_columns = list(df.columns)
        smooth_cols = [
            col
            for col in data_columns
            if any(col.startswith(prefix) for prefix in prefixes)
            and not col.startswith("ori_")
        ]

        ori_cols: list[str] = []
        if args.smooth_orientation:
            if _is_quaternion_orientation(data_columns):
                logging.warning(
                    "Skipping orientation smoothing for %s (quaternion-like columns detected).",
                    csv_path.name,
                )
            else:
                ori_cols = [col for col in data_columns if col.startswith("ori_")]

        cols_to_smooth = smooth_cols + ori_cols
        if not cols_to_smooth:
            logging.warning("No matching columns to smooth in %s.", csv_path.name)
            df.to_csv(output_path, index=False)
            logging.info("Wrote %s", output_path)
            continue

        logging.info("Smoothing columns for %s: %s", csv_path.name, ", ".join(cols_to_smooth))

        df = _smooth_columns(df, smooth_cols, args.window_length, args.polyorder, args.mode)
        if ori_cols:
            df = _smooth_orientation(
                df,
                ori_cols,
                args.window_length,
                args.polyorder,
                args.mode,
                args.ori_units,
            )

        df.to_csv(output_path, index=False)
        logging.info("Wrote %s", output_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())


