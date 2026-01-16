from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

TIME_COL = "second_elapsed"
WINDOW_LENGTH_SEC = 6.0
STRIDE_SEC = 1.5

CLASS_ID = {
    "lng": 0,
    "srt": 1,
    "sgt": 2,
    "wlk": 3,
}
LABEL_COLS = ["class_id"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_label_from_filename(path: Path) -> int:
    class3 = path.stem.split("_", 1)[0]
    return CLASS_ID[class3]


def estimate_dt_seconds(df: pd.DataFrame) -> float:
    series = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(series)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def generate_window_slices(n: int, window_size: int, stride: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start + window_size <= n:
        yield start, start + window_size
        start += stride


def process_file(in_path: Path, out_dir: Path, overwrite: bool) -> dict:
    stats = {
        "windows_total": 0,
        "windows_written": 0,
        "windows_skipped_existing": 0,
    }

    class_id = parse_label_from_filename(in_path)

    df = pd.read_csv(in_path)
    df = df.sort_values(TIME_COL)
    dt = estimate_dt_seconds(df)
    window_size_rows = int(round(WINDOW_LENGTH_SEC / dt))
    stride_rows = int(round(STRIDE_SEC / dt))
    if dt <= 0 or window_size_rows < 2 or stride_rows < 1:
        return stats

    n_rows = len(df)

    ordered_cols = list(df.columns)
    if TIME_COL in ordered_cols:
        ordered_cols.remove(TIME_COL)
        ordered_cols = [TIME_COL] + ordered_cols
    for col in LABEL_COLS:
        if col in ordered_cols:
            ordered_cols.remove(col)

    base_df = df[ordered_cols].copy()
    out_dir.mkdir(parents=True, exist_ok=True)

    for window_index, (start, stop) in enumerate(
        generate_window_slices(n_rows, window_size_rows, stride_rows)
    ):
        stats["windows_total"] += 1
        window_df = base_df.iloc[start:stop].copy()

        window_df["class_id"] = class_id
        window_df = window_df[ordered_cols + LABEL_COLS]

        out_path = out_dir / f"{in_path.stem}__win{window_index:04d}.csv"
        if out_path.exists() and not overwrite:
            stats["windows_skipped_existing"] += 1
            continue
        window_df.to_csv(out_path, index=False)
        stats["windows_written"] += 1
    return stats


def run_window(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob(pattern))
    if not csv_paths:
        return

    total_written = 0
    total_windows = 0
    total_skipped_existing = 0
    for csv_path in csv_paths:
        stats = process_file(csv_path, output_dir, overwrite)
        total_written += stats["windows_written"]
        total_windows += stats["windows_total"]
        total_skipped_existing += stats["windows_skipped_existing"]

    total_skipped = total_windows - total_written
    LOGGER.info(
        "Windows written: %d | skipped total: %d (existing=%d)",
        total_written,
        total_skipped,
        total_skipped_existing,
    )


def main() -> None:
    root = project_root()
    input_dir = root / "Data" / "processed" / "smoothed"
    output_dir = root / "Data" / "processed" / "windows"
    pattern = "*.csv"
    overwrite = False
    run_window(input_dir=input_dir, output_dir=output_dir, pattern=pattern, overwrite=overwrite)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
