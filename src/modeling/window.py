# IMPORT AND CONSTANT

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd
LOGGER = logging.getLogger(__name__)

TIME_COL = "second_elapsed"
WINDOW_LENGTH_SEC = 6.0 # windowing parameters
STRIDE_SEC = 1.5
CLASS_ID = {"lng": 0,"srt": 1,"sgt": 2,"wlk": 3,} # creating class ID dictionary
LABEL_COLS = ["class_id"]
#-------------------------------------------------------------------------------------------------

def project_root() -> Path: # root folder
    return Path(__file__).resolve().parents[2]
#---------------------------------------------------------------------------------------------------
# WINDOWS DEFINITION
def generate_window_slices(n: int, window_size: int, stride: int) -> Iterable[tuple[int, int]]:
    # returning tuples of start-stop steps for each window
    start = 0 # first window at 0
    while start + window_size <= n: # while the new full size window fits
        yield start, start + window_size # slicing, rows extraction
        start += stride # move forward by one stride
# -----------------------------------------------------------------------------------------------------
# MAIN WINDOW CREATION FUNCTION WITH STATS RETURN

def process_file(in_path: Path, out_dir: Path, overwrite: bool) -> dict:
    stats = {"windows_total": 0,"windows_written": 0,"windows_skipped_existing": 0,}
    # getting class id
    class_label = in_path.stem.split("_", 1)[0]
    class_id = CLASS_ID[class_label]

    # getting data from csv
    df = pd.read_csv(in_path)
    df = df.sort_values(TIME_COL)
    dt = 0.02 # delta time known by default

    window_size_rows = int(round(WINDOW_LENGTH_SEC / dt)) # defining window size
    stride_rows = int(round(STRIDE_SEC / dt)) # steps for each stride
    if dt <= 0 or window_size_rows < 2 or stride_rows < 1:
        return stats # error check
    n_rows = len(df)
    ordered_cols = [col for col in df.columns if col not in LABEL_COLS]
    # leaving out label column

    base_df = df[ordered_cols].copy() # new df without label col
    out_dir.mkdir(parents=True, exist_ok=True) # output folder creation

    for window_index, (start, stop) in enumerate(generate_window_slices(n_rows, window_size_rows, stride_rows)):
        # for pair start/stop is in tuple list given by proper function
        stats["windows_total"] += 1
        window_df = base_df.iloc[start:stop].copy() # getting the relative data rows

        window_df["class_id"] = class_id # id assignment
        window_df = window_df[ordered_cols + LABEL_COLS]
        # inserting class col in window

        out_path = out_dir / f"{in_path.stem}_win{window_index:04d}.csv" #saving

        if out_path.exists() and not overwrite: # skip condition
            stats["windows_skipped_existing"] += 1
            continue

        window_df.to_csv(out_path, index=False)
        stats["windows_written"] += 1

    return stats
#---------------------------------------------------------------------------------------------------------------------------
# CALLING MAIN FUNC + LOG OUTPUT

def run_window(input_dir: Path,output_dir: Path,pattern: str = "*.csv",overwrite: bool = False,) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob(pattern))

    if not csv_paths:
        return

    total_written = 0
    total_windows = 0
    total_skipped_existing = 0

    for csv_path in csv_paths: # loop over every file in folder
        stats = process_file(csv_path, output_dir, overwrite)

        # counting processed windows
        total_written += stats["windows_written"]
        total_windows += stats["windows_total"]
        total_skipped_existing += stats["windows_skipped_existing"]

    total_skipped = total_windows - total_written
    LOGGER.info("Windows written: %d | skipped total: %d (existing=%d)",total_written,total_skipped,total_skipped_existing,
    )
#-------------------------------------------------------------------------------------------------------

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
