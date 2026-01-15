# IMPORT AND COSTANTS DEFINITION

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
# check of date-time and numeric like type of the series

LOGGER = logging.getLogger(__name__) # to identify provenience of messages
SENSOR_PREFIX = {"Accelerometer": "acc_", "Gyroscope": "gyr_", "Orientation": "ori_"}
SENSORS = list(SENSOR_PREFIX.keys())
FREQ = "20ms"
# -----------------------------------------------------------------------------------------------
# ROOT FOLDER LOCATING
def project_root() -> Path:
    # Return the project root folder.
    return Path(__file__).resolve().parents[2]
#-----------------------------------------------------------------------------------------
# CREATING AND FILLING A NESTED DICTIONARIES
# Discover sensor CSV files and group them by their same recording key
# (group of 3 different sensors data of the same recording session aggregated)

def discover_files(extracted_dir: Path) -> Dict[str, Dict[str, Path]]:
    if not extracted_dir.exists(): # if not FALSE
        raise FileNotFoundError(f"Extracted folder not found: {extracted_dir}")

    grouped: Dict[str, Dict[str, Path]] = {}
    # creating a multidimensional nested dictionary:
    # {key 1 (rec_key name) : { key 2 (sensor) : associated path , ... }}


    for path in extracted_dir.rglob("*.csv"): # for every path in input folder
        parts = path.stem.rsplit("_", 2) # splitting path in 3: <class> <ID> <sensor>
        if len(parts) != 3:
            LOGGER.warning("Skipping unrecognized filename: %s", path.name)
            continue # iflen(parts) != 3 go on with next path

        class_label, rec_id, sensor = parts # assignment of name parts to variables
        rec_key = f"{class_label}_{rec_id}" # I'm taking future name rec_key

        grouped.setdefault(rec_key, {}) # adding the first <class>_<ID> key to grouped dict [1st layer]

        if sensor in grouped[rec_key]: # preventing having 2 same sensor keys
            LOGGER.warning("Duplicate %s for %s, keeping first", sensor, rec_key)
            continue

        grouped[rec_key][sensor] = path  # adding to each key pair the relative path [2nd layer]

    return grouped
#-------------------------------------------------------------------------------------------------------

def parse_time_series(s: pd.Series) -> pd.DatetimeIndex:
    # Parse a time series into a DatetimeIndex.
    if is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        return pd.DatetimeIndex(dt)

    if is_numeric_dtype(s):
        values = pd.to_numeric(s, errors="coerce")
        abs_med = np.nanmedian(np.abs(values.to_numpy(dtype=float)))
        if np.isnan(abs_med):
            dt = pd.to_datetime(values, errors="coerce", utc=True)
            return pd.DatetimeIndex(dt)
        if abs_med > 1e17:
            unit = "ns"
        elif abs_med > 1e14:
            unit = "us"
        elif abs_med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        dt = pd.to_datetime(values, unit=unit, errors="coerce", utc=True)
        return pd.DatetimeIndex(dt)

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return pd.DatetimeIndex(dt)
#------------------------------------------------------------------------------------------------------

def load_sensor_frame(path: Path, sensor: str) -> Optional[pd.DataFrame]:
    # Load a sensor CSV and return a frame with prefixed measurement columns.
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    if "time" not in df.columns:
        df = pd.read_csv(path, sep=";", engine="python")
        df.columns = [str(col).strip() for col in df.columns]
        df = df.loc[:, [col for col in df.columns if col and not col.lower().startswith("unnamed")]]
    if "time" not in df.columns:
        LOGGER.warning("Missing time column in %s", path.name)
        return None

    parsed_time = parse_time_series(df["time"])
    invalid = parsed_time.isna()
    if invalid.any():
        LOGGER.warning("Dropped %s rows with invalid time in %s", invalid.sum(), path.name)
    if invalid.all():
        LOGGER.warning("No valid time values in %s", path.name)
        return None

    df = df.loc[~invalid].copy()
    df["time"] = parsed_time[~invalid].values

    drop_cols = [col for col in df.columns if col.lower() in {"second_elapsed", "seconds_elapsed"}]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    measurement_cols = [col for col in df.columns if col != "time"]
    if measurement_cols:
        df[measurement_cols] = df[measurement_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )
    prefix = SENSOR_PREFIX[sensor]
    rename = {col: f"{prefix}{col}" for col in measurement_cols}
    df = df.rename(columns=rename)
    return df


def resample_to_grid(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, t_start: pd.Timestamp, t_end: pd.Timestamp
) -> pd.DataFrame:
    # Resample a sensor frame onto the target time grid.
    df = df.sort_values("time").drop_duplicates(subset="time", keep="first")
    df = df.set_index("time")
    df = df.loc[(df.index >= t_start) & (df.index <= t_end)]
    df = df.reindex(df.index.union(target_index)).sort_index()

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        df[numeric_cols] = df[numeric_cols].ffill(limit=2).bfill(limit=2)
    df = df.reindex(target_index)
    return df


def merge_recording(
    rec_key: str, sensor_paths: Dict[str, Path], output_dir: Path
) -> Optional[Path]:
    # Merge sensors for a recording and save the merged CSV.
    frames = []
    tmins = []
    tmaxs = []
    used_files = []

    for sensor in SENSORS:
        path = sensor_paths[sensor]
        df = load_sensor_frame(path, sensor)
        if df is None:
            return None
        frames.append(df)
        tmins.append(df["time"].min())
        tmaxs.append(df["time"].max())
        used_files.append(path.name)

    t_start = max(tmins)
    t_end = min(tmaxs)
    if t_end <= t_start:
        LOGGER.warning("No overlap for %s (start=%s, end=%s)", rec_key, t_start, t_end)
        return None

    target_index = pd.date_range(start=t_start, end=t_end, freq=FREQ)
    resampled = [resample_to_grid(df, target_index, t_start, t_end) for df in frames]

    merged = pd.concat(resampled, axis=1)
    if merged.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in merged frame for {rec_key}")

    second_elapsed = (merged.index - merged.index[0]).total_seconds()
    merged.insert(0, "second_elapsed", second_elapsed)
    merged = merged.reset_index(drop=True)

    nan_ratios = merged.isna().mean()
    high_nan = nan_ratios[nan_ratios > 0.01]
    for col, ratio in high_nan.items():
        LOGGER.warning("High NaN ratio for %s.%s: %.2f%%", rec_key, col, ratio * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rec_key}.csv"
    merged.to_csv(out_path, index=False)

    overlap_seconds = (t_end - t_start).total_seconds()
    LOGGER.info(
        "Merged %s using %s | overlap %.2fs | output %s | shape %s",
        rec_key,
        ", ".join(used_files),
        overlap_seconds,
        out_path,
        merged.shape,
    )
    return out_path


def run_merge(extracted_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    # Run the merge process for all recordings in extracted_dir.
    root = project_root()
    extracted_dir = extracted_dir or (root / "Data" / "interim" / "extracted")
    output_dir = output_dir or (root / "Data" / "interim" / "merged")

    grouped = discover_files(extracted_dir)
    LOGGER.info("Recordings found: %s", len(grouped))

    merged_count = 0
    for rec_key, sensors in sorted(grouped.items()):
        missing = set(SENSORS) - sensors.keys()
        if missing:
            LOGGER.warning("Skipping %s, missing: %s", rec_key, ", ".join(sorted(missing)))
            continue
        out_path = merge_recording(rec_key, sensors, output_dir)
        if out_path is not None:
            merged_count += 1

    LOGGER.info("Recordings merged: %s", merged_count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_merge()


if __name__ == "__main__":
    main()
