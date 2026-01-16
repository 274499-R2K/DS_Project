
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


LOGGER = logging.getLogger(__name__)
MIN_ROWS = 10


def project_root() -> Path:
    # Return the project root folder.
    return Path(__file__).resolve().parents[2]


def trim_dataframe(
    df: pd.DataFrame, trim_start: float, trim_end: float, time_col: str = "second_elapsed"
) -> pd.DataFrame:
    # Trim a dataframe based on elapsed seconds.
    df = df.sort_values(time_col)
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    lower = t_min + trim_start
    upper = t_max - trim_end
    return df.loc[(df[time_col] >= lower) & (df[time_col] <= upper)].copy()


def process_file(
    in_path: Path, out_path: Path, trim_start: float, trim_end: float, overwrite: bool
) -> bool:
    if out_path.exists() and not overwrite:
        LOGGER.info("Skipping %s (exists, use --overwrite to replace)", out_path.name)
        return False

    df = pd.read_csv(in_path)
    if "second_elapsed" not in df.columns:
        LOGGER.warning("Skipping %s: missing second_elapsed column", in_path.name)
        return False

    if not is_numeric_dtype(df["second_elapsed"]):
        df["second_elapsed"] = pd.to_numeric(df["second_elapsed"], errors="coerce")

    invalid = df["second_elapsed"].isna()
    if invalid.all():
        LOGGER.warning("Skipping %s: second_elapsed is not numeric", in_path.name)
        return False
    if invalid.any():
        LOGGER.warning("Dropping %s rows with invalid second_elapsed in %s", invalid.sum(), in_path.name)
        df = df.loc[~invalid].copy()

    df = df.sort_values("second_elapsed")
    t_min = df["second_elapsed"].min()
    t_max = df["second_elapsed"].max()

    trimmed = trim_dataframe(df, trim_start, trim_end)
    kept_rows = len(trimmed)
    if kept_rows == 0 or kept_rows < MIN_ROWS:
        LOGGER.warning("Skipping %s: too short after trim (t_max=%.3f, rows kept=%s)",
            in_path.name,
            t_max,
            kept_rows,
        )
        return False

    kept_t_min = trimmed["second_elapsed"].min()
    kept_t_max = trimmed["second_elapsed"].max()
    original_rows = len(df)
    removed_rows = original_rows - kept_rows
    LOGGER.info(
        "%s | rows %s -> %s (removed %s) | span %.3fs -> %.3fs",
        in_path.name,
        original_rows,
        kept_rows,
        removed_rows,
        (t_max - t_min),
        (kept_t_max - kept_t_min),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(out_path, index=False)
    return True


def run_trim(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    trim_start: float = 3.0,
    trim_end: float = 4.0,
    pattern: str = "*.csv",
    overwrite: bool = False,
) -> None:
    root = project_root()
    input_dir = input_dir or (root / "Data" / "interim" / "merged")
    output_dir = output_dir or (root / "Data" / "processed" / "trimmed")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = sorted(input_dir.glob(pattern))
    if not files:
        LOGGER.warning("No files found in %s with pattern %s", input_dir, pattern)
        return

    processed = 0
    for path in files:
        out_path = output_dir / path.name
        if process_file(path, out_path, trim_start, trim_end, overwrite):
            processed += 1

    LOGGER.info("Trimmed files written: %s", processed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trim merged recordings by fixed seconds.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to folder with merged CSVs (default: Data/interim/merged).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output folder (default: Data/processed/trimmed).",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=3.0,
        help="Seconds to remove from the start (default: 3.0).",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=4.0,
        help="Seconds to remove from the end (default: 4.0).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input CSVs (default: *.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: off).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    run_trim(
        input_dir=args.input,
        output_dir=args.output,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        pattern=args.pattern,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
