from __future__ import annotations
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)
K_FOLDS = 5
OVERWRITE = True

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
#------------------------------------------------------------------------------------------------------------

def _collect_windows(input_dir: Path, pattern: str = "*.csv") -> Dict[str, Dict[str, List[Path]]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    csv_paths = sorted(input_dir.glob(pattern))

    if not csv_paths:
        LOGGER.warning("No files found in %s with pattern %s", input_dir, pattern)
        return {}

    grouped: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in csv_paths:
        parts = path.stem.split("_", 2)
        if len(parts) < 2:
            LOGGER.warning("Skipping unrecognized filename: %s", path.name)
            continue
        class_label, rec_id = parts[0], parts[1]
        grouped[class_label][rec_id].append(path)

    return {label: dict(rec_map) for label, rec_map in grouped.items()}


def _write_filename_csv(paths: Iterable[Path], out_path: Path) -> None:
    filenames = [path.name for path in paths]
    df = pd.DataFrame({"filename": filenames})
    df.to_csv(out_path, index=False)


def _build_splits(
    grouped: Dict[str, Dict[str, List[Path]]],
    seed: int,
) -> Tuple[List[Path], List[List[Path]], Dict[str, object]]:
    rng = random.Random(seed)
    class_counts = {label: len(rec_map) for label, rec_map in grouped.items()}
    n_min = min(class_counts.values())

    n_test = int(math.floor(n_min/6))
    n_trainval = n_min - n_test
    n_per_fold = n_trainval // K_FOLDS

    test_paths: List[Path] = []
    folds: List[List[Path]] = [[] for _ in range(K_FOLDS)]
    discarded_per_class: Dict[str, int] = {}

    for class_label in sorted(grouped.keys()):
        rec_map = grouped[class_label]
        rec_ids = list(rec_map.keys())
        rng.shuffle(rec_ids)
        rec_ids = rec_ids[:n_min]

        test_cls = rec_ids[:n_test]
        trainval = rec_ids[n_test:]

        for i in range(K_FOLDS):
            start = i * n_per_fold
            end = start + n_per_fold
            fold_ids = trainval[start:end]
            for rec_id in fold_ids:
                folds[i].extend(rec_map[rec_id])

        discarded = len(trainval) - (n_per_fold * K_FOLDS)
        discarded_per_class[class_label] = discarded
        for rec_id in test_cls:
            test_paths.extend(rec_map[rec_id])

        LOGGER.info(
            "Class %s | total_records=%d | used=%d | test=%d | trainval=%d | per_fold=%d | discarded=%d",
            class_label,
            class_counts[class_label],
            n_min,
            n_test,
            len(trainval),
            n_per_fold,
            discarded,
        )

    stats = {
        "n_min": n_min,
        "n_test_records": n_test,
        "n_trainval_records": n_trainval,
        "n_per_fold_records": n_per_fold,
        "discarded_per_class": discarded_per_class,
    }
    return test_paths, folds, stats


def run_split(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    seed: int = 42,
) -> None:
    grouped = _collect_windows(input_dir)
    if not grouped:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    kfold_dir = output_dir / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    targets = [output_dir / "test.csv"] + [
        kfold_dir / f"kf{i}.csv" for i in range(1, K_FOLDS + 1)
    ]
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            existing_list = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output files already exist: {existing_list}. Use overwrite=True to replace."
            )

    test_paths, folds, stats = _build_splits(grouped, seed=seed)

    all_selected = test_paths + [p for fold in folds for p in fold]
    if len(set(all_selected)) != len(all_selected):
        raise ValueError("Split contains duplicate filenames across outputs.")

    LOGGER.info(
        "Split summary | classes=%d | n_min=%d | test_records/class=%d | trainval_records/class=%d | per_fold_records/class=%d | discarded_records/class=%s",
        len(grouped),
        stats["n_min"],
        stats["n_test_records"],
        stats["n_trainval_records"],
        stats["n_per_fold_records"],
        ", ".join(
            f"{label}:{count}"
            for label, count in sorted(stats["discarded_per_class"].items())
        ),
    )

    _write_filename_csv(test_paths, output_dir / "test.csv")
    for i, fold in enumerate(folds, start=1):
        _write_filename_csv(fold, kfold_dir / f"kf{i}.csv")

    LOGGER.info("Wrote split files: test=%d | folds=%s",len(test_paths),", ".join(str(len(fold)) for fold in folds),)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = project_root()
    input_dir = root / "Data" / "processed" / "windows"
    output_dir = root / "Data" / "split"
    overwrite = OVERWRITE
    run_split(input_dir=input_dir, output_dir=output_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
