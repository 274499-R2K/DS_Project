from __future__ import annotations

import argparse
import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.3,
    "lr": 0.005,
    "batch_size": 32,
    "epochs": 50,
    "weight_decay": 5e-4,
    "grad_clip": 0.5,
}

CLASS_ID = {"lng": 0, "srt": 1, "sgt": 2, "wlk": 3}
LABEL_COLS = {"class_id", "label", "y", "class"}
WINDOW_EXTS = {".npz", ".npy", ".csv"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_filenames(items: Iterable[str]) -> List[str]:
    filenames = []
    for item in items:
        if not item:
            continue
        name = Path(str(item)).name.strip()
        if name:
            filenames.append(name)
    return filenames


def _infer_label_from_name(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0].lower()
    if prefix in CLASS_ID:
        return CLASS_ID[prefix]
    raise ValueError(f"Cannot infer label from filename: {path.name}")


def _ensure_time_first(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for window features, got shape {x.shape}")
    if x.shape[0] < x.shape[1] and x.shape[0] <= 32:
        x = x.T
    return x


def load_window(path: Path) -> Tuple[torch.Tensor, int]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        key_map = {key.lower(): key for key in data.files}
        x_key = key_map.get("x")
        y_key = key_map.get("y")
        if not x_key:
            raise KeyError(f"No feature array found in {path.name}")
        x = _ensure_time_first(np.array(data[x_key]))
        if y_key:
            y_raw = np.array(data[y_key])
            if y_raw.size > 1:
                y = int(np.argmax(y_raw))
            else:
                y = int(y_raw.reshape(-1)[0])
        else:
            y = _infer_label_from_name(path)
        return torch.tensor(x, dtype=torch.float32), int(y)
    if suffix == ".npy":
        x = _ensure_time_first(np.load(path))
        y = _infer_label_from_name(path)
        return torch.tensor(x, dtype=torch.float32), int(y)
    if suffix == ".csv":
        df = pd.read_csv(path)
        label = None
        for col in df.columns:
            if col.lower() in LABEL_COLS:
                label = int(df[col].iloc[0])
                df = df.drop(columns=[col])
                break
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(f"No numeric feature columns found in {path.name}")
        x = _ensure_time_first(numeric_df.to_numpy())
        if label is None:
            label = _infer_label_from_name(path)
        return torch.tensor(x, dtype=torch.float32), int(label)
    raise ValueError(f"Unsupported window extension: {path.suffix}")


def list_window_files(windows_dir: Path) -> Dict[str, Path]:
    if not windows_dir.exists():
        raise FileNotFoundError(f"Windows directory not found: {windows_dir}")
    files = [p for p in windows_dir.rglob("*") if p.suffix.lower() in WINDOW_EXTS]
    if not files:
        raise FileNotFoundError(f"No window files found in {windows_dir}")
    mapping: Dict[str, Path] = {}
    for path in files:
        if path.name in mapping:
            raise ValueError(f"Duplicate window filename detected: {path.name}")
        mapping[path.name] = path
    return mapping


def _parse_json_manifest(payload: object) -> List[Dict[str, List[str]]]:
    folds: List[Dict[str, List[str]]] = []
    if isinstance(payload, dict):
        if "folds" in payload and isinstance(payload["folds"], list):
            for item in payload["folds"]:
                folds.extend(_parse_json_manifest(item))
            return folds
        if "train" in payload or "val" in payload:
            train = _normalize_filenames(payload.get("train", []))
            val = _normalize_filenames(payload.get("val", []))
            folds.append({"train": train, "val": val})
            return folds
        for value in payload.values():
            if isinstance(value, dict) and ("train" in value or "val" in value):
                folds.extend(_parse_json_manifest(value))
        return folds
    if isinstance(payload, list):
        for item in payload:
            folds.extend(_parse_json_manifest(item))
    return folds


def _parse_csv_manifest(path: Path) -> List[Dict[str, List[str]]]:
    df = pd.read_csv(path)
    cols = {col.lower(): col for col in df.columns}
    folds: List[Dict[str, List[str]]] = []

    if {"fold", "split", "filename"}.issubset(cols):
        for _, fold_df in df.groupby(cols["fold"]):
            split_series = fold_df[cols["split"]].astype(str).str.strip().str.lower()
            train = fold_df[split_series == "train"][cols["filename"]]
            val = fold_df[split_series == "val"][cols["filename"]]
            folds.append({"train": _normalize_filenames(train), "val": _normalize_filenames(val)})
        return folds

    if {"fold", "filename"}.issubset(cols):
        for _, fold_df in df.groupby(cols["fold"]):
            val = fold_df[cols["filename"]]
            folds.append({"train": [], "val": _normalize_filenames(val)})
        return folds

    if "filename" in cols:
        return [{"train": [], "val": _normalize_filenames(df[cols["filename"]])}]

    if "val" in cols or "train" in cols:
        train = _normalize_filenames(df[cols.get("train")] if "train" in cols else [])
        val = _normalize_filenames(df[cols.get("val")] if "val" in cols else [])
        return [{"train": train, "val": val}]

    first_col = df.columns[0]
    return [{"train": [], "val": _normalize_filenames(df[first_col])}]


def _parse_txt_manifest(path: Path) -> List[Dict[str, List[str]]]:
    folds: List[Dict[str, List[str]]] = []
    current = {"train": [], "val": []}
    current_split = "val"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("fold"):
            if current["train"] or current["val"]:
                folds.append(current)
            current = {"train": [], "val": []}
            current_split = "val"
            continue
        if low.startswith("train"):
            current_split = "train"
            if ":" in line:
                line = line.split(":", 1)[1]
            else:
                continue
        elif low.startswith("val"):
            current_split = "val"
            if ":" in line:
                line = line.split(":", 1)[1]
            else:
                continue
        items = [item.strip() for item in line.split(",") if item.strip()]
        current[current_split].extend(_normalize_filenames(items))
    if current["train"] or current["val"]:
        folds.append(current)
    return folds


def load_folds(kfold_dir: Path, all_filenames: Iterable[str]) -> List[Dict[str, List[str]]]:
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold directory not found: {kfold_dir}")
    manifest_files = sorted([p for p in kfold_dir.iterdir() if p.is_file()])
    if not manifest_files:
        raise FileNotFoundError(f"No manifest files found in {kfold_dir}")

    folds: List[Dict[str, List[str]]] = []
    full_definition: Optional[List[Dict[str, List[str]]]] = None

    for path in manifest_files:
        suffix = path.suffix.lower()
        parsed: List[Dict[str, List[str]]] = []
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = _parse_json_manifest(payload)
        elif suffix == ".csv":
            parsed = _parse_csv_manifest(path)
        else:
            parsed = _parse_txt_manifest(path)

        if not parsed:
            continue

        if len(parsed) > 1 and any(fold.get("train") for fold in parsed):
            if full_definition is not None:
                raise ValueError(
                    f"Multiple manifest files define full folds: {full_definition} and {path.name}"
                )
            full_definition = parsed
            continue

        folds.extend(parsed)

    if full_definition is not None:
        folds = full_definition

    if not folds:
        raise ValueError(f"No folds parsed from {kfold_dir}")

    all_set = set(_normalize_filenames(all_filenames))
    for fold in folds:
        fold["train"] = _normalize_filenames(fold.get("train", []))
        fold["val"] = _normalize_filenames(fold.get("val", []))
        if not fold["val"]:
            raise ValueError("Fold manifest missing validation filenames.")
        if not fold["train"]:
            fold["train"] = sorted(list(all_set - set(fold["val"])))
    return folds


class WindowDataset(Dataset):
    def __init__(self, windows: List[torch.Tensor], labels: List[int]) -> None:
        if len(windows) != len(labels):
            raise ValueError("Windows and labels length mismatch.")
        self.windows = windows
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def _pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    if all(x.shape[0] == max_len for x in xs):
        return torch.stack(xs), torch.stack(ys)
    out = torch.zeros(len(xs), max_len, feat_dim, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
    return out, torch.stack(ys)


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head_dropout = nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        last = h_n[-1]
        out = self.head_dropout(last)
        return self.classifier(out)


def _fit_scaler(train_windows: List[torch.Tensor]) -> StandardScaler:
    if not train_windows:
        raise ValueError("No training windows provided for scaler.")
    feat_dim = train_windows[0].shape[1]
    flat = np.concatenate([x.reshape(-1, feat_dim).cpu().numpy() for x in train_windows], axis=0)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def _get_feature_dim(windows: List[torch.Tensor], label: str) -> int:
    if not windows:
        raise ValueError(f"No windows provided for {label}.")
    dims = {x.shape[1] for x in windows}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions in {label}: {sorted(dims)}")
    return dims.pop()


def _apply_scaler(
    windows: List[torch.Tensor], scaler: StandardScaler, feat_dim: int
) -> List[torch.Tensor]:
    scaled: List[torch.Tensor] = []
    for x in windows:
        flat = x.reshape(-1, feat_dim).cpu().numpy()
        flat = scaler.transform(flat)
        scaled.append(torch.tensor(flat.reshape(x.shape), dtype=torch.float32))
    return scaled


def _load_windows(paths: List[Path]) -> Tuple[List[torch.Tensor], List[int]]:
    windows: List[torch.Tensor] = []
    labels: List[int] = []
    for path in paths:
        x, y = load_window(path)
        windows.append(x)
        labels.append(y)
    return windows, labels


def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    metrics = {
        "confusion_matrix": cm,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", labels=[0, 1, 2, 3], zero_division=0)),
    }
    return metrics


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None
) -> Tuple[float, List[int], List[int]]:
    model.eval()
    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, y_true, y_pred


def train_one_fold(
    train_paths: List[Path],
    val_paths: List[Path],
    device: torch.device,
    fold_index: int,
    total_folds: int,
) -> Dict[str, object]:
    if not train_paths or not val_paths:
        raise ValueError("Train/val paths cannot be empty.")
    train_windows, train_labels = _load_windows(train_paths)
    val_windows, val_labels = _load_windows(val_paths)

    feat_dim = _get_feature_dim(train_windows, "training data")
    _ = _get_feature_dim(val_windows, "validation data")
    scaler = _fit_scaler(train_windows)
    train_windows = _apply_scaler(train_windows, scaler, feat_dim)
    val_windows = _apply_scaler(val_windows, scaler, feat_dim)

    train_ds = WindowDataset(train_windows, train_labels)
    val_ds = WindowDataset(val_windows, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        collate_fn=_pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        collate_fn=_pad_collate,
    )

    model = GRUClassifier(
        input_size=feat_dim,
        hidden_size=int(CONFIG["hidden_size"]),
        num_layers=int(CONFIG["num_layers"]),
        dropout=float(CONFIG["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_macro_f1 = -1.0

    print(
        "Fold {}/{} | start | train_samples={} | val_samples={}".format(
            fold_index, total_folds, len(train_ds), len(val_ds)
        ),
        flush=True,
    )

    for epoch in range(int(CONFIG["epochs"])):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["grad_clip"]))
            optimizer.step()
            train_losses.append(loss.item())

        _, y_true, y_pred = _evaluate(model, val_loader, device, criterion)
        metrics = _compute_metrics(y_true, y_pred)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(
            "Fold {}/{} | Epoch {}/{} | train_loss={:.4f} | val_macro_f1={:.4f} | val_acc={:.4f}".format(
                fold_index,
                total_folds,
                epoch + 1,
                int(CONFIG["epochs"]),
                train_loss,
                metrics["macro_f1"],
                metrics["accuracy"],
            )
        , flush=True)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_true, y_pred = _evaluate(model, val_loader, device, criterion=None)
    return _compute_metrics(y_true, y_pred)


def run_train_cv(
    windows_dir: Path,
    kfold_dir: Path,
    device: Optional[str] = None,
) -> Dict[str, object]:
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device_name)

    window_map = list_window_files(windows_dir)
    folds = load_folds(kfold_dir, window_map.keys())

    per_fold = []
    confusion_sum = np.zeros((4, 4), dtype=int)

    for idx, fold in enumerate(folds, start=1):
        train_paths = [window_map[name] for name in fold["train"] if name in window_map]
        val_paths = [window_map[name] for name in fold["val"] if name in window_map]

        missing = [name for name in fold["train"] + fold["val"] if name not in window_map]
        if missing:
            raise FileNotFoundError(f"Missing window files: {missing[:5]} (and more)")

        metrics = train_one_fold(train_paths, val_paths, device_t, idx, len(folds))
        confusion_sum += metrics["confusion_matrix"]
        per_fold.append(
            {
                "fold": idx,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    def _mean_std(key: str) -> Tuple[float, float]:
        values = [fold[key] for fold in per_fold]
        return float(np.mean(values)), float(np.std(values))

    mean_acc, std_acc = _mean_std("accuracy")
    mean_macro, std_macro = _mean_std("macro_f1")
    mean_micro, std_micro = _mean_std("micro_f1")

    return {
        "per_fold": per_fold,
        "mean": {"accuracy": mean_acc, "macro_f1": mean_macro, "micro_f1": mean_micro},
        "std": {"accuracy": std_acc, "macro_f1": std_macro, "micro_f1": std_micro},
        "confusion_matrix": confusion_sum,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train GRU classifier with K-Fold CV.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        set_seeds(args.seed)

    root = project_root()
    windows_dir = root / "Data" / "processed" / "windows"
    kfold_dir = root / "Data" / "split" / "kfold"

    results = run_train_cv(windows_dir, kfold_dir, device=args.device)

    mean = results["mean"]
    std = results["std"]
    print(
        "CV summary | acc={:.4f}+/-{:.4f} | macro_f1={:.4f}+/-{:.4f} | micro_f1={:.4f}+/-{:.4f}".format(
            mean["accuracy"],
            std["accuracy"],
            mean["macro_f1"],
            std["macro_f1"],
            mean["micro_f1"],
            std["micro_f1"],
        )
    )
    print("Confusion matrix (summed across folds):")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()
