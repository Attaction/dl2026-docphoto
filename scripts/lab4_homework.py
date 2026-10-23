#!/usr/bin/env python3
"""Lab 4 homework 2 and 4: longer runs at 768 px with safe augmentation, and early stopping done honestly.

    uv run --env-file .env python scripts/lab4_homework.py

Trains the three configs below (one W&B run each), then compares every run
with the 8-epoch checkpoints of the lab (`lab4_cnn_768`, no augmentation, and
`lab4_cnn_768_safe`, flip and shift) by the paired page bootstrap on the
validation unit. Writes the per-epoch history and the comparison table to
`models/lab4_homework_*.csv` (not in git, next to the checkpoints).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, evaluate, losses, train_cnn  # noqa: E402
from src.data.dataset import load_table  # noqa: E402
from src.data.inputs import load_inputs  # noqa: E402
from src.data.splits import build_splits  # noqa: E402
from src.models.build import build_model  # noqa: E402

RUNS = ["lab4_cnn_768_safe_e12", "lab4_cnn_768_safe_e16", "lab4_cnn_768_safe_e16_stop"]
REFERENCES = {"8 epochs, none": "lab4_cnn_768", "8 epochs, safe": "lab4_cnn_768_safe"}


def load_cfg(name: str) -> dict:
    with open(config.ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def checkpoint_predictions(name: str, X: np.ndarray, val_idx: np.ndarray, device: str) -> np.ndarray:
    """Validation predictions of a saved checkpoint, with the scaler it was trained with."""
    ckpt = torch.load(config.ROOT / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    loss = losses.from_state(ckpt["loss_state"])
    model = build_model(ckpt["config"], ckpt["in_shape"], loss).to(device)
    model.load_state_dict(ckpt["state_dict"])
    sc = ckpt["scaler"]
    batches = train_cnn.Batches(torch.from_numpy(np.ascontiguousarray(X)), sc, (0.5 - sc["mean"]) / sc["scale"])
    return train_cnn.predict(model, batches, val_idx, loss, device)


def main() -> None:
    matplotlib.use("Agg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_cfg("lab4_cnn_768")
    df = load_table()
    sp = base["split"]
    train_idx, val_idx = build_splits(df, sp["group_column"], sp["val_fraction"], sp["seed"])
    X = load_inputs(base, df)
    vdf = df.iloc[val_idx]
    ok = (vdf["eval_ok"] == 1).to_numpy()
    y, pages = vdf[config.TARGET].to_numpy()[ok], vdf["page_id"].to_numpy()[ok]

    preds = {label: checkpoint_predictions(name, X, val_idx, device)[ok] for label, name in REFERENCES.items()}
    histories = []
    for name in RUNS:
        r = train_cnn.train_model(load_cfg(name), X, df, train_idx, val_idx)
        preds[name] = r["val_pred"][ok]
        histories.append(r["history"].assign(run=name, selected_epoch=r["selected_epoch"], url=r["url"]))

    rows = {}
    for label, p in preds.items():
        row = {"val RMSE": evaluate.rmse(y, p)}
        for ref in REFERENCES:
            bs = evaluate.page_bootstrap(y, preds[ref], p, pages)
            row[f"minus {ref}"], row[f"lo vs {ref}"], row[f"hi vs {ref}"] = bs["diff"], bs["lo"], bs["hi"]
            row[f"better than {ref}"] = bs["share_b_better"]
        rows[label] = row
    table = pd.DataFrame(rows).T
    history = pd.concat(histories, ignore_index=True)

    out = config.ROOT / "models"
    history.to_csv(out / "lab4_homework_history.csv", index=False)
    table.to_csv(out / "lab4_homework_bootstrap.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = [c for c in ["epoch", "train_rmse", "stop_rmse", "val_rmse"] if c in history]
        for name, h in history.groupby("run", sort=False):
            print(f"\n{name} (selected epoch {h['selected_epoch'].iloc[0]}, {h['url'].iloc[0]})")
            print(h[cols].set_index("epoch").round(4).to_string())
        print("\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
