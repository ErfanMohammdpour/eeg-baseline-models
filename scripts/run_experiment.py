#!/usr/bin/env python3
"""Run classical EEG baselines from experiment JSON."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.config_io import eeg_electrode_configs, read_json_config
from baseline.features.transforms import (
    Compose,
    ConcatenateWorker,
    FlattenChannel,
    LabelToDict,
    PSD,
    WPTE,
    WTE,
)
from baseline.training.runners import (
    find_inf_idx,
    find_nan_idx,
    fit_eval,
    get_splitter,
    select_dataset,
    select_model,
)


def prepare_transforms(worker_configs):
    transforms = [LabelToDict()]
    worker_transform = []
    for worker in worker_configs:
        name = worker["name"]
        if name == "wte":
            transforms.append(WTE())
            worker_transform.append(WTE())
        elif name == "wpte":
            transforms.append(WPTE())
            worker_transform.append(WPTE())
        elif name == "psd":
            transforms.append(PSD())
            worker_transform.append(PSD())
        else:
            raise ValueError(f"unknown feature worker: {name}")
    transforms.append(FlattenChannel())
    transforms.append(ConcatenateWorker(worker_transform))
    return Compose(transforms)


def load_xy(dataset):
    import numpy as np
    from tqdm import tqdm

    X, Y = [], []
    for i in tqdm(range(len(dataset)), desc="extract"):
        sample, label = dataset[i]
        if isinstance(label, dict) and "concat" in label:
            X.append(label["concat"])
            Y.append(label["label"])
        else:
            X.append(sample)
            Y.append(label)
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)
    mask = ~(find_nan_idx(X) | find_inf_idx(X))
    return X[mask], Y[mask]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment JSON")
    parser.add_argument("--dataset", default="KlinikDataset")
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--channels",
        default=str(ROOT / "configs/eeg_recording_standard/international_10_20_21.py"),
    )
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="baseline_models_eeg")
    args = parser.parse_args()

    exp = read_json_config(args.config)
    if isinstance(exp, dict):
        models = exp.get("models", [])
        transforms = exp.get("transforms", [])
        splitters = exp.get("data_splitters", [{"data_splitter": "train_test_split"}])
    else:
        raise SystemExit("experiment config must be a JSON object")

    positions, _ = eeg_electrode_configs(args.channels)
    os.environ.setdefault("WANDB_MODE", "disabled" if not args.wandb else "online")

    import wandb

    for model_cfg, transform_cfg, splitter_cfg in itertools.product(
        models, transforms, splitters
    ):
        workers = transform_cfg.get("workers", [])
        transform = prepare_transforms(workers)
        opts = {
            "dataset": args.dataset,
            "data_path": args.data_path,
            "eeg_electrode_positions": positions,
            "length": args.length,
            "n_splits": args.n_splits,
            **model_cfg,
            **splitter_cfg,
        }
        dataset = select_dataset(opts, transform=transform)
        X, Y = load_xy(dataset)
        splitter = get_splitter(
            splitter=opts.get("data_splitter", "train_test_split"),
            n_splits=args.n_splits,
        )
        groups = getattr(dataset, "groups", None)
        run = wandb.init(
            project=args.wandb_project,
            config=opts,
            reinit=True,
            mode="online" if args.wandb else "disabled",
        )
        clf = select_model(opts)
        for train_idx, test_idx in splitter(X, Y, groups):
            print(f"Model: {opts.get('model')} | features: {[w['name'] for w in workers]}")
            fit_eval(clf, X, Y, train_idx, test_idx, dataset)
        run.finish()


if __name__ == "__main__":
    main()
