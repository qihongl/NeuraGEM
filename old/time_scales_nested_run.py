"""Run NeuraGEM on a custom nested-timescales dataset and log latent correlations."""

from __future__ import annotations

import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import plot_style
from functions_and_utils import Logger
import train_and_infer_functions as tif
from time_scales_nested_utils import (
    NestedTimescalesConfig,
    NestedTimescalesDataset,
    patch_nested_dataloaders,
    plot_logger_timeseries,
)

plot_style.set_plot_style()

# ----------------------------------------------------------------------------- #
# Experiment configuration (edit in-place)                                      #
# ----------------------------------------------------------------------------- #

EXPORT_BASE_DIR = Path("./exports/nested_timescales")
RUN_NAME = "nested_timescales_0p1"
SEEDS: Tuple[int, ...] = tuple(range(20))
SKIP_EXISTING = False


# Apply dataset patch so train_model picks up custom dataset
patch_nested_dataloaders()


# ----------------------------------------------------------------------------- #
# Experiment helpers                                                            #
# ----------------------------------------------------------------------------- #

CRITERION = nn.MSELoss(reduction="none")


def run_inference(model, base_config: NestedTimescalesConfig, seed: int):
    inference_config = deepcopy(base_config)
    inference_config.reconfigure_for_prediction(inference_config.experiment_to_run)
    inference_config.dataset_name = "nested_timescales"
    inference_config.no_of_steps_in_weight_space = 0
    inference_config._allow_latent_updates = True  # noqa: SLF001
    inference_config.nested_block_means = base_config.nested_block_means
    inference_config.nested_latent_values = base_config.nested_latent_values
    inference_config.nested_phase = "inference"

    dataset = NestedTimescalesDataset(
        inference_config,
        seed=seed + 10_000,
        total_steps=inference_config.nested_inference_steps,
    )
    dataloader = DataLoader(dataset, batch_size=inference_config.batch_size, shuffle=False)

    model.config = inference_config
    model.eval()

    inference_logger = Logger()
    inference_logger.log_phase("Nested inference")
    inference_logger.config = inference_config
    tif.predictive_learning(inference_logger, inference_config, dataloader, model, CRITERION)
    inference_logger.others["task_latents"] = dataset.latent_matrix
    return inference_logger


def save_result(seed: int, payload: Dict[str, Any]) -> None:
    run_dir = EXPORT_BASE_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"seed-{seed:03d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[saved] {out_path}")


def seed_already_done(seed: int) -> bool:
    run_dir = EXPORT_BASE_DIR / RUN_NAME
    out_path = run_dir / f"seed-{seed:03d}.pkl"
    return out_path.exists()


def run_single_seed(seed: int) -> None:
    if SKIP_EXISTING and seed_already_done(seed):
        print(f"[skip] seed {seed} already completed.")
        return

    config = NestedTimescalesConfig()
    config.env_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_logger, model, trained_config, _ = tif.train_model(
        config,
        seed=seed,
        run_test_phase=False,
    )

    inference_logger = run_inference(model, trained_config, seed)

    figure_dir = EXPORT_BASE_DIR / RUN_NAME
    plot_logger_timeseries(
        train_logger,
        title=f"Seed {seed} – Training logger",
        figure_path=figure_dir / f"seed-{seed:03d}_training.pdf",
        show=False,
    )
    plot_logger_timeseries(
        inference_logger,
        title=f"Seed {seed} – Inference logger",
        figure_path=figure_dir / f"seed-{seed:03d}_inference.pdf",
        show=False,
    )

    payload = {
        "seed": seed,
        "train_logger": train_logger,
        "inference_logger": inference_logger,
        "config": trained_config,
    }
    save_result(seed, payload)


def main() -> None:
    print(f"Running nested timescales experiment '{RUN_NAME}' for {len(SEEDS)} seed(s).")
    for seed in SEEDS:
        run_single_seed(seed)


if __name__ == "__main__":
    main()
