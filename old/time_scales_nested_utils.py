"""Shared helpers for nested timescale experiments."""
from __future__ import annotations

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import plot_style
from configs import ContextualSwitchingTaskConfig

plot_style.set_plot_style()
cs = plot_style.Color_scheme()

# Default experiment parameters (can be overridden via config kwargs)
LATENT_DIMENSIONS = 5
# BLOCK_MEANS = (40, 120, )#50, 70, 90)
BLOCK_MEANS = (20, 50, 80,  110, 140)
LATENT_VALUES = (0.2, 0.8)
TRAINING_TIMESTEPS = 1000
PASSIVE_TIMESTEPS = 300
INFERENCE_TIMESTEPS = 800
Z_slow_lr_scale = 0.4
Z_slow_l2_scale = 0.1

class NestedTimescalesConfig(ContextualSwitchingTaskConfig):
    def __init__(
        self,
        *,
        latent_dimensions: int = LATENT_DIMENSIONS,
        block_means: Sequence[int] = BLOCK_MEANS,
        latent_values: Sequence[float] = LATENT_VALUES,
        training_steps: int = TRAINING_TIMESTEPS,
        passive_steps: int = PASSIVE_TIMESTEPS,
        inference_steps: int = INFERENCE_TIMESTEPS,
    ):
        super().__init__(experiment_to_run="figure")
        self.dataset_name = "nested_timescales"
        self.input_size = latent_dimensions
        self.output_size = latent_dimensions
        self.seq_len = 10
        self.stride = 1
        self.batch_size = 1
        self.epochs = 1
        self.blocked_phase_length = 1200
        self.no_of_blocks = max(1, self.blocked_phase_length // 25)
        self.default_std = 0.1
        self.latent_dims = [4]
        self.latent_chunks = 2
        self.latent_activation = "softmax_chunked"
        self.latent_aggregation_op = "exponential_increase"
        self.exponential_increase_steepness = [0, 0]
        self.exponential_increase_multipliers = [1, 1]
        self.pass_previous_latent = True
        self.LU_optimizer = "SGD"
        
        if self.LU_optimizer == "SGD": 
            self.LU_momentum = 0.
            # self.loss_reduction_LU = "sum"
        self.LU_Adam_betas = (0.6, 0.7)
        self.l2_loss = 0.0001 if self.LU_optimizer == "Adam" else 3e-1
        self.LU_lr = 0.7 if self.LU_optimizer == "Adam" else 1.0

        self.no_of_steps_in_latent_space = 1
        self.WU_lr = 1e-4
        self.hidden_size = 64
        self.add_passive_learning_phase = True
        self.passive_phase_length = passive_steps
        self.start_always_on_the_same_block = True
        self.nested_block_means = tuple(block_means)
        self.nested_latent_values = tuple(latent_values)
        self.nested_training_steps = training_steps
        self.nested_passive_steps = passive_steps
        self.nested_inference_steps = inference_steps
        self.train_on_shuffled_data = False
        self.nested_phase: str | None = None
        self.chunk_LU_lrs = [self.LU_lr, self.LU_lr * Z_slow_lr_scale]
        self.chunk_l2_losses = [self.l2_loss, self.l2_loss * Z_slow_l2_scale]
        self.update_export_path()


class NestedTimescalesDataset(Dataset):
    def __init__(
        self,
        config: NestedTimescalesConfig,
        seed: int | None = None,
        total_steps: int | None = None,
    ):
        self.config = config
        self.seq_len = config.seq_len
        self.stride = config.stride
        default_steps = getattr(config, "nested_training_steps", TRAINING_TIMESTEPS)
        self.total_steps = total_steps if total_steps is not None else default_steps
        self.latent_block_means = config.nested_block_means
        self.latent_value_options = config.nested_latent_values
        self.noise_std = config.default_std
        self.rng = np.random.default_rng(seed if seed is not None else config.env_seed)
        self.inputs, self.latent_matrix = self._generate_sequences()

    def __len__(self) -> int:
        return max(1, (self.total_steps - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_len
        inputs = torch.from_numpy(self.inputs[start:end]).float()
        llcids = torch.from_numpy(self.latent_matrix[start:end]).float()
        hlcids = torch.zeros(llcids.shape[0], 1, dtype=torch.float32)
        return inputs, llcids, hlcids

    def _sample_block_length(self, mean: int) -> int:
        low = max(1, int(round(mean / 2)))
        high = max(low + 1, int(round(mean * 2)))
        while True:
            length = int(self.rng.geometric(1.0 / max(mean, 1)))
            if low <= length <= high:
                return length

    def _generate_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        latents = np.zeros((self.total_steps, len(self.latent_block_means)), dtype=np.float32)
        for dim, mean in enumerate(self.latent_block_means):
            t = 0
            current_val = float(self.rng.choice(self.latent_value_options))
            while t < self.total_steps:
                block_len = min(self._sample_block_length(mean), self.total_steps - t)
                latents[t : t + block_len, dim] = current_val
                current_val = float(self.rng.choice(self.latent_value_options))
                t += block_len
        noise = self.rng.normal(0.0, self.noise_std, size=latents.shape).astype(np.float32)
        inputs = latents + noise
        return inputs, latents


def _select_phase(config: NestedTimescalesConfig) -> str:
    if getattr(config, "nested_phase", None):
        return config.nested_phase  # type: ignore[return-value]
    if not getattr(config, "_allow_latent_updates", True):
        return "passive"
    return "train"


def _phase_total_steps(config: NestedTimescalesConfig, phase: str) -> int:
    if phase == "passive":
        return getattr(config, "nested_passive_steps", PASSIVE_TIMESTEPS)
    if phase == "inference":
        return getattr(config, "nested_inference_steps", INFERENCE_TIMESTEPS)
    return getattr(config, "nested_training_steps", TRAINING_TIMESTEPS)


def build_nested_loaders(config: NestedTimescalesConfig):
    phase = _select_phase(config)
    total_steps = _phase_total_steps(config, phase)
    dataset = NestedTimescalesDataset(config, seed=config.env_seed, total_steps=total_steps)
    dataset_test = NestedTimescalesDataset(
        config,
        seed=config.env_seed + 1000,
        total_steps=total_steps,
    )
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.train_on_shuffled_data)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    return dataset, dataset_test, train_loader, test_loader


def patch_nested_dataloaders():
    import train_and_infer_functions as tif

    original_create = tif.create_datasets_and_loaders

    def _patched(config, pattern=None):
        if getattr(config, "dataset_name", "") == "nested_timescales":
            return build_nested_loaders(config)
        return original_create(config, pattern)

    tif.create_datasets_and_loaders = _patched
    return original_create


def flatten_logger_sequence(seq_list):
    if not seq_list:
        return None
    arr = np.concatenate(seq_list, axis=0)
    if arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], -1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def plot_logger_timeseries(
    logger: Any,
    title: str,
    figure_path: Path | None = None,
    show: bool = True,
) -> None:
    inputs = flatten_logger_sequence(getattr(logger, "inputs", []))
    preds = flatten_logger_sequence(getattr(logger, "predicted_outputs", []))
    latents = flatten_logger_sequence(getattr(logger, "latent_values", []))
    task_latents = logger.others.get("task_latents")
    if task_latents is not None and task_latents.ndim == 1:
        task_latents = task_latents[:, None]
    if task_latents is None:
        ll = flatten_logger_sequence(getattr(logger, "llcids", []))
        task_latents = ll

    n_panels = 4
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(cs.panel_large_size[0], cs.panel_large_size[1] * 1.5),
        sharex=True,
        dpi=200,
    )
    axes = np.atleast_1d(axes)

    if inputs is not None:
        im = axes[0].imshow(inputs.T, aspect="auto", cmap="viridis", interpolation="none")
        axes[0].set_ylabel("Input dim")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    else:
        axes[0].text(0.5, 0.5, "No inputs logged", ha="center")

    if preds is not None:
        axes[1].plot(preds, alpha=0.7, linewidth=0.5)
        axes[1].set_ylabel("Pred")
    else:
        axes[1].text(0.5, 0.5, "No predictions", ha="center")

    # latent_plot_type = "imshow"  # "line"
    latent_plot_type = "line"
    if latent_plot_type == "line" and latents is not None:
        axes[2].plot(latents, alpha=0.7, linewidth=0.5)
        axes[2].set_ylabel("Z units")
    elif latent_plot_type == "imshow" and latents is not None:
        im2 = axes[2].imshow(latents.T, aspect="auto", cmap="coolwarm", interpolation="none")
        axes[2].set_ylabel("Z units")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    else:
        axes[2].text(0.5, 0.5, "No latents", ha="center")

    if task_latents is not None:
        im3 = axes[3].imshow(task_latents.T, aspect="auto", cmap="magma", interpolation="none")
        axes[3].set_ylabel("Task latent")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.02)
    else:
        axes[3].text(0.5, 0.5, "No task latent", ha="center")

    axes[-1].set_xlabel("Time steps")
    fig.suptitle(title)
    fig.tight_layout()

    if figure_path:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, bbox_inches="tight")
        print(f"[figure] saved to {figure_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
