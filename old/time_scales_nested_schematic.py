"""Generate a schematic plot of nested-timescale inputs."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from time_scales_nested_utils import NestedTimescalesConfig, NestedTimescalesDataset

plot_style.set_plot_style()
cs = plot_style.Color_scheme()


def _select_window_for_dim(series: np.ndarray, window: int, target_switches: int) -> tuple[int, int]:
    """Pick a window that hits the target switch count for this dimension as closely as possible."""
    best_start, best_end = 0, window
    best_diff = float("inf")
    for start in range(0, len(series) - window + 1):
        end = start + window
        segment = series[start:end]
        switches = int((np.diff(segment) != 0).sum())
        diff = abs(switches - target_switches)
        if diff < best_diff:
            best_diff = diff
            best_start, best_end = start, end
            if best_diff == 0:
                break
    return best_start, best_end


def make_schematic(
    total_steps: int = 600,
    seed: int = 0,
    figure_path: Path | None = None,
    target_switches: tuple[int, ...] = (12, 8, 5, 3, 1),
) -> None:
    """Sample a dataset and plot stacked input streams to illustrate block switches."""
    config = NestedTimescalesConfig()
    # Generate a longer run, then pick a window per dimension to match desired switch counts.
    long_steps = max(total_steps * 3, total_steps + 400)
    dataset = NestedTimescalesDataset(config, seed=seed, total_steps=long_steps)

    n_dims = dataset.inputs.shape[1]
    targets = list(target_switches) + [target_switches[-1]] * max(0, n_dims - len(target_switches))

    inputs_segments = []
    for dim in range(n_dims):
        start, end = _select_window_for_dim(dataset.latent_matrix[:, dim], total_steps, targets[dim])
        inputs_segments.append(dataset.inputs[start:end, dim])

    t = np.arange(total_steps)

    fig, axes = plt.subplots(
        n_dims,
        1,
        figsize=(cs.panel_large_size[0] * .9, cs.panel_large_size[1] * 0.7),
        sharex=True,
        dpi=400,
    )

    cmap = plt.get_cmap("viridis")

    # Plot each input dimension as semi-transparent lines with color gradation.
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(
            t,
            inputs_segments[i],
            color=cmap(i / max(n_dims - 1, 1)),
            alpha=0.6,
            linewidth=.8,
            rasterized=True,
        )
        ax.set_yticks([])
        ax.set_ylabel(fr"$X^{{{i + 1}}}$", rotation=0, labelpad=2, ha="right", va="center")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0, labelbottom=False)

    # Only the bottom axis shows a minimal x-axis.
    axes = np.atleast_1d(axes)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].tick_params(axis="x", length=3, labelbottom=True)
    axes[-1].set_xlabel("Time steps")
    axes[-1].set_xlim(t[0], t[-1])

    fig.tight_layout(h_pad=0.026)

    if figure_path:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, bbox_inches="tight")
        print(f"[figure] saved to ./{figure_path}")

    plt.show()


def main():
    EXPORT_BASE_DIR = Path("./exports/contextual_switching_task/time_scales")
    RUN_NAME = "nested_timescales_0p1"
    figure_path = EXPORT_BASE_DIR / RUN_NAME / "nested_timescales_schematic.pdf"
    make_schematic(figure_path=figure_path)


if __name__ == "__main__":
    main()
