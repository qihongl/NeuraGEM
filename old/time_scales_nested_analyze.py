"""Analyze nested timescale experiments and plot Z-vs-task latent correlations."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import plot_style
from time_scales_nested_utils import (
    NestedTimescalesConfig,
    plot_logger_timeseries,
    flatten_logger_sequence,
)
import sys

if not hasattr(sys.modules.get("__main__"), "NestedTimescalesConfig"):
    sys.modules["__main__"].NestedTimescalesConfig = NestedTimescalesConfig  # type: ignore[attr-defined]

plot_style.set_plot_style()
cs = plot_style.Color_scheme()

EXPORT_BASE_DIR = Path("./exports/nested_timescales")
# RUN_NAME = "nested_timescales_v1"
# RUN_NAME = "nested_timescales-0p2/"
RUN_NAME = "nested_timescales_0p1/"

FIGURE_PATH: Path | None = Path(EXPORT_BASE_DIR / RUN_NAME / "z_chunk_correlations.pdf")
SHOW_FIGURE = True  # Set to True for interactive viewing
NULL_SHIFT_COUNT = 200


def load_records() -> List[Dict]:
    run_dir = EXPORT_BASE_DIR / RUN_NAME
    if not run_dir.exists():
        raise FileNotFoundError(f"No exports found at {run_dir}")
    records = []
    for result_file in sorted(run_dir.glob("seed-*.pkl")):
        with open(result_file, "rb") as f:
            records.append(pickle.load(f))
    if not records:
        raise RuntimeError(f"No result files found at {run_dir}")
    return records


def compute_correlations_with_null(logger, n_shifts: int = NULL_SHIFT_COUNT, rng: np.random.Generator | None = None):
    latent_values = np.concatenate(logger.latent_values, axis=0)
    latent_values = latent_values.reshape(latent_values.shape[0], -1)

    seq_len = getattr(logger.config, "seq_len", 0)
    stride = getattr(logger.config, "stride", 1)
    drop = max(0, seq_len - stride)
    task_latents = np.asarray(logger.others.get("task_latents"))[drop:]
    if task_latents.ndim == 1:
        task_latents = task_latents[:, None]

    min_len = min(latent_values.shape[0], task_latents.shape[0])
    latent_values = latent_values[-min_len:]
    task_latents = task_latents[-min_len:]

    chunks = getattr(logger.config, "latent_chunks", 1) or 1
    chunk_size = max(1, latent_values.shape[1] // chunks)

    chunk_series = []
    for chunk in range(chunks):
        start = chunk * chunk_size
        end = min(latent_values.shape[1], (chunk + 1) * chunk_size)
        chunk_values = latent_values[:, start:end]
        if chunk_values.size == 0:
            continue
        chunk_series.append(chunk_values[:, 0])

    n_chunks = len(chunk_series)
    n_dims = task_latents.shape[1]
    real_abs = np.full((n_chunks, n_dims), np.nan)
    null_mean = np.full_like(real_abs, np.nan)
    null_std = np.full_like(real_abs, np.nan)

    if rng is None:
        seed = getattr(logger.config, "env_seed", 0)
        rng = np.random.default_rng(seed + 98765)

    for ci, series in enumerate(chunk_series):
        series = np.asarray(series, float)
        series_std = series.std()
        if series_std < 1e-8:
            continue
        T = len(series)
        for di in range(n_dims):
            target = np.asarray(task_latents[:, di], float)
            target_std = target.std()
            if target_std < 1e-8:
                continue

            r_real = np.corrcoef(series, target)[0, 1]
            real_abs[ci, di] = abs(r_real)

            null_vals = []
            for _ in range(n_shifts):
                shift = int(rng.integers(1, max(2, T)))
                shifted = np.roll(series, shift)
                r_null = np.corrcoef(shifted, target)[0, 1]
                null_vals.append(abs(r_null))
            null_vals = np.asarray(null_vals, float)
            null_mean[ci, di] = null_vals.mean()
            null_std[ci, di] = null_vals.std(ddof=1) + 1e-8

    # Timescale compensation: downweight latents with longer average blocks
    block_means = getattr(logger.config, "nested_block_means", None)
    if block_means is not None:
        block_means = np.asarray(block_means, dtype=float)
        if block_means.ndim == 0:
            block_means = np.array([block_means])
        min_block = block_means.min() if block_means.size else 1.0
        timescale_weights = (min_block / np.maximum(block_means, 1e-6)) ** 0.5
        real_abs = real_abs * timescale_weights
        null_mean = null_mean * timescale_weights
        null_std = null_std * timescale_weights

    adjusted = real_abs - null_mean
    z_scores = adjusted / null_std
    return {
        "real_abs": real_abs,
        "adjusted": adjusted,
        "z_scores": z_scores,
        "null_mean": null_mean,
        "null_std": null_std,
    }


def aggregate_correlations(records: List[Dict], key: str = "z_scores") -> np.ndarray:
    corr_list = []
    for record in records:
        logger = record["inference_logger"]
        stats = compute_correlations_with_null(logger)
        corr_list.append(stats[key])
    return np.stack(corr_list, axis=0)




def plot_correlations(corr_matrix: np.ndarray) -> None:
        mean = np.nanmean(corr_matrix, axis=0)
        counts = np.sum(~np.isnan(corr_matrix), axis=0)
        sem = np.nanstd(corr_matrix, axis=0, ddof=1) / np.sqrt(np.maximum(counts, 1))

        chunks, task_dims = mean.shape
        x_positions = np.arange(task_dims)

        fig, axes = plt.subplots(chunks, 1, figsize=cs.panel_large_size, sharex=True, dpi = 150, sharey=True)

        if chunks == 1:
            axes = [axes]

        bar_color = ["tab:red", "tab:blue"]
        z_labels = [r"$Z_{\mathrm{fast}}$", r"$Z_{\mathrm{slow}}$"]

        for idx, ax in enumerate(axes):
            ax.bar(
                x_positions,
                mean[idx],
                yerr=sem[idx],
                capsize=3,
                color=bar_color[idx],
                alpha=0.8,
            )
            
            # Plot individual data points
            if plot_individual_points := False:
                for x_pos in range(task_dims):
                    y_values = corr_matrix[:, idx, x_pos]
                    valid_y = y_values[~np.isnan(y_values)]
                    if len(valid_y) > 0:
                        x_jittered = x_pos + np.random.uniform(-0.15, 0.15, size=len(valid_y))
                        ax.scatter(
                            x_jittered,
                            valid_y,
                            color='grey',
                            alpha=0.6,
                            s=10,
                            zorder=2
                        )
            
            z_label = z_labels[idx] if idx < len(z_labels) else f"Z{idx + 1}"
            ax.set_ylabel(f"{z_label}\n Adjusted |corr|")
            ax.set_ylim(bottom=0)
            ax.axhline(0, color="black", linewidth=0.5)
        block_means = (20, 60,  80, 120, 160,) # Coppied from config for labeling
        axes[-1].set_xlabel("Task latents block size")
        axes[-1].set_xticks(x_positions, [f"{int(block_means[i])}" for i in range(task_dims)])

        if FIGURE_PATH:
            fig.savefig(FIGURE_PATH, bbox_inches="tight")
            print(f"[figure] saved to {'./' + str(FIGURE_PATH)}")
        if SHOW_FIGURE:
            plt.show()
        else:
            plt.close(fig)



def plot_z1_vs_z2_correlations(corr_matrix: np.ndarray) -> None:
    """Plot scatter plots showing relationship between Z1 and Z2 correlations for each latent."""
    chunks, task_dims = corr_matrix.shape[1:]
    
    if chunks < 2:
        print("Need at least 2 Z chunks to compare Z1 vs Z2 correlations")
        return
    
    # Determine subplot layout
    n_cols = min(3, task_dims)
    n_rows = (task_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3),
                              dpi=300, sharey=True)
    if task_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten() if task_dims > 1 else axes
    
    for latent_idx in range(task_dims):
        ax = axes[latent_idx]
        
        # Extract Z1 and Z2 correlations for this latent across all seeds
        z1_corrs = corr_matrix[:, 0, latent_idx]  # All seeds, Z1, this latent
        z2_corrs = corr_matrix[:, 1, latent_idx]  # All seeds, Z2, this latent
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(z1_corrs) | np.isnan(z2_corrs))
        z1_valid = z1_corrs[valid_mask]
        z2_valid = z2_corrs[valid_mask]
        
        if len(z1_valid) > 0:
            # Plot scatter
            ax.scatter(z1_valid, z2_valid, color='grey', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Draw diagonal line (y=x)
            all_vals = np.concatenate([z1_valid, z2_valid])
            lim_min, lim_max = all_vals.min(), all_vals.max()
            margin = (lim_max - lim_min) * 0.1
            lim_min -= margin
            lim_max += margin
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='y=x')
            
            # Calculate and display correlation
            if len(z1_valid) > 1:
                corr_coef = np.corrcoef(z1_valid, z2_valid)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
        
        ax.set_xlabel('Z1 |corr|')
        ax.set_ylabel('Z2 |corr|')
        ax.set_title(f'Latent {latent_idx + 1}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(task_dims, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Z1 vs Z2 Correlation Comparison ({RUN_NAME})', fontsize=12)
    fig.tight_layout()
    
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def compute_per_dimension_mse(logger):
    """
    Compute MSE for each observation dimension separately.
    
    Returns:
        mse_per_dim: array of shape (n_dimensions,) with MSE for each dimension
    """
    # Concatenate inputs (ground truth observations/targets)
    inputs = np.concatenate(logger.inputs, axis=0)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    
    # Concatenate predicted outputs
    if not logger.predicted_outputs:
        return None
    
    predicted_outputs = np.concatenate(logger.predicted_outputs, axis=0)
    predicted_outputs = predicted_outputs.reshape(-1, predicted_outputs.shape[-1])
    
    # Ensure same length
    min_len = min(inputs.shape[0], predicted_outputs.shape[0])
    inputs = inputs[-min_len:]
    predicted_outputs = predicted_outputs[-min_len:]
    
    # Compute MSE per dimension
    mse_per_dim = ((predicted_outputs - inputs) ** 2).mean(axis=0)
    
    return mse_per_dim


def _sig_code(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "†"  # marginal
    return "ns"


def _format_compact_stats(*, z_name: str, timescale_label: str, focus_dims: List[int], block_means, r: float, p: float, n: int) -> str:
    """Return a compact multi-line summary suitable for console logs + paper copying."""
    # Pretty label for copying into LaTeX.
    z_label_tex = {
        "Z_fast": r"Z_{\text{fast}}",
        "Z_slow": r"Z_{\text{slow}}",
    }.get(z_name, z_name)

    bs = [block_means[i] for i in focus_dims if i < len(block_means)]
    df = n - 2
    r2 = r ** 2
    direction = "negative" if r < 0 else "positive"
    sig = _sig_code(p)
    dim_str = ",".join(map(str, focus_dims))
    bs_str = ",".join(str(int(x)) for x in bs) if len(bs) else "?"

    # One-liner that’s easy to paste into text.
    paper = (
        f"{z_label_tex}\\;\\text{{vs}}\\;\\mathrm{{MSE}}\\;"
        f"({timescale_label.lower()} dims [{dim_str}] / blocks [{bs_str}]): "
        f"r={r:.3f}, p={p:.4f}, df={df}, n={n}, R^2={r2:.3f} ({sig}, {direction})"
    )

    # Slightly richer block for console.
    return "\n".join(
        [
            f"{z_name} | {timescale_label} | dims [{dim_str}] blocks [{bs_str}]",
            f"r={r:.4f}, p={p:.6f}, df={df}, n={n}, R^2={r2:.4f}, sig={sig} ({direction})",
            f"paper: {paper}",
        ]
    )


def _print_compact_block(title: str, body: str) -> None:
    line = "=" * 72
    print("\n" + line)
    print(title)
    print(line)
    print(body)


def _pearson_r_p(x, y) -> tuple[float, float]:
    """Compatibility helper: returns (r, p) as floats across SciPy versions.

    We intentionally treat the SciPy return type as dynamic here to avoid
    version-dependent typing friction.
    """
    from typing import Any

    res: Any = pearsonr(x, y)
    # New SciPy: object with attributes (statistic, pvalue)
    if hasattr(res, "statistic") and hasattr(res, "pvalue"):
        return float(res.statistic), float(res.pvalue)
    # Old SciPy: plain tuple
    return float(res[0]), float(res[1])


def analyze_z_correlation_vs_mse(
    records: List[Dict],
    z_chunk_idx: int,
    focus_dims: List[int],
    z_name: str = "Z",
    *,
    ax: Axes | None = None,
    save_fig: bool = True,
) -> None:
    """
    Analyze relationship between Z-latent correlation and prediction MSE.
    
    Args:
        records: List of experimental records
        z_chunk_idx: Which Z chunk to analyze (0 for Z_fast, 1 for Z_slow)
        focus_dims: Dimensions to analyze (e.g., [0, 1] for fastest timescales)
        z_name: Display name for the Z variable (e.g., "Z_fast", "Z_slow")
    """
    # Keep console output compact: one summary block + optional per-dimension details.
    # Flip the flag below if you want the extra breakdown printed.
    timescale_label = "Fast" if z_chunk_idx == 0 else "Slow"
    show_per_dimension = False
    
    # Collect data
    data_points = []
    
    for record in records:
        logger = record["inference_logger"]
        seed = record.get("seed", "unknown")
        
        # Compute correlations
        corr_stats = compute_correlations_with_null(logger)
        adjusted = corr_stats["adjusted"]  # shape: (n_chunks, n_dims)
        
        # Check if this Z chunk exists
        if z_chunk_idx >= adjusted.shape[0]:
            continue
        
        # Compute MSE per dimension
        mse_per_dim = compute_per_dimension_mse(logger)
        if mse_per_dim is None:
            continue
        
        # For each focus dimension, record Z correlation and MSE
        for dim_idx in focus_dims:
            if dim_idx >= adjusted.shape[1]:
                continue
            
            z_corr = adjusted[z_chunk_idx, dim_idx]  # This Z's correlation with this latent
            
            data_points.append({
                'seed': seed,
                'dim': dim_idx,
                'z_corr': z_corr,
                'mse': mse_per_dim[dim_idx],
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        print("No valid data points!")
        return
    
    # Statistical analysis
    r, p = _pearson_r_p(df_clean["z_corr"], df_clean["mse"])
    n = len(df_clean)
    
    # Get config info
    first_logger = records[0]["inference_logger"]
    block_means = getattr(first_logger.config, "nested_block_means", (20, 50, 80, 110, 140))
    
    # Compact output block
    title = f"{z_name} correlation vs MSE (Z chunk {z_chunk_idx})"
    body = _format_compact_stats(
        z_name=z_name,
        timescale_label=timescale_label,
        focus_dims=focus_dims,
        block_means=block_means,
        r=r,
        p=p,
        n=n,
    )
    _print_compact_block(title, body)

    if show_per_dimension:
        print("\nPer-dimension (optional):")
    
        for dim_idx in focus_dims:
            df_dim = df_clean[df_clean['dim'] == dim_idx]
            if len(df_dim) < 3:
                continue

            r_dim, p_dim = pearsonr(df_dim['z_corr'], df_dim['mse'])
            block_size = block_means[dim_idx] if dim_idx < len(block_means) else "?"
            print(f"  dim {dim_idx} (block {block_size}): r={r_dim:.3f}, p={p_dim:.4f}, n={len(df_dim)}, df={len(df_dim)-2}")
    
    # Create publication-quality figure (or draw into provided axes)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(cs.panel_small_size), dpi=300)
    assert ax is not None
    
    # Color by dimension
    colors = ['grey']
    for i, dim_idx in enumerate(focus_dims):
        df_dim = df_clean[df_clean['dim'] == dim_idx]
        block_size = block_means[dim_idx] if dim_idx < len(block_means) else "?"
        ax.scatter(df_dim['z_corr'], df_dim['mse'], 
                  c=colors[0], alpha=0.6, s=10, 
                  edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df_clean['z_corr'], df_clean['mse'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df_clean['z_corr'].min(), df_clean['z_corr'].max(), 100)
    ax.plot(x_line, p_line(x_line), "k--", alpha=0.8, linewidth=1, 
            label=f'r = {r:.3f}, p = {p:.4f}')
    
    # Use mathtext for subscripted Z labels.
    z_label_plot = {
        "Z_fast": r"$Z_{\mathrm{fast}}$",
        "Z_slow": r"$Z_{\mathrm{slow}}$",
    }.get(z_name, z_name)
    ax.set_xlabel(f"{z_label_plot} corr with {'fast' if focus_dims[0] == 0 else 'slow'} latents")
    ax.set_ylabel(f'MSE on {", ".join([f"$X^{{{fd+1}}}$" for fd in focus_dims])} dims')
    # ax.legend(loc='best', framealpha=0.9, fontsize=6)
    
    # Save figure (only when running standalone)
    if ax is not None:
        # Subplot mode: saving handled by the caller.
        save_fig = False

    if save_fig:
        z_label_clean = z_name.lower().replace('_', '')
        fig_path = EXPORT_BASE_DIR / RUN_NAME / f"{z_label_clean}_{'fast' if focus_dims[0] == 0 else 'slow'}_correlation_vs_mse.pdf"
        ax.figure.savefig(fig_path, bbox_inches="tight", transparent=True)  # type: ignore[attr-defined]
        print(f"\n[figure] saved to ./{fig_path}")

        if SHOW_FIGURE:
            plt.show()
        else:
            plt.close(ax.figure)  # type: ignore[arg-type]

    return None


def plot_all_z_correlation_vs_mse_subplots(
    records: List[Dict],
    *,
    fastest_dims: List[int],
    slowest_dims: List[int],
    fig_path: Path | None = None,
) -> None:
    """Create a 2x2 panel figure with the four correlation-vs-MSE analyses.

    We re-use the existing per-panel styling (cs.panel_small_size) and simply
    place the four panels into a single figure.
    """
    # Place panels in a 2x2 grid. Keep each panel size roughly equal to
    # the previous standalone figure size.
    panel_w, panel_h = cs.panel_small_size
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(panel_w * 2.0, panel_h * 2.0),
        dpi=300,
        constrained_layout=True,
    )

    # Top-left: Z_fast vs fast dims
    analyze_z_correlation_vs_mse(
        records,
        z_chunk_idx=0,
        focus_dims=fastest_dims,
        z_name="Z_fast",
        ax=axes[0, 0],
        save_fig=False,
    )
    # axes[0, 0].set_title(r"$Z_{\mathrm{fast}}$ vs fast latents", fontsize=8)

    # Top-right: Z_slow vs slow dims
    analyze_z_correlation_vs_mse(
        records,
        z_chunk_idx=1,
        focus_dims=slowest_dims,
        z_name="Z_slow",
        ax=axes[0, 1],
        save_fig=False,
    )
    # axes[0, 1].set_title(r"$Z_{\mathrm{slow}}$ vs slow latents", fontsize=8)

    # Bottom-left: Z_fast vs slow dims (control)
    analyze_z_correlation_vs_mse(
        records,
        z_chunk_idx=0,
        focus_dims=slowest_dims,
        z_name="Z_fast",
        ax=axes[1, 0],
        save_fig=False,
    )
    # axes[1, 0].set_title(r"$Z_{\mathrm{fast}}$ vs slow latents (control)", fontsize=8)

    # Bottom-right: Z_slow vs fast dims (control)
    analyze_z_correlation_vs_mse(
        records,
        z_chunk_idx=1,
        focus_dims=fastest_dims,
        z_name="Z_slow",
        ax=axes[1, 1],
        save_fig=False,
    )
    # axes[1, 1].set_title(r"$Z_{\mathrm{slow}}$ vs fast latents (control)", fontsize=8)

    if fig_path is None:
        fig_path = EXPORT_BASE_DIR / RUN_NAME / "z_correlation_vs_mse_panels.pdf"

    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    print(f"\n[figure] saved to ./{fig_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    records = load_records()
    print('no of records: ', len(records))
    key = "adjusted"
    
    # Generate correlation bar plots
    corr_matrix = aggregate_correlations(records, key=key)
    plot_correlations(corr_matrix)
    plot_z1_vs_z2_correlations(corr_matrix)

    mean = np.nanmean(corr_matrix, axis=0)
    print("Mean correlations per Z chunk (rows) vs task latent (cols):")
    np.set_printoptions(precision=3, suppress=True, nanstr="nan")
    print(mean)
    
    # Determine number of latent dimensions
    first_logger = records[0]["inference_logger"]
    n_dims = mean.shape[1] if len(mean.shape) > 1 else 1
    
    # Correlation-vs-MSE panels
    slowest_dims = [n_dims - 2, n_dims - 1]  # Last two dimensions
    fastest_dims = [0, 1]  # First two dimensions
    plot_all_z_correlation_vs_mse_subplots(
        records,
        fastest_dims=fastest_dims,
        slowest_dims=slowest_dims,
    )
    



if __name__ == "__main__":
    main()
