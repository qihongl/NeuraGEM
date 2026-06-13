"""A version of Model comparison analysis 
block_size now controllable as to which segment in the block
it aggregates error from. 

"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from configs import ContextualSwitchingTaskConfig
from functions_and_utils_2 import calculate_error
from Bayesian_obs_generalization import run_bayesian_generalization

plot_style.set_plot_style()
COLOR_SCHEME = plot_style.Color_scheme()
mpl.rcParams.update({"figure.dpi": 300})

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None

try:  # pragma: no cover - optional dependency
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except Exception:  # pragma: no cover - environment without mpl_toolkits
    inset_axes = None
import sys


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


DEFAULT_EXPORT_ROOT = Path("./exports/contextual_switching_task/experiments")


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing one model variant in the comparison plots."""

    artifact_name: str
    label: str
    color: str
    base_filters: Mapping[str, Any]


MODEL_INFO: Dict[str, ModelInfo] = {
    "rnn_seq_len_5": ModelInfo(
        artifact_name="rnn",
        label=r"RNN$^{\mathrm{short}}$",
        color=COLOR_SCHEME.short_horizon_rnn,
        base_filters={"seq_len": 5, "WU_lr": 1e-3},
    ),
    "rnn_seq_len_50": ModelInfo(
        artifact_name="rnn",
        label=r"RNN$^{\mathrm{long}}$",
        color=COLOR_SCHEME.long_horizon_rnn,
        base_filters={"seq_len": 50, "WU_lr": 1e-3},
    ),
    "neuragem": ModelInfo(
        artifact_name="neuragem",
        label="NeuraGEM",
        color=COLOR_SCHEME.neuragem,
        base_filters={"WU_lr": 1e-4},
    ),
    "bayesian": ModelInfo(
        artifact_name="bayesian_observer",
        label="Bayesian",
        color=COLOR_SCHEME.bayesian,
        base_filters={},
    ),
}

MODEL_ORDER: Tuple[str, ...] = tuple(MODEL_INFO.keys())


@dataclass(frozen=True)
class ComparisonSpec:
    """Parameter overrides for a single comparison within a run."""

    key: str
    model_overrides: Mapping[str, Mapping[str, Any]]
    title: str | None = None


@dataclass(frozen=True)
class RunSpec:
    """Definition of one experiment suite to analyse."""

    run_name: str
    ood_test_type: str
    comparisons: Sequence[ComparisonSpec]
    iid_reference_values: Sequence[float]
    weights_frozen: bool = True
    max_seeds: int | None = None


RUN_SPECS: Dict[str, RunSpec] = {
    "new_runs_stds": RunSpec(
        run_name="new_runs_stds",
        ood_test_type="ood_stds",
        iid_reference_values=(0.3,),
        comparisons=(
            ComparisonSpec(
                key="default_std_0p3",
                title="Observation noise sweep",
                model_overrides={
                    "rnn_seq_len_5": {"default_std": 0.3},
                    "rnn_seq_len_50": {"default_std": 0.3},
                    "neuragem": {"default_std": 0.3, "l2_loss": 0.0001},
                    "bayesian": {"default_std": 0.3, "observation_noise_mode": {"default": "fixed", "ood_stds": "fixed"}},
                },
            ),
        ),
    ),
    "new_runs_means": RunSpec(
        # run_name="new_runs_means",
        run_name="new_runs_50_means",
        ood_test_type="ood_means",
        iid_reference_values=(0.2, 0.8),
        comparisons=(
            ComparisonSpec(
                key="default_std_0p3",
                title="OOD means (std 0.3)",
                model_overrides={
                    "rnn_seq_len_5": {"default_std": 0.3},
                    "rnn_seq_len_50": {"default_std": 0.3},
                    # "neuragem": {"default_std": 0.3, "l2_loss": 0.0008},
                    "bayesian": {"default_std": 0.3},
                },
            ),
            ComparisonSpec(
                key="default_std_0p4",
                title="OOD means (std 0.4)",
                model_overrides={
                    "rnn_seq_len_5": {"default_std": 0.4},
                    "rnn_seq_len_50": {"default_std": 0.4},
                    "neuragem": {"default_std": 0.4, "l2_loss": 0.0008},
                    "bayesian": {"default_std": 0.4},
                },
            ),
        ),
    ),
    "new_runs_block_size": RunSpec(
        run_name="new_runs_50_block_size",
        ood_test_type="block_size",
        iid_reference_values=(25,),
        comparisons=(
            # ComparisonSpec(
            #     key="default_std_0p3",
            #     title="Block-size sweep (std 0.3)",
            #     model_overrides={
            #         "rnn_seq_len_5": {"default_std": 0.3},
            #         "rnn_seq_len_50": {"default_std": 0.3},
            #         "neuragem": {"default_std": 0.3, "l2_loss": 0.0001},
            #         "bayesian": {"default_std": 0.3},
            #     },
            # ),
            ComparisonSpec(
                key="default_std_0p4",
                title="Block-size sweep (std 0.4)",
                model_overrides={
                    "rnn_seq_len_5": {"default_std": 0.4},
                    "rnn_seq_len_50": {"default_std": 0.4},
                    "neuragem": {"default_std": 0.4, "l2_loss": 0.0001},
                    "bayesian": {"default_std": 0.4},
                },
            ),
        ),
    ),
}
# temp remove other runs for debugging
RUN_SPECS.pop('new_runs_stds')
RUN_SPECS.pop('new_runs_means')
# RUN_SPECS.pop('new_runs_block_size')
@dataclass
class AnalysisParams:
    """Global knobs for the analysis pipeline."""

    error_type: str = "abs_from_mean"
    pre_window: int = 0
    post_window: int = 10
    # block_size_window_mode: str = "full_block"  # options: full_block, peri_switch, mid_block
    # block_size_window_mode: str = "peri_switch"  # options: full_block, peri_switch, mid_block
    block_size_window_mode: str = "mid_block"  # options: full_block, peri_switch, mid_block
    block_size_peri_post_window: int = 5  # post-window for peri-switch mode
    block_size_midpoint: int = 40  # center (in steps after switch) for mid-block window
    block_size_mid_halfwidth: int = 5  # captures [midpoint-halfwidth, midpoint+halfwidth)
    statistical_analysis: bool = True
    boot_iterations: int = 1000
    show_plots: bool = True


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _combination_key(params: Mapping[str, Any]) -> str:
    return "_".join(
        f"{key}-{_format_param_value(params[key])}"
        for key in sorted(params)
    )


def _result_pattern(
    run_spec: RunSpec,
    model_key: str,
    filters: Mapping[str, Any],
    export_root: Path,
) -> Tuple[Path, str]:
    combo_key = _combination_key(filters)
    folder = export_root / run_spec.run_name / combo_key
    pattern = f"results_{MODEL_INFO[model_key].artifact_name}_frozen_{run_spec.weights_frozen}_{combo_key}_seed-*.pkl"
    return folder, pattern


def _parse_seed_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("seed-")[-1])
    except ValueError:
        return -1


def _load_payloads(
    run_spec: RunSpec,
    model_key: str,
    filters: Mapping[str, Any],
    export_root: Path,
    max_seeds: int | None,
) -> List[Dict[str, Any]]:
    folder, pattern = _result_pattern(run_spec, model_key, filters, export_root)
    if not folder.exists():
        print(f"  Missing folder for {model_key}: {folder}")
        return []

    paths = sorted(folder.glob(pattern), key=_parse_seed_from_path)
    if not paths:
        print(f"  No result files matched pattern {pattern} in {folder}")
        return []
    if max_seeds is not None:
        paths = paths[:max_seeds]

    payloads: List[Dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            payloads.append(payload)
        except (pickle.UnpicklingError, EOFError) as exc:
            print(f"    Failed to load {path}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"    Unexpected error while loading {path}: {exc}")
    print(f"  Loaded {len(payloads)} seed files for {model_key} ({_combination_key(filters)})")
    return payloads


def _merge_filters(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _apply_config_overrides(config: ContextualSwitchingTaskConfig, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)


def _load_bayesian_payloads(
    run_spec: RunSpec,
    comparison: ComparisonSpec,
) -> List[Dict[str, Any]]:
    base_filters = MODEL_INFO["bayesian"].base_filters
    overrides = dict(comparison.model_overrides.get("bayesian", {}))
    observation_noise_mode = overrides.pop("observation_noise_mode", None)
    observation_noise_mode = overrides.pop("observation_noise_modes", observation_noise_mode)
    combined_overrides = _merge_filters(base_filters, overrides)
    config = ContextualSwitchingTaskConfig("figure")
    _apply_config_overrides(config, combined_overrides)

    run_kwargs = dict(
        base_config=config,
        test_types=(run_spec.ood_test_type,),
        save_artifacts=False,
    )
    if observation_noise_mode is not None:
        run_kwargs["observation_noise_mode"] = observation_noise_mode
    artifacts = run_bayesian_generalization(**run_kwargs)
    test_loggers = artifacts.generalization_loggers.get(run_spec.ood_test_type, {})
    if not test_loggers:
        return []
    return [{"test_logger": test_loggers}]


def gather_payloads(
    run_spec: RunSpec,
    comparison: ComparisonSpec,
    export_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    payloads: Dict[str, List[Dict[str, Any]]] = {}
    for model_key in MODEL_ORDER:
        if model_key == "bayesian":
            continue
        if model_key not in comparison.model_overrides:
            continue
        filters = _merge_filters(
            MODEL_INFO[model_key].base_filters,
            comparison.model_overrides[model_key],
        )
        payloads[model_key] = _load_payloads(
            run_spec,
            model_key,
            filters,
            export_root,
            run_spec.max_seeds,
        )
    bayesian_payloads = _load_bayesian_payloads(run_spec, comparison)
    if bayesian_payloads:
        payloads["bayesian"] = bayesian_payloads
    return payloads


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockWindowSpec:
    post_window: int
    column_slice: slice | None = None


def _resolve_block_size_window(block_size: int, params: AnalysisParams) -> BlockWindowSpec | None:
    """Return the post-window and optional column slice for block-size sweeps."""

    block_size = int(block_size)
    if block_size <= 0:
        return None
    mode = params.block_size_window_mode
    if mode == "full_block" or not mode:
        return BlockWindowSpec(post_window=block_size)
    if mode == "peri_switch":
        post_window = min(block_size, params.block_size_peri_post_window)
        return BlockWindowSpec(post_window=max(1, post_window))
    if mode == "mid_block":
        start_offset = max(0, params.block_size_midpoint - params.block_size_mid_halfwidth)
        end_offset = params.block_size_midpoint + params.block_size_mid_halfwidth
        if block_size < end_offset:
            return None  # not enough room before the next switch
        post_window = max(1, end_offset)
        column_slice = slice(
            params.pre_window + start_offset,
            params.pre_window + end_offset,
        )
        return BlockWindowSpec(post_window=post_window, column_slice=column_slice)
    return BlockWindowSpec(post_window=block_size)


def compute_mean_sem(
    logger_list: Sequence[Any],
    *,
    error_type: str,
    pre_window: int,
    post_window: int,
    enforce_block_size_min_post_window: bool,
    column_slice: slice | Sequence[int] | np.ndarray | None = None,
    context_label: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, List[float]] | Tuple[None, ...]:
    all_errors: List[np.ndarray] = []
    time_avg_per_logger: List[float] = []

    for logger in logger_list:
        try:
            _, peri_switch_errors = calculate_error(
                logger,
                error_type=error_type,
                pre_window=pre_window,
                post_window=post_window,
                enforce_block_size_min_post_window=enforce_block_size_min_post_window,
            )
        except ValueError as exc:
            print(f"    Skipping logger in {context_label}: {exc}")
            continue
        if not peri_switch_errors:
            continue
        data = np.vstack(peri_switch_errors)
        if column_slice is not None:
            data = data[:, column_slice]
        all_errors.append(data)
        time_avg_per_logger.append(float(np.mean(data)))

    if not all_errors:
        print(f"    No peri-switch strips computed for {context_label}.")
        return (None,) * 6

    errors_np = np.vstack(all_errors)
    sample_count = errors_np.shape[0]
    mean_errors = errors_np.mean(axis=0)
    ddof = 1 if sample_count > 1 else 0
    sem_errors = errors_np.std(axis=0, ddof=ddof) / np.sqrt(sample_count)
    time_axis_arr = np.arange(-pre_window, post_window)
    if column_slice is not None:
        time_axis_arr = time_axis_arr[column_slice]
    time_avg_error = float(np.mean(mean_errors))
    if time_avg_per_logger:
        count = len(time_avg_per_logger)
        ddof_time = 1 if count > 1 else 0
        time_sem_error = float(np.std(time_avg_per_logger, ddof=ddof_time) / np.sqrt(count))
    else:
        time_sem_error = float("nan")
    return mean_errors, sem_errors, time_axis_arr, time_avg_error, time_sem_error, time_avg_per_logger


def _scaling_factor(ood_test_type: str) -> float:
    # scaling brings MSE values onto a comparable scale for plotting STD 0.3 or 0.4 for comparison. Computed using the variance of model responses for each std level.
    if ood_test_type == "ood_stds":
        return 0.2027 / 0.1751
    if ood_test_type == "ood_means":
        return 0.2027 / 0.1710
    return 1.0


def adjust_curve(
    ood_test_type: str,
    model_key: str,
    mean_values: Sequence[float],
    sem_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean_values, dtype=float)
    sem_arr = np.asarray(sem_values, dtype=float)

    if ood_test_type in {"ood_stds", "ood_means"}:
        factor = _scaling_factor(ood_test_type)
        mean_arr = mean_arr * factor
        sem_arr = sem_arr * factor
    return mean_arr, sem_arr


def summarise_test_loggers(
    run_spec: RunSpec,
    comparison_key: str,
    model_payloads: Mapping[str, Sequence[Dict[str, Any]]],
    params: AnalysisParams,
) -> Dict[str, Dict[Any, Tuple[float, float, List[float]]]]:
    summary: Dict[str, Dict[Any, Tuple[float, float, List[float]]]] = {}

    for model_key, payloads in model_payloads.items():
        per_ood: MutableMapping[Any, List[Any]] = {}
        for payload in payloads:
            test_logger_dict = payload.get("test_logger")
            if not test_logger_dict:
                continue
            for ood_value, logger in test_logger_dict.items():
                per_ood.setdefault(ood_value, []).append(logger)

        model_summary: Dict[Any, Tuple[float, float, List[float]]] = {}
        for ood_value, logger_list in per_ood.items():
            column_slice: slice | Sequence[int] | np.ndarray | None = None
            post_window = params.post_window
            if run_spec.ood_test_type == "block_size":
                window_spec = _resolve_block_size_window(int(ood_value), params)
                if window_spec is None:
                    print(
                        f"    Skipping logger in {comparison_key}:{model_key}@{ood_value}: "
                        "block too short for selected window",
                    )
                    continue
                post_window = window_spec.post_window
                column_slice = window_spec.column_slice
            error_type = params.error_type
            if model_key == "bayesian":
                error_type = "abs_from_mean" # "mse"
                # error_type = "mse"
            mean_errors, sem_errors, _, time_avg_error, time_sem_error, per_seed_vals = compute_mean_sem(
                logger_list,
                error_type=error_type,
                pre_window=params.pre_window,
                post_window=post_window,
                enforce_block_size_min_post_window=False,
                column_slice=column_slice,
                context_label=f"{comparison_key}:{model_key}@{ood_value}",
            )
            if mean_errors is None:
                continue
            model_summary[ood_value] = (
                time_avg_error,
                time_sem_error,
                per_seed_vals or [],
            )
        summary[model_key] = model_summary
    return summary


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def compare_models_at_points(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float]]]],
    model_a: str,
    model_b: str,
    ordered_values: Sequence[Any],
    *,
    alpha_levels: Tuple[float, float, float] = (0.05, 0.01, 0.001),
) -> List[Dict[str, Any]]:
    if model_a not in summary or model_b not in summary:
        return []

    results: List[Dict[str, Any]] = []
    for idx, value in enumerate(ordered_values):
        entry_a = summary[model_a].get(value)
        entry_b = summary[model_b].get(value)
        if not entry_a or not entry_b:
            continue
        seeds_a = entry_a[2]
        seeds_b = entry_b[2]
        if len(seeds_a) < 2 or len(seeds_b) < 2:
            continue
        if stats is not None:
            _, p_val = stats.ttest_ind(seeds_a, seeds_b, equal_var=False)
        else:  # fallback: normal approximation
            a = np.asarray(seeds_a, dtype=float)
            b = np.asarray(seeds_b, dtype=float)
            mean_diff = a.mean() - b.mean()
            var_a = a.var(ddof=1)
            var_b = b.var(ddof=1)
            se = np.sqrt(var_a / len(a) + var_b / len(b))
            if se == 0:
                p_val = 1.0
            else:
                t_stat = mean_diff / se
                p_val = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
        if np.isnan(p_val):
            signif = "n/s"
        elif p_val < alpha_levels[2]:
            signif = "***"
        elif p_val < alpha_levels[1]:
            signif = "**"
        elif p_val < alpha_levels[0]:
            signif = "*"
        else:
            signif = "n/s"
        results.append(
            {
                "value": value,
                "index": idx,
                "p_value": p_val,
                "significance": signif,
            }
        )
    return results


def bootstrap_model_slopes(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float]]]],
    model_key: str,
    ordered_values: Sequence[Any],
    *,
    boot_iterations: int,
    random_state: int | None = None,
) -> Tuple[np.ndarray | None, float, float]:
    rng = np.random.default_rng(random_state)
    per_value = []
    for value in ordered_values:
        entry = summary.get(model_key, {}).get(value)
        if entry is None or not entry[2]:
            return None, float("nan"), float("nan")
        per_value.append(np.asarray(entry[2], dtype=float))
    if len(per_value) < 2:
        return None, float("nan"), float("nan")

    x = np.asarray(ordered_values, dtype=float)
    slopes = []
    for _ in range(boot_iterations):
        means = []
        for seeds in per_value:
            idx = rng.integers(0, len(seeds), size=len(seeds))
            means.append(float(seeds[idx].mean()))
        y = np.asarray(means, dtype=float) * _scaling_factor("ood_stds")
        coeffs = np.polyfit(x, y, 1)
        slopes.append(coeffs[0])
    slopes_arr = np.asarray(slopes)
    return slopes_arr, float(slopes_arr.mean()), float(slopes_arr.std(ddof=1) / np.sqrt(len(slopes_arr)))


def compare_model_slopes(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float]]]],
    model_a: str,
    model_b: str,
    ordered_values: Sequence[Any],
    *,
    boot_iterations: int,
) -> Dict[str, Any]:
    slopes_a, mean_a, sem_a = bootstrap_model_slopes(
        summary, model_a, ordered_values, boot_iterations=boot_iterations
    )
    slopes_b, mean_b, sem_b = bootstrap_model_slopes(
        summary, model_b, ordered_values, boot_iterations=boot_iterations, random_state=1
    )
    if slopes_a is None or slopes_b is None:
        return {
            "mean_slope_a": float("nan"),
            "mean_slope_b": float("nan"),
            "sem_slope_a": float("nan"),
            "sem_slope_b": float("nan"),
            "p_value": float("nan"),
            "significance": "n/s",
        }

    if stats is not None:
        _, p_val = stats.ttest_ind(slopes_a, slopes_b, equal_var=False)
    else:
        diff = slopes_a.mean() - slopes_b.mean()
        var_a = slopes_a.var(ddof=1)
        var_b = slopes_b.var(ddof=1)
        se = np.sqrt(var_a / len(slopes_a) + var_b / len(slopes_b))
        if se == 0:
            p_val = 1.0
        else:
            t_stat = diff / se
            p_val = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))

    if np.isnan(p_val):
        signif = "n/s"
    elif p_val < 0.001:
        signif = "***"
    elif p_val < 0.01:
        signif = "**"
    elif p_val < 0.05:
        signif = "*"
    else:
        signif = "n/s"

    return {
        "mean_slope_a": mean_a,
        "sem_slope_a": sem_a,
        "mean_slope_b": mean_b,
        "sem_slope_b": sem_b,
        "p_value": p_val,
        "significance": signif,
    }


def plot_test_summary(
    run_spec: RunSpec,
    comparison: ComparisonSpec,
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float]]]],
    params: AnalysisParams,
    export_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=COLOR_SCHEME.panel_small_size)
    model_curves: Dict[str, Tuple[List[Any], np.ndarray]] = {}

    for model_key in MODEL_ORDER:
        entries = summary.get(model_key, {})
        if not entries:
            continue
        ordered = sorted(entries.keys(), key=lambda v: float(v))
        means = [entries[val][0] for val in ordered]
        sems = [entries[val][1] for val in ordered]
        means_adj, sems_adj = means, sems
        # adjust_curve(run_spec.ood_test_type, model_key, means, sems)

        # numeric x values for plotting (keep `ordered` as original for lookups later)
        x_vals = np.asarray([float(v) for v in ordered], dtype=float)
        means_arr = np.asarray(means_adj, dtype=float)
        sems_arr = np.asarray(sems_adj, dtype=float)

        # plot mean line with small markers
        ax.plot(
            x_vals,
            means_arr,
            marker="o",
            linewidth=0.8,
            markersize=2,
            label=MODEL_INFO[model_key].label,
            color=MODEL_INFO[model_key].color,
        )
        # shaded error band
        ax.fill_between(
            x_vals,
            means_arr - sems_arr,
            means_arr + sems_arr,
            color=MODEL_INFO[model_key].color,
            alpha=0.2,
            linewidth=0,
        )

        model_curves[model_key] = (ordered, means_adj)

    xlim = {
        "ood_stds": (0.2, None),
    }.get(run_spec.ood_test_type, (None, None))
    if xlim[0] is not None or xlim[1] is not None:
        ax.set_xlim(xlim)
    xlabel = {
        "ood_stds": "Observation std",
        "ood_means": "Observation mean",
        "block_size": "Block size",
    }.get(run_spec.ood_test_type, "OOD value")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time-averaged MSE")
    title = comparison.title or comparison.key
    # ax.set_title(title, fontsize=6)
    # ax.legend(fontsize=6)

    for ref in run_spec.iid_reference_values:
        ax.axvline(ref, linestyle="--", color=COLOR_SCHEME.iid_data, alpha=0.5, linewidth=2)

    if params.statistical_analysis and len(MODEL_ORDER) >= 2:
        model_a, model_b = MODEL_ORDER[:2]
        ordered_values = sorted(
            set(summary.get(model_a, {}).keys()).intersection(summary.get(model_b, {}).keys()),
            key=lambda v: float(v),
        )
        if run_spec.ood_test_type == "ood_stds" and ordered_values:
            ordered_values = [0.3, 0.7]  
        elif run_spec.ood_test_type == "ood_means" and ordered_values:
            ordered_values = [-0.2, 0.5, 1.2]      
        elif run_spec.ood_test_type == "block_size" and ordered_values:
            ordered_values = ordered_values[:1] + ordered_values[5:6] + ordered_values[-1:]

        stats_results = compare_models_at_points(summary, model_a, model_b, ordered_values)
        if stats_results and model_a in model_curves and model_b in model_curves:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            curve_a = dict(zip(*model_curves[model_a]))
            curve_b = dict(zip(*model_curves[model_b]))
            for res in stats_results:
                value = res["value"]
                y_here = max(curve_a.get(value, y_min), curve_b.get(value, y_min))
                ax.text(
                    value,
                    y_here + 0.04 * y_range,
                    res["significance"],
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        if run_spec.ood_test_type == "ood_stds" and ordered_values:
            slope_results = compare_model_slopes(
                summary,
                model_a,
                model_b,
                ordered_values,
                boot_iterations=params.boot_iterations,
            )
            if inset_axes is not None:
                inset = inset_axes(
                    ax,
                    width="80%",
                    height="100%",
                    loc="lower right",
                    bbox_to_anchor=(1.02, 0.22, 0.3, 0.3), # pushing the inset to the right and up. This overrides the width and height really though they are active.
                    bbox_transform=ax.transAxes,
                    borderpad=0.5,
                )
            else:
                bbox = ax.get_position()
                inset = fig.add_axes([bbox.x1 - 0.22, bbox.y0 + 0.05, 0.2, 0.16])
            xs = np.arange(2)
            heights = [slope_results["mean_slope_a"], slope_results["mean_slope_b"]]
            errs = [slope_results["sem_slope_a"], slope_results["sem_slope_b"]]
            inset.bar(
                xs,
                heights,
                yerr=errs,
                color=[MODEL_INFO[model_a].color, MODEL_INFO[model_b].color],
                alpha=0.8,
                capsize=3,
                width=0.27,
            )
            inset.set_xticks(xs)
            inset.set_xticklabels(
                [MODEL_INFO[model_a].label, MODEL_INFO[model_b].label],
                rotation=30,
                fontsize=6,
            )
            inset.set_ylabel("Slope", fontsize=6)
            inset.tick_params(axis="y", labelsize=6)
            inset.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.0, 0.2]))
            inset.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
            y_top = max(h + (e if np.isfinite(e) else 0) for h, e in zip(heights, errs))
            inset.text(0.5, y_top * 1.05 if y_top != 0 else 0.02, slope_results["significance"], ha="center", va="bottom", fontsize=7)

    output_path = export_dir / f"{comparison.key}_{run_spec.ood_test_type}_test_summary.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved test summary plot: {"./"+str(output_path)}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process_comparison(
    run_spec: RunSpec,
    comparison: ComparisonSpec,
    export_root: Path,
    params: AnalysisParams,
) -> None:
    analysis_dir = export_root / run_spec.run_name / "model_comparison" / comparison.key
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {run_spec.run_name} – {comparison.key}")
    payloads = gather_payloads(run_spec, comparison, export_root)
    if not any(payloads.values()):
        print("  No payloads found; skipping comparison.")
        return

    summary = summarise_test_loggers(run_spec, comparison.key, payloads, params)
    if not any(summary.values()):
        print("  No test loggers summarised; skipping test plot.")
        return
    plot_test_summary(run_spec, comparison, summary, params, analysis_dir)

    def main(argv: Sequence[str] | None = None) -> None:

        parser = argparse.ArgumentParser(description="Model comparison analysis for adaptation experiments")
        parser.add_argument(
            "--block-size-window-mode",
            choices=("full_block", "peri_switch", "mid_block"),
            default=AnalysisParams.block_size_window_mode,
            help="Window selection for block-size sweeps (full block, peri-switch window, or mid-block slice).",
        )
        parser.add_argument(
            "--block-size-peri-post-window",
            type=int,
            default=AnalysisParams.block_size_peri_post_window,
            help="Post-window length (steps after switch) for peri-switch mode.",
        )
        parser.add_argument(
            "--block-size-midpoint",
            type=int,
            default=AnalysisParams.block_size_midpoint,
            help="Center offset (steps after switch) for the mid-block slice.",
        )
        parser.add_argument(
            "--block-size-mid-halfwidth",
            type=int,
            default=AnalysisParams.block_size_mid_halfwidth,
            help="Half-width (in steps) around the midpoint for the mid-block slice.",
        )

        # In Jupyter/IPython environments, avoid parsing the notebook's argv by using an empty list.
        if argv is None:
            if "ipykernel" in sys.modules or "IPython" in sys.modules:
                argv = []
        args = parser.parse_args(argv)

        params = AnalysisParams(
            block_size_window_mode=args.block_size_window_mode,
            block_size_peri_post_window=args.block_size_peri_post_window,
            block_size_midpoint=args.block_size_midpoint,
            block_size_mid_halfwidth=args.block_size_mid_halfwidth,
        )
        export_root = DEFAULT_EXPORT_ROOT

        for run_key, run_spec in RUN_SPECS.items():
            run_path = export_root / run_spec.run_name
            if not run_path.exists():
                print(f"Run directory not found for {run_spec.run_name}; continuing with available models.")
            for comparison in run_spec.comparisons:
                process_comparison(run_spec, comparison, export_root, params)


if __name__ == "__main__":
    main()
