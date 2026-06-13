"""Helper routines for cst_analyze_gen_tests."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple, cast

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


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


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
    # Primary model pair for caption + inset in this panel.
    # If None, we fall back to parameters/default logic.
    primary_compare: Tuple[str, str] | None = None


@dataclass(frozen=True)
class RunSpec:
    """Definition of one experiment suite to analyse."""

    run_name: str
    ood_test_type: str
    comparisons: Sequence[ComparisonSpec]
    iid_reference_values: Sequence[float]
    weights_frozen: bool = True
    max_seeds: int | None = None


@dataclass
class AnalysisParams:
    """Global knobs for the analysis pipeline."""

    error_type: str = "abs_from_mean"
    pre_window: int = 3
    post_window: int = 30
    hazard_rate: float | None = None  # for bayesian model
    statistical_analysis: bool = True
    boot_iterations: int = 1000
    show_plots: bool = True
    seeds_limit: int | None = 20
    aggregate_inset: bool = True
    aggregate_alpha: float = 0.05
    aggregate_multiple_comparisons: str = "holm"  # 'none' | 'bonferroni' | 'holm'
    aggregate_inset_models: Tuple[str, str] = ("neuragem", "rnn_seq_len_50")


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
        hazard_rate=comparison.model_overrides.get("bayesian", {}).get("hazard_rate", None) if run_spec.ood_test_type != "ood_stds" else 0.5,
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
    params: AnalysisParams,
) -> Dict[str, List[Dict[str, Any]]]:
    payloads: Dict[str, List[Dict[str, Any]]] = {}

    # Cap the number of seeds loaded per model for faster iteration.
    # If run_spec.max_seeds is set, it takes precedence but still respects the global cap.
    effective_max_seeds: int | None
    if params.seeds_limit is None:
        effective_max_seeds = run_spec.max_seeds
    elif run_spec.max_seeds is None:
        effective_max_seeds = params.seeds_limit
    else:
        effective_max_seeds = min(run_spec.max_seeds, params.seeds_limit)

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
            effective_max_seeds,
        )

    # Only run/plot the Bayesian observer when it is explicitly requested in
    # this comparison. Otherwise, it can appear unexpectedly even if the entry
    # was commented out in `model_overrides`.
    if "bayesian" in comparison.model_overrides:
        bayesian_payloads = _load_bayesian_payloads(run_spec, comparison)
        if bayesian_payloads:
            payloads["bayesian"] = bayesian_payloads
    return payloads


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_mean_sem(
    logger_list: Sequence[Any],
    *,
    error_type: str,
    pre_window: int,
    post_window: int,
    enforce_block_size_min_post_window: bool,
    context_label: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, List[float], List[int]] | Tuple[None, ...]:
    all_errors: List[np.ndarray] = []
    time_avg_per_logger: List[float] = []
    strip_counts: List[int] = []

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
        all_errors.append(data)
        time_avg_per_logger.append(float(np.mean(data)))
        strip_counts.append(int(data.shape[0]))

    if not all_errors:
        print(f"    No peri-switch strips computed for {context_label}.")
        return (None,) * 7

    errors_np = np.vstack(all_errors)
    sample_count = errors_np.shape[0]
    mean_errors = errors_np.mean(axis=0)
    ddof = 1 if sample_count > 1 else 0
    sem_errors = errors_np.std(axis=0, ddof=ddof) / np.sqrt(sample_count)
    time_axis_arr = np.arange(-pre_window, post_window)
    time_avg_error = float(np.mean(mean_errors))
    if time_avg_per_logger:
        count = len(time_avg_per_logger)
        ddof_time = 1 if count > 1 else 0
        time_sem_error = float(np.std(time_avg_per_logger, ddof=ddof_time) / np.sqrt(count))
    else:
        time_sem_error = float("nan")
    return mean_errors, sem_errors, time_axis_arr, time_avg_error, time_sem_error, time_avg_per_logger, strip_counts


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
) -> Dict[str, Dict[Any, Tuple[float, float, List[float], List[int]]]]:
    """Summarise MSE per model and test value.

    Returns mapping: model -> ood_value -> (mean, sem, per_seed_vals, per_seed_strip_counts)

    Notes
    -----
    * `per_seed_vals` has one value per seed/logger.
    * `per_seed_strip_counts` stores peri-switch-strip counts for the same seeds.
    """
    summary: Dict[str, Dict[Any, Tuple[float, float, List[float], List[int]]]] = {}

    for model_key, payloads in model_payloads.items():
        per_ood: MutableMapping[Any, List[Any]] = {}
        for payload in payloads:
            test_logger_dict = payload.get("test_logger")
            if not test_logger_dict:
                continue
            for ood_value, logger in test_logger_dict.items():
                per_ood.setdefault(ood_value, []).append(logger)

        model_summary: Dict[Any, Tuple[float, float, List[float], List[int]]] = {}
        for ood_value, logger_list in per_ood.items():
            post_window = (
                int(ood_value)
                if run_spec.ood_test_type == "block_size"
                else params.post_window
            )
            error_type = params.error_type
            if model_key == "bayesian":
                error_type = "abs_from_mean"  # "mse"
                # error_type = "mse"
            mean_errors, sem_errors, _, time_avg_error, time_sem_error, per_seed_vals, strip_counts = compute_mean_sem(
                logger_list,
                error_type=error_type,
                pre_window=params.pre_window,
                post_window=post_window,
                enforce_block_size_min_post_window=False,
                context_label=f"{comparison_key}:{model_key}@{ood_value}",
            )
            if mean_errors is None:
                continue
            # compute_mean_sem returns floats; keep explicit casts to satisfy type checkers.
            model_summary[ood_value] = (
                cast(float, time_avg_error),
                cast(float, time_sem_error),
                list(per_seed_vals or []),
                list(strip_counts or []),
            )
        summary[model_key] = model_summary
    return summary


# ---------------------------------------------------------------------------
# Aggregate statistics across x-axis values
# ---------------------------------------------------------------------------


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        p = float(p_values[idx])
        adj = (m - rank) * p
        adj = min(1.0, max(prev, adj))
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def _bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    return [min(1.0, float(p) * m) for p in p_values]


def _format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "p=nan"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.3g}"  # Nature Neuroscience typically ok with 3 sig figs


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/s"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n/s"


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    diff = a - b
    n = diff.size
    if n < 2:
        return float("nan"), float("nan"), 0
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        return t, p, n - 1
    t = mean / (sd / math.sqrt(n))
    if stats is not None:
        p = float(stats.t.sf(abs(t), df=n - 1) * 2)
    else:
        # normal approx fallback
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return float(t), float(p), n - 1


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    na = a.size
    nb = b.size
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    ma = float(a.mean())
    mb = float(b.mean())
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    se2 = va / na + vb / nb
    if se2 == 0:
        t = float("inf") if ma != mb else 0.0
        p = 0.0 if ma != mb else 1.0
        return t, p, float("inf")
    t = (ma - mb) / math.sqrt(se2)
    # Welch-Satterthwaite df
    df = (se2 ** 2) / ((va ** 2) / ((na ** 2) * (na - 1)) + (vb ** 2) / ((nb ** 2) * (nb - 1)))
    if stats is not None and np.isfinite(df):
        p = float(stats.t.sf(abs(t), df=df) * 2)
    else:
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return float(t), float(p), float(df)


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
    if sd == 0:
        return float("inf")
    return float(diff.mean() / sd)


def _hedges_g_independent(a: np.ndarray, b: np.ndarray) -> float:
    na = a.size
    nb = b.size
    if na < 2 or nb < 2:
        return float("nan")
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return float("inf")
    d = (float(a.mean()) - float(b.mean())) / sp
    # small-sample correction
    j = 1 - (3 / (4 * (na + nb) - 9))
    return float(j * d)


def aggregate_mse_per_seed_weighted(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float], List[int]]]],
    model_key: str,
    ordered_values: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return per-seed weighted aggregate MSE and total strip counts.

    Each seed gets a weighted mean across OOD values:

    $$\bar{e}_s = \frac{\sum_v w_{s,v} e_{s,v}}{\sum_v w_{s,v}}$$

    where $w_{s,v}$ is the number of peri-switch strips contributing to the
    time-averaged MSE at OOD value v.
    """
    per_value_vals: List[np.ndarray] = []
    per_value_wts: List[np.ndarray] = []
    for value in ordered_values:
        entry = summary.get(model_key, {}).get(value)
        if entry is None:
            return np.asarray([]), np.asarray([])
        vals = np.asarray(entry[2], dtype=float)
        wts = np.asarray(entry[3], dtype=float)
        if vals.size == 0 or wts.size == 0 or vals.size != wts.size:
            return np.asarray([]), np.asarray([])
        per_value_vals.append(vals)
        per_value_wts.append(wts)

    # Align by seed index (assumes payload loading yields consistent ordering per model/value).
    # If some OOD values have fewer seeds, truncate to the minimum number.
    n = min(arr.size for arr in per_value_vals)
    if n < 2:
        return np.asarray([]), np.asarray([])
    vals_stack = np.vstack([arr[:n] for arr in per_value_vals])  # [n_values, n_seeds]
    wts_stack = np.vstack([arr[:n] for arr in per_value_wts])
    denom = wts_stack.sum(axis=0)
    # avoid divide-by-zero
    denom = np.where(denom == 0, np.nan, denom)
    agg = np.nansum(vals_stack * wts_stack, axis=0) / denom
    total_w = np.nansum(wts_stack, axis=0)
    return agg.astype(float), total_w.astype(float)


def compare_models_on_aggregate_mse(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float], List[int]]]],
    model_a: str,
    model_b: str,
    ordered_values: Sequence[Any],
    *,
    ood_test_type: str,
    multiple_comparisons: str = "holm",
) -> Dict[str, Any]:
    """Model comparison on weighted aggregate MSE across OOD values."""
    a, w_a = aggregate_mse_per_seed_weighted(summary, model_a, ordered_values)
    b, w_b = aggregate_mse_per_seed_weighted(summary, model_b, ordered_values)
    if a.size == 0 or b.size == 0:
        return {"ok": False, "reason": "insufficient overlap"}

    # We assume seeds are aligned across models in this project and always use
    # paired tests for consistency (per manuscript). If alignment is ever
    # violated (missing/different seed sets), callers should fix the upstream
    # data loading rather than silently switching statistical tests.
    if a.size != b.size:
        n = int(min(a.size, b.size))
        if n < 2:
            return {"ok": False, "reason": "insufficient paired samples"}
        a = a[:n]
        b = b[:n]

    # Apply same scaling adjustments used for plotted means.
    if ood_test_type in {"ood_stds", "ood_means"}:
        factor = _scaling_factor(ood_test_type)
        a = a * factor
        b = b * factor
    elif ood_test_type == "block_size":
        a = _adjust_block_size_curve(model_a, a)
        b = _adjust_block_size_curve(model_b, b)

    t, p, df = _paired_ttest(a, b)
    effect = _cohens_d_paired(a, b)
    test_name = "paired t-test"
    n = int(a.size)
    df_out: float | int = int(df)

    return {
        "ok": True,
        "test": test_name,
        "t": float(t),
        "p": float(p),
        "df": df_out,
        "n": n,
        "effect": float(effect),
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "sem_a": float(np.nanstd(a, ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan"),
        "sem_b": float(np.nanstd(b, ddof=1) / math.sqrt(b.size)) if b.size > 1 else float("nan"),
        "delta": float(np.nanmean(a - b)),
        "total_strips_a_mean": float(np.nanmean(w_a)) if w_a.size else float("nan"),
        "total_strips_b_mean": float(np.nanmean(w_b)) if w_b.size else float("nan"),
        "multiple_comparisons": multiple_comparisons,
    }


def caption_sentence_for_aggregate_test(
    run_spec: RunSpec,
    comparison: ComparisonSpec,
    model_a: str,
    model_b: str,
    result: Mapping[str, Any],
    *,
    p_adjusted: float | None = None,
) -> str:
    label_a = MODEL_INFO[model_a].label
    label_b = MODEL_INFO[model_b].label
    test = result.get("test", "t-test")
    t = result.get("t", float("nan"))
    df = result.get("df", float("nan"))
    p = float(p_adjusted) if p_adjusted is not None else float(result.get("p", float("nan")))
    effect = result.get("effect", float("nan"))
    mean_a = result.get("mean_a", float("nan"))
    sem_a = result.get("sem_a", float("nan"))
    mean_b = result.get("mean_b", float("nan"))
    sem_b = result.get("sem_b", float("nan"))
    delta = result.get("delta", float("nan"))
    n = result.get("n")
    if isinstance(n, tuple):
        n_str = f"n={n[0]},{n[1]}"
    else:
        n_str = f"n={n}"
    df_str = f"df={df:.1f}" if isinstance(df, float) and not float(df).is_integer() else f"df={int(df)}" if np.isfinite(df) else "df=nan"

    # Manuscript-style formatting (matches prior CST write-up).
    # Note: We still keep the underlying statistic identical (aggregate across x-values
    # with peri-switch-strip weights).
    return (
        f"Aggregate time-averaged MSE across all {run_spec.ood_test_type}: "
        f"{label_a} (mean±s.e.m. {mean_a:.3g}±{sem_a:.2g}) vs {label_b} (mean±s.e.m. {mean_b:.3g}±{sem_b:.2g}), "
        f"{test}, {n_str} model runs with independent seeds, t({df_str})={t:.3g}, {_format_p_value(p)}, "
        f"effect size={effect:.3g}, Δ={delta:.3g}."
    )


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
    """Deprecated: pointwise tests at selected x-axis values.

    The original analysis used significance testing at a small number of
    hand-picked x-axis points. This is intentionally removed in favour of
    principled aggregate comparisons across all novel tests.
    """
    return []


def bootstrap_model_slopes(
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float], List[int]]]],
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
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float], List[int]]]],
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
        ttest_res = stats.ttest_ind(slopes_a, slopes_b, equal_var=False)
        # SciPy returns either a tuple-like or a stats object depending on version.
        p_val = float(cast(Any, getattr(ttest_res, "pvalue", ttest_res[1])))
    else:
        diff = slopes_a.mean() - slopes_b.mean()
        var_a = slopes_a.var(ddof=1)
        var_b = slopes_b.var(ddof=1)
        se = np.sqrt(var_a / len(slopes_a) + var_b / len(slopes_b))
        if se == 0:
            p_val = 1.0
        else:
            t_stat = diff / se
            p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

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
    summary: Mapping[str, Mapping[Any, Tuple[float, float, List[float], List[int]]]],
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
        means_adj, sems_adj = adjust_curve(run_spec.ood_test_type, model_key, means, sems)

        # numeric x values for plotting (keep `ordered` as original for lookups later)
        x_vals = np.asarray([float(v) for v in ordered], dtype=float)
        means_arr = np.asarray(means_adj, dtype=float)
        sems_arr = np.asarray(sems_adj, dtype=float)

        # plot mean line (Bayesian observer shown as dashed line without markers)
        is_bayesian = model_key == "bayesian"
        ax.plot(
            x_vals,
            means_arr,
            linestyle="--" if is_bayesian else "-",
            marker="" if is_bayesian else "o",
            linewidth=0.8 if not is_bayesian else 1.2,
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
    # Compress figure width for ood_stds plots
    if run_spec.ood_test_type == "ood_stds":
        bbox = ax.get_position()
        ax.set_position([bbox.x0, bbox.y0, bbox.width * 0.6, bbox.height])
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
    ax.legend(fontsize=6)

    for ref in run_spec.iid_reference_values:
        ax.axvline(ref, linestyle="--", color=COLOR_SCHEME.iid_data, alpha=0.5, linewidth=2)

    ordered_values_all = sorted(
        {val for model_key in MODEL_ORDER for val in summary.get(model_key, {}).keys()},
        key=lambda v: float(v),
    )

    caption_lines: List[str] = []

    # Aggregate MSE comparisons across all novel tests.
    if params.statistical_analysis and len(MODEL_ORDER) >= 2 and ordered_values_all:
        pair_results: List[Tuple[Tuple[str, str], Dict[str, Any]]] = []
        p_raw: List[float] = []
        for i, model_a in enumerate(MODEL_ORDER):
            for model_b in MODEL_ORDER[i + 1 :]:
                # Use intersection of OOD values so aggregate is computed on shared set.
                common_values = sorted(
                    set(summary.get(model_a, {}).keys()).intersection(summary.get(model_b, {}).keys()),
                    key=lambda v: float(v),
                )
                if len(common_values) < 2:
                    continue
                res = compare_models_on_aggregate_mse(
                    summary,
                    model_a,
                    model_b,
                    common_values,
                    ood_test_type=run_spec.ood_test_type,
                    multiple_comparisons=params.aggregate_multiple_comparisons,
                )
                if not res.get("ok"):
                    continue
                pair_results.append(((model_a, model_b), res))
                p_raw.append(float(res.get("p", float("nan"))))

        # Multiple-comparisons correction across the set of pairwise tests within this panel.
        p_adj = p_raw
        if params.aggregate_multiple_comparisons == "bonferroni":
            p_adj = _bonferroni_adjust(p_raw)
        elif params.aggregate_multiple_comparisons == "holm":
            p_adj = _holm_adjust(p_raw)

        # Caption output: by default report the primary comparison for this panel.
        primary_pair = comparison.primary_compare
        for (model_a, model_b), res, p_corr in zip(
            [pair[0] for pair in pair_results],
            [pair[1] for pair in pair_results],
            p_adj,
        ):
            if primary_pair is not None and (model_a, model_b) != primary_pair and (model_b, model_a) != primary_pair:
                continue
            caption_lines.append(
                caption_sentence_for_aggregate_test(
                    run_spec,
                    comparison,
                    model_a,
                    model_b,
                    res,
                    p_adjusted=float(p_corr),
                )
            )

        # Add a small inset summary for aggregate means.
        # For this plot variant, we use the declared primary pair if provided.
        if params.aggregate_inset:
            m1, m2 = comparison.primary_compare or params.aggregate_inset_models
            if m1 in MODEL_INFO and m2 in MODEL_INFO:
                common_values0 = sorted(
                    set(summary.get(m1, {}).keys()).intersection(summary.get(m2, {}).keys()),
                    key=lambda v: float(v),
                )
                if len(common_values0) >= 2:
                    a, _ = aggregate_mse_per_seed_weighted(summary, m1, common_values0)
                    b, _ = aggregate_mse_per_seed_weighted(summary, m2, common_values0)
                    if a.size and b.size:
                        if run_spec.ood_test_type in {"ood_stds", "ood_means"}:
                            factor = _scaling_factor(run_spec.ood_test_type)
                            a = a * factor
                            b = b * factor
                        elif run_spec.ood_test_type == "block_size":
                            a = _adjust_block_size_curve(m1, a)
                            b = _adjust_block_size_curve(m2, b)

                        mean_a = float(np.nanmean(a))
                        mean_b = float(np.nanmean(b))
                        sem_a = float(np.nanstd(a, ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan")
                        sem_b = float(np.nanstd(b, ddof=1) / math.sqrt(b.size)) if b.size > 1 else float("nan")

                        # Place outside to the right of the axes to avoid covering data.
                        if inset_axes is not None:
                            inset_agg = inset_axes(
                                ax,
                                width="30%",
                                height="26%",
                                loc="center left",
                                bbox_to_anchor=(1.12, 0.52, 1.0, 1.0),
                                bbox_transform=ax.transAxes,
                                borderpad=0.0,
                            )
                        else:
                            bbox = ax.get_position()
                            inset_agg = fig.add_axes((bbox.x1 + 0.04, bbox.y0 + 0.30, 0.10, 0.12))

                        xs = np.arange(2)
                        inset_agg.bar(
                            xs,
                            [mean_a, mean_b],
                            yerr=[sem_a, sem_b],
                            color=[MODEL_INFO[m1].color, MODEL_INFO[m2].color],
                            alpha=0.8,
                            capsize=3,
                            width=0.4,
                        )
                        inset_agg.set_xticks(xs)
                        inset_agg.set_xticklabels(
                            [MODEL_INFO[m1].label, MODEL_INFO[m2].label],
                            rotation=30,
                            fontsize=6,
                            ha="right",
                        )
                        for tick in inset_agg.get_xticklabels():
                            tick.set_rotation_mode("anchor")
                            tick.set_verticalalignment("top")

                        # Make the inset visually slimmer by placing the y-axis on the right.
                        # NOTE: `tick_right()` moves ticks/labels, but the y-axis *line* is a spine.
                        # Some mpl styles hide/show spines; explicitly move the right spine on.
                        inset_agg.spines["left"].set_visible(False)
                        inset_agg.spines["right"].set_visible(True)
                        inset_agg.yaxis.set_ticks_position("right")
                        inset_agg.yaxis.set_label_position("right")
                        inset_agg.set_ylabel("Agg. MSE", fontsize=6, labelpad=2)
                        inset_agg.tick_params(axis="y", labelsize=6, pad=1)

                        # Reduce x-label crowding: align labels slightly to the left and add padding.
                        inset_agg.tick_params(axis="x", labelsize=6, pad=2)
                        inset_agg.margins(x=0.05)

                        # Ensure the right spine is actually used for ticks.
                        inset_agg.tick_params(axis="y", which="both", left=False, right=True)

                        # Add significance annotation for the primary comparison.
                        if params.statistical_analysis:
                            res_primary = compare_models_on_aggregate_mse(
                                summary,
                                m1,
                                m2,
                                common_values0,
                                ood_test_type=run_spec.ood_test_type,
                                multiple_comparisons=params.aggregate_multiple_comparisons,
                            )
                            if res_primary.get("ok"):
                                p_here = float(res_primary.get("p", float("nan")))
                                stars = _p_to_stars(p_here)
                                y_top = max(
                                    mean_a + (sem_a if np.isfinite(sem_a) else 0.0),
                                    mean_b + (sem_b if np.isfinite(sem_b) else 0.0),
                                )
                                y_text = y_top + 0.06 * (abs(y_top) if y_top != 0 else 1.0)
                                inset_agg.text(
                                    0.5,
                                    y_text,
                                    stars,
                                    ha="center",
                                    va="bottom",
                                    fontsize=7,
                                )
                        y_low = min(
                            mean_a - (sem_a if np.isfinite(sem_a) else 0.0),
                            mean_b - (sem_b if np.isfinite(sem_b) else 0.0),
                        )
                        y_top = max(
                            mean_a + (sem_a if np.isfinite(sem_a) else 0.0),
                            mean_b + (sem_b if np.isfinite(sem_b) else 0.0),
                        )
                        span = max(y_top - y_low, 1e-1)
                        y_pad = 0.2 * span
                        inset_agg.set_ylim(y_low - y_pad * 1.5, y_top + y_pad * 0.8)
                        y_min, y_max = inset_agg.get_ylim()
                        span = max(y_max - y_min, 1e-9)

                        def _candidate_ticks(step: float) -> List[float]:
                            start = math.ceil((y_min - 1e-9) / step)
                            end = math.floor((y_max + 1e-9) / step)
                            if end < start:
                                return []
                            decimals = max(0, int(round(-math.log10(step))))
                            return [round(k * step, decimals) for k in range(start, end + 1)]

                        ticks: List[float] = []
                        for step in (0.1, 0.05, 0.02, 0.01):
                            ticks = _candidate_ticks(step)
                            if len(ticks) >= 2:
                                break

                        if len(ticks) >= 2:
                            mid = (y_min + y_max) / 2
                            lower = max([t for t in ticks if t <= mid], default=ticks[0])
                            upper = min([t for t in ticks if t >= mid], default=ticks[-1])
                            if lower == upper:
                                idx = ticks.index(lower)
                                if idx + 1 < len(ticks):
                                    upper = ticks[idx + 1]
                                elif idx - 1 >= 0:
                                    lower = ticks[idx - 1]
                            tick_low, tick_high = sorted([lower, upper])
                        else:
                            tick_low, tick_high = y_min, y_max
                        inset_agg.set_yticks([tick_low, tick_high])
                        inset_agg.set_xlim(-0.7, 1.7)


    # Keep the existing slope test for ood_stds (as requested).
    if params.statistical_analysis and run_spec.ood_test_type == "ood_stds" and len(MODEL_ORDER) >= 2:
        model_a, model_b = MODEL_ORDER[:2]
        ordered_values = sorted(
            set(summary.get(model_a, {}).keys()).intersection(summary.get(model_b, {}).keys()),
            key=lambda v: float(v),
        )
        if ordered_values:
            slope_results = compare_model_slopes(
                summary,
                model_a,
                model_b,
                ordered_values,
                boot_iterations=params.boot_iterations,
            )
            print(
                "  Slope comparison "
                f"({run_spec.ood_test_type}, {comparison.key}): "
                f"{MODEL_INFO[model_a].label} slope={slope_results['mean_slope_a']:.3g}±{slope_results['sem_slope_a']:.2g} vs "
                f"{MODEL_INFO[model_b].label} slope={slope_results['mean_slope_b']:.3g}±{slope_results['sem_slope_b']:.2g}, "
                f"p={slope_results['p_value']:.3g} ({slope_results['significance']})."
            )
            if inset_axes is not None:
                inset = inset_axes(
                    ax,
                    width="80%",
                    height="100%",
                    loc="lower right",
                    bbox_to_anchor=(1.02, 0.22, 0.3, 0.3),
                    bbox_transform=ax.transAxes,
                    borderpad=0.5,
                )
            else:
                bbox = ax.get_position()
                inset = fig.add_axes((bbox.x1 - 0.22, bbox.y0 + 0.05, 0.2, 0.16))
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
            y_top = max(h + (e if np.isfinite(e) else 0) for h, e in zip(heights, errs))
            inset.text(
                0.5,
                y_top * 1.05 if y_top != 0 else 0.02,
                slope_results["significance"],
                ha="center",
                va="bottom",
                fontsize=7,
            )

    output_path = export_dir / f"{comparison.key}_{run_spec.ood_test_type}_test_summary.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved test summary plot: ./{output_path}")

    if caption_lines:
        caption_path = export_dir / f"{comparison.key}_{run_spec.ood_test_type}_aggregate_stats_caption.txt"
        caption_text = "\n".join(caption_lines) + "\n"
        caption_path.write_text(caption_text)
        print("\n" + "\n".join(caption_lines))
        print(f"  Saved caption-ready summary: ./{caption_path}")
