"""Bayesian observer baseline for contextual switching generalization tests.

This module collects the Bayesian analysis that was previously interleaved with
the training scripts.  It provides utilities to:

* fit (via a simple grid search) the hazard rate of a Bayesian observer on the
  training distribution,
* run the observer on the contextual switching task, and
* evaluate generalization on the same three sweeps used for the neural models
  (novel means, novel block sizes, and novel observation noise).

The core entry point is :func:`run_bayesian_generalization`, which returns the
loggers in the same shape as the neural pipelines and optionally persists them
to ``exports/contextual_switching_task/experiments`` for downstream analysis.
"""

from __future__ import annotations

import math
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from configs import ContextualSwitchingTaskConfig
from datasets import TaskDataset
from functions_and_utils import Logger


DEFAULT_EXPORT_ROOT = Path("./exports/contextual_switching_task/experiments")

ObservationNoiseMode = Literal["fixed", "ground_truth"]
_OBSERVATION_NOISE_MODE_ALIASES = {"dataset": "ground_truth"}
_VALID_OBSERVATION_NOISE_MODES = {"fixed", "ground_truth"}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BayesianRunArtifacts:
    """Container for train/test loggers and metadata."""

    train_logger: Logger
    generalization_loggers: Dict[str, Dict[Union[int, float], Logger]]
    hazard_rate: float
    training_mse: float
    metadata: Dict[str, Any]

    def as_pickle_payload(self) -> Dict[str, Any]:
        """Return a serialisable representation for saving to disk."""

        return {
            "train_logger": self.train_logger,
            "test_loggers": self.generalization_loggers,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Core Bayesian observer implementation
# ---------------------------------------------------------------------------


def _normal_pdf(y: float, means: np.ndarray, std: float) -> np.ndarray:
    """Return the normal PDF evaluated at ``y`` for each mean in ``means``."""

    coef = 1.0 / (math.sqrt(2.0 * math.pi) * std)
    exponent = -0.5 * ((y - means) / std) ** 2
    return coef * np.exp(exponent)


def _run_bayesian_filter(
    observations: np.ndarray,
    latent_values: np.ndarray,
    hazard_rate: float,
    observation_std: float,
    prior: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a one-step-ahead Bayesian filter on a sequence of observations."""

    latent_values = np.asarray(latent_values, dtype=float)
    if latent_values.ndim != 1:
        raise ValueError("latent_values must be a one-dimensional array")
    if prior is None or len(prior) != len(latent_values):
        prior = np.ones_like(latent_values, dtype=float) / len(latent_values)
    prior = prior.astype(float, copy=True)
    posterior = prior.copy()

    predictions = np.zeros_like(observations, dtype=float)
    posterior_history = np.zeros((len(observations), len(latent_values)), dtype=float)

    eps = 1e-12
    hazard_rate = float(np.clip(hazard_rate, eps, 1.0 - eps))

    for idx, y in enumerate(observations):
        predictions[idx] = float(np.dot(posterior, latent_values))
        likelihoods = _normal_pdf(float(y), latent_values, observation_std)
        predictive_prior = (1.0 - hazard_rate) * posterior + hazard_rate * prior
        numerator = predictive_prior * likelihoods
        total = float(np.sum(numerator))
        if total <= eps:
            numerator = np.full_like(numerator, fill_value=eps)
            total = float(np.sum(numerator))
        posterior = numerator / total
        posterior_history[idx] = posterior

    return predictions, posterior_history


def _estimate_hazard_rate(block_sizes: Sequence[float]) -> float:
    """Roughly estimate the switch hazard from the observed block sizes."""

    block_sizes = np.asarray(block_sizes, dtype=float)
    if block_sizes.size == 0:
        return 0.05  # fallback
    mean_block = float(np.mean(block_sizes))
    mean_block = max(mean_block, 1.0)
    hazard = 1.0 / mean_block
    # Keep the grid search stable by clamping to a sensible range.
    return float(np.clip(hazard, 1e-4, 0.5))


def _fit_hazard_rate(
    dataset: TaskDataset,
    observation_std: float,
    candidate_hazards: Sequence[float],
) -> Tuple[float, float]:
    """Select the hazard rate that minimises MSE on the provided dataset."""

    observations = np.asarray(dataset.data_sequence, dtype=float)
    latent_values = np.unique(np.asarray(dataset.latent_sequence, dtype=float))

    best_hazard: Optional[float] = None
    best_mse = float("inf")

    for candidate in candidate_hazards:
        preds, _ = _run_bayesian_filter(observations, latent_values, candidate, observation_std)
        mse = float(np.mean((preds - observations) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_hazard = float(candidate)

    if best_hazard is None:
        raise RuntimeError("Failed to select a hazard rate from the candidate grid.")

    return best_hazard, best_mse


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _get_generalization_values(
    test_type: str, base_config: ContextualSwitchingTaskConfig
) -> Sequence[Union[int, float]]:
    """Default sweep values mirroring :func:`train_and_infer_functions.run_generalized_tests`."""

    if test_type == "ood_means":
        return [float(x) for x in np.round(np.arange(-0.2, 1.3, 0.1), 1)]
    if test_type == "training_means":
        return [float(x) for x in base_config.training_data_means]
    if test_type == "ood_stds":
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    if test_type == "block_size":
        return [10, 12, 13, 20, 30, 40, 50, 60, 70, 80]
    raise ValueError(f"Unsupported test_type '{test_type}'.")


def _pad_dataset_for_generalization(dataset: TaskDataset, config: ContextualSwitchingTaskConfig) -> None:
    """Pad the dataset with a neutral block so early samples are logged."""

    pad_length = config.seq_len + config.pre_window
    rng = np.random.default_rng(config.env_seed + 999)
    first_mean = dataset.latent_sequence[0]
    candidate_means = [m for m in config.training_data_means if m != first_mean]
    if not candidate_means:
        candidate_means = [0.505]
    pad_mean = float(rng.choice(candidate_means))

    dataset.latent_sequence = [pad_mean] * pad_length + list(dataset.latent_sequence)
    high_level_prefix = list(dataset.high_level_latent_sequence[:pad_length])
    dataset.high_level_latent_sequence = high_level_prefix + list(dataset.high_level_latent_sequence)
    while len(dataset.high_level_latent_sequence) < len(dataset.latent_sequence):
        dataset.high_level_latent_sequence.append(dataset.high_level_latent_sequence[-1])
    dataset.block_sizes.insert(0, pad_length)
    dataset.data_sequence = dataset.generate_data_sequence()


def _prepare_dataset_for_test(
    base_config: ContextualSwitchingTaskConfig,
    test_type: str,
    value: Union[int, float],
    seed_offset: int,
) -> TaskDataset:
    """Create a dataset configured for a particular generalisation sweep."""

    config = deepcopy(base_config)
    config.env_seed = base_config.env_seed + seed_offset

    if test_type in {"ood_means", "training_means"}:
        config.training_data_means = [float(value)]
        no_of_blocks = 2
    elif test_type == "ood_stds":
        config.default_std = float(value)
        no_of_blocks = 10
    elif test_type == "block_size":
        config.block_size = int(value)
        config.block_duration_distribution = "fixed_block_size"
        baseline_block_size = 40
        default_no_of_blocks = 20
        if config.block_size < baseline_block_size:
            multiplier = 1 + 0.5 * ((baseline_block_size / config.block_size) - 1)
            no_of_blocks = int(default_no_of_blocks * multiplier)
        else:
            no_of_blocks = default_no_of_blocks
    else:
        raise ValueError(f"Unsupported test_type '{test_type}'.")

    dataset = TaskDataset(no_of_blocks=no_of_blocks, config=config)

    if test_type in {"ood_means", "ood_stds", "block_size"}:
        _pad_dataset_for_generalization(dataset, config)

    return dataset


# ---------------------------------------------------------------------------
# Logger construction
# ---------------------------------------------------------------------------


def _as_time_step_array(value: float) -> np.ndarray:
    """Return value formatted like the neural loggers (batch=1, stride=1, features=1)."""

    return np.array([[[value]]], dtype=np.float32)


def _resolve_observation_noise_mode(
    setting: Union[
        ObservationNoiseMode,
        Mapping[str, ObservationNoiseMode],
        Mapping[str, str],
        str,
    ],
    test_type: str,
) -> ObservationNoiseMode:
    """Return the observation-noise mode to use for a given generalisation test."""

    if isinstance(setting, MappingABC):
        if "default" in setting:
            mode: str = str(setting.get(test_type, setting["default"]))
        else:
            mode = str(setting.get(test_type, "fixed"))
    else:
        mode = str(setting)
    mode = _OBSERVATION_NOISE_MODE_ALIASES.get(mode, mode)
    if mode not in _VALID_OBSERVATION_NOISE_MODES:
        valid = ", ".join(sorted(_VALID_OBSERVATION_NOISE_MODES | set(_OBSERVATION_NOISE_MODE_ALIASES)))
        raise ValueError(f"Unsupported observation_noise_mode '{mode}'. Valid options: {valid}.")
    return mode  # type: ignore[return-value]


def _simulate_bayesian_on_dataset(
    dataset: TaskDataset,
    hazard_rate: float,
    prior: Optional[np.ndarray] = None,
    observation_std: Optional[float] = None,
    latent_values_override: Optional[np.ndarray] = None,
) -> Tuple[Logger, float]:
    """Run the Bayesian observer on ``dataset`` and return a populated logger."""

    observation_std = float(observation_std)
    observations = np.asarray(dataset.data_sequence, dtype=float)
    if latent_values_override is not None:
        latent_values = np.asarray(latent_values_override, dtype=float)
    else:
        latent_values = np.unique(np.asarray(dataset.latent_sequence, dtype=float))
        latent_values.sort()

    if prior is not None and len(prior) != len(latent_values):
        prior = None  # Reset to uniform if lengths mismatch

    predictions, posterior_history = _run_bayesian_filter(
        observations=observations,
        latent_values=latent_values,
        hazard_rate=hazard_rate,
        observation_std=observation_std,
        prior=prior,
    )

    logger = Logger()
    logger.phases = [("Bayesian observer", 0)]
    logger.config = dataset.config

    hl_sequence = (
        np.asarray(dataset.high_level_latent_sequence, dtype=float)
        if hasattr(dataset, "high_level_latent_sequence")
        else np.zeros_like(observations)
    )

    for obs, pred, ll_val, hl_val, posterior in zip(
        observations, predictions, dataset.latent_sequence, hl_sequence, posterior_history
    ):
        step_input = _as_time_step_array(obs)
        step_pred = _as_time_step_array(pred)
        step_ll = _as_time_step_array(float(ll_val))
        step_hl = _as_time_step_array(float(hl_val))
        logger.inputs.append(step_input)
        logger.predicted_outputs.append(step_pred)
        logger.llcids.append(step_ll)
        logger.hlcids.append(step_hl)
        logger.prediction_losses.append(_as_time_step_array((pred - obs) ** 2))
        logger.latent_values.append(posterior.reshape(1, 1, -1).astype(np.float32))

    mse = float(np.mean((predictions - observations) ** 2))
    logger.others["mse"] = mse
    logger.others["predictions"] = predictions
    logger.others["posterior_history"] = posterior_history
    logger.others["bayesian_params"] = {
        "hazard_rate": hazard_rate,
        "observation_std": observation_std,
        "latent_values": latent_values,
    }

    return logger, mse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_bayesian_generalization(
    base_config: Optional[ContextualSwitchingTaskConfig] = None,
    seed: int = 0,
    hazard_rate: Optional[float] = None,
    hazard_grid: Optional[Sequence[float]] = None,
    test_types: Sequence[str] = ("ood_means", "block_size", "ood_stds"),
    values_override: Optional[Mapping[str, Sequence[Union[int, float]]]] = None,
    observation_noise_mode: Union[
        ObservationNoiseMode,
        Mapping[str, ObservationNoiseMode],
        Mapping[str, str],
    ] = "fixed",
    export_root: Path = DEFAULT_EXPORT_ROOT,
    run_name: str = "bayesian_observer",
    save_artifacts: bool = True,
) -> BayesianRunArtifacts:
    """Run the Bayesian observer on training data and requested generalisation suites.

    Args:
        observation_noise_mode: Strategy for supplying the observation noise during
            generalisation. ``"fixed"`` reuses the training distribution's
            ``default_std`` for every sweep. ``"ground_truth"`` feeds the dataset's
            own noise level to the observer (currently implemented for the
            ``"ood_stds"`` sweep). A mapping may be provided to select modes per
            test type (optionally including a ``"default"`` key).
    """

    config = deepcopy(base_config) if base_config is not None else ContextualSwitchingTaskConfig("figure")
    config.env_seed = seed
    train_dataset = TaskDataset(no_of_blocks=config.no_of_blocks, config=config)
    training_latent_values = np.array(config.training_data_means, dtype=float)

    observation_std = float(config.default_std)
    if hazard_rate is None:
        estimated = _estimate_hazard_rate(train_dataset.block_sizes)
        candidate_grid = list(hazard_grid) if hazard_grid is not None else []
        if estimated not in candidate_grid:
            candidate_grid.append(estimated)
        if not candidate_grid:
            candidate_grid = [estimated]
        hazard_rate, training_fit_mse = _fit_hazard_rate(train_dataset, observation_std, candidate_grid)
    else:
        training_fit_mse = float("nan")

    train_logger, training_mse = _simulate_bayesian_on_dataset(
        dataset=train_dataset,
        hazard_rate=hazard_rate,
        observation_std=observation_std,
        latent_values_override=training_latent_values,
    )
    train_logger.others["hazard_grid_fit_mse"] = training_fit_mse

    generalization_loggers: Dict[str, Dict[Union[int, float], Logger]] = {}
    values_per_test: Dict[str, Sequence[Union[int, float]]] = {}
    observation_modes_applied: Dict[str, str] = {}
    
    for test_type in test_types:
        seed_offset = 1
        sweep_values = (
            values_override[test_type]
            if values_override and test_type in values_override
            else _get_generalization_values(test_type, config)
        )
        mode_for_test = _resolve_observation_noise_mode(observation_noise_mode, test_type)
        observation_modes_applied[test_type] = mode_for_test
        values_per_test[test_type] = list(sweep_values)
        results_for_test: Dict[Union[int, float], Logger] = {}

        for value in sweep_values:
            dataset = _prepare_dataset_for_test(config, test_type, value, seed_offset=seed_offset)
            value_key: Union[int, float]
            if isinstance(value, (np.integer, int)):
                value_key = int(value)
            elif isinstance(value, (np.floating, float)):
                value_key = float(value)
            else:
                value_key = float(value)
            observation_std_for_run = observation_std # training dataset std by default
            if mode_for_test == "ground_truth":
                if test_type == "ood_stds":
                    observation_std_for_run = float(dataset.default_std) # Ground truth std from the testing dataset.
                else:
                    # Placeholder for future extensions when ground-truth values
                    # are defined for additional test types.
                    observation_std_for_run = observation_std
            logger, mse = _simulate_bayesian_on_dataset(
                dataset=dataset,
                hazard_rate=hazard_rate,
                observation_std=observation_std_for_run,
                latent_values_override=training_latent_values,
            )
            logger.others["mse"] = mse
            logger.others["observation_noise_mode"] = mode_for_test
            if mode_for_test == "ground_truth" and test_type == "ood_stds":
                logger.others["observation_noise_ground_truth_std"] = float(dataset.default_std)
            results_for_test[value_key] = logger

        generalization_loggers[test_type] = results_for_test

    metadata: Dict[str, Any] = {
        "hazard_rate": hazard_rate,
        "training_mse": training_mse,
        "training_hazard_fit_mse": training_fit_mse,
        "observation_std": observation_std,
        "seed": seed,
        "test_types": tuple(test_types),
        "values_per_test": values_per_test,
        "observation_noise_mode": dict(observation_noise_mode) if isinstance(observation_noise_mode, MappingABC) else observation_noise_mode,
        "observation_noise_mode_per_test": observation_modes_applied,
    }

    artifacts = BayesianRunArtifacts(
        train_logger=train_logger,
        generalization_loggers=generalization_loggers,
        hazard_rate=hazard_rate,
        training_mse=training_mse,
        metadata=metadata,
    )

    if save_artifacts:
        _save_artifacts(artifacts, run_name=run_name, seed=seed, export_root=export_root)

    return artifacts


def _save_artifacts(
    artifacts: BayesianRunArtifacts,
    run_name: str,
    seed: int,
    export_root: Path = DEFAULT_EXPORT_ROOT,
) -> Path:
    """Persist artefacts to disk and return the destination directory."""

    export_root = Path(export_root)
    run_dir = export_root / run_name / f"hazard-{artifacts.hazard_rate:.4f}_seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = artifacts.as_pickle_payload()
    output_path = run_dir / "results_bayesian_observer.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return run_dir


if __name__ == "__main__":
    # Allow quick manual runs when the module is executed directly.
    run_bayesian_generalization(save_artifacts=True)
