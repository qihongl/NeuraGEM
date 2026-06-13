"""Run adaptation experiments for both RNN and NeuraGEM models.

The script keeps the original training and evaluation flow while clarifying how
experiments are specified and executed. Experiments are described via data
structures so that shared configuration lives in one place and the three
experiment suites can be queued in a single invocation.
"""

from __future__ import annotations

import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from train_and_infer_functions import train_model, run_generalized_tests
from functions_and_utils import plot_logger_panels
from configs import ContextualSwitchingTaskConfig

import plot_style

plot_style.set_plot_style()


@dataclass(frozen=True)
class ModelConfigSpec:
    """Configuration for sweeping a single model's parameters."""

    param_grid: Dict[str, Sequence[Any]]
    pass_params_to_testing_phase: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    """Definition of an experiment that may span multiple models."""

    run_name: str
    ood_test_type: str
    models: Dict[str, ModelConfigSpec]
    train_overrides: Dict[str, Any] | None = None
    test_overrides: Dict[str, Any] | None = None
    weights_frozen: bool = True


@dataclass(frozen=True)
class ExperimentJob:
    """Concrete job to execute, derived from an experiment specification."""

    run_name: str
    ood_test_type: str
    model_name: str
    param_combination: Dict[str, Any]
    pass_params_to_testing_phase: bool
    train_overrides: Dict[str, Any] | None
    test_overrides: Dict[str, Any] | None
    weights_frozen: bool


DEFAULT_SEED_COUNT = 50
DEFAULT_MODELS: Tuple[str, ...] = ("rnn", "neuragem")

BASE_PARAM_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "rnn": {
        "default_std": [0.3, 0.4],
        "seq_len": [5, 50],
        "WU_lr": [1e-3],
    },
    "neuragem": {
        "default_std": [0.3, 0.4],
        "WU_lr": [1e-4],
        "l2_loss": [0.0008],
    },
}

BASE_TRAIN_OVERRIDES: Dict[str, Any] = {
    "blocked_phase_length": 1000,
    "start_always_on_the_same_block": False,
}


def build_model_param_grid(
    model_name: str,
    seed_count: int,
    overrides: Dict[str, Sequence[Any]] | None = None,
) -> Dict[str, Sequence[Any]]:
    """Return a fresh parameter grid for the requested model."""

    if model_name not in BASE_PARAM_GRIDS:
        raise KeyError(f"Unknown model '{model_name}'")

    grid = deepcopy(BASE_PARAM_GRIDS[model_name])
    if overrides:
        for key, values in overrides.items():
            grid[key] = values
    grid["seed"] = list(range(seed_count))
    return grid


# Experiments to run in a single invocation. Extend or edit this list to control
# which experiment suites get executed.
EXPERIMENT_SPECS: List[ExperimentSpec] = [
    ExperimentSpec(
        run_name="new_runs_50_stds",
        ood_test_type="ood_stds",
        models={
            "rnn": ModelConfigSpec(
                param_grid=build_model_param_grid(
                    "rnn", DEFAULT_SEED_COUNT, overrides={"default_std": [0.3]}
                ),
            ),
            "neuragem": ModelConfigSpec(
                param_grid=build_model_param_grid(
                    "neuragem",
                    DEFAULT_SEED_COUNT,
                    overrides={"default_std": [0.3], "l2_loss": [0.0001]},
                ),
            ),
        },
        train_overrides=dict(BASE_TRAIN_OVERRIDES),
        test_overrides=None,
    ),
    ExperimentSpec(
        run_name="new_runs_50_means",
        ood_test_type="ood_means",
        models={
            "rnn": ModelConfigSpec(
                param_grid=build_model_param_grid("rnn", DEFAULT_SEED_COUNT),
            ),
            "neuragem": ModelConfigSpec(
                param_grid=build_model_param_grid("neuragem", DEFAULT_SEED_COUNT),
            ),
        },
        train_overrides=dict(BASE_TRAIN_OVERRIDES),
        test_overrides={"l2_loss": 0.0001},
    ),
    ExperimentSpec(
        run_name="new_runs_50_block_size",
        ood_test_type="block_size",
        models={
            "rnn": ModelConfigSpec(
                param_grid=build_model_param_grid("rnn", DEFAULT_SEED_COUNT),
            ),
            "neuragem": ModelConfigSpec(
                param_grid=build_model_param_grid(
                    "neuragem", DEFAULT_SEED_COUNT, overrides={"l2_loss": [0.0001]}
                ),
            ),
        },
        train_overrides=dict(BASE_TRAIN_OVERRIDES),
        test_overrides={"l2_loss": 0.0009},
    ),
]


def save_results(filename, data, export_path):
    os.makedirs(export_path, exist_ok=True)
    with open(os.path.join(export_path, filename), "wb") as f:
        pickle.dump(data, f)


def load_results(filename, export_path):
    filepath = os.path.join(export_path, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded results from file: {filename}")
        return data
    else:
        print(f"ERROR: File does NOT exist {filepath}.")
        return None


def generate_param_combinations(param_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid into concrete combinations."""

    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in product(*param_grid.values())]


def run_single_experiment(
    model_name: str,
    param_combination: Dict[str, Any],
    seed: int,
    weights_frozen: bool,
    ood_test_type: str,
    train_overrides: Dict[str, Any] | None,
    test_overrides: Dict[str, Any] | None,
    pass_params_to_testing_phase: bool = False,
):
    """
    Runs one experiment for a given model and parameter combination.
    Returns a tuple: (training logger, testing logger dictionary).
    """

    config = ContextualSwitchingTaskConfig(experiment_to_run="figure")
    for param, value in param_combination.items():
        if not pass_params_to_testing_phase:
            if param == "seed":
                continue
            setattr(config, param, value)
    config.env_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if train_overrides is not None:
        for key, value in train_overrides.items():
            setattr(config, key, value)

    if model_name == "neuragem":
        pass
    else:  # for rnn
        config.no_of_steps_in_latent_space = 0

    train_logger, model, config, _ = train_model(config, seed=seed, run_test_phase=False)

    if pass_params_to_testing_phase:
        for param, value in param_combination.items():
            if param == "seed":
                continue
            setattr(config, param, value)
    if test_overrides is not None:
        for key, value in test_overrides.items():
            setattr(config, key, value)
    model.config = config
    model.LU_optimizer = model.get_LU_optimizer()  # update optimizer after config changes

    config.env_seed = seed + 1
    test_logger = run_generalized_tests(
        model, config, weights_frozen=weights_frozen, test_type=ood_test_type
    )

    return train_logger, test_logger


def _check_override_conflicts(
    model_name: str,
    param_grid: Dict[str, Sequence[Any]],
    overrides: Dict[str, Any] | None,
    stage: str,
) -> None:
    if overrides is None:
        return
    conflicting = [
        key for key in param_grid.keys() if key != "seed" and key in overrides
    ]
    if conflicting:
        raise ValueError(
            f"Parameters {conflicting} appear in both the sweep grid and {stage} overrides for"
            f" model '{model_name}'."
        )


def build_experiment_jobs(
    experiment_specs: Sequence[ExperimentSpec],
    models_to_run: Iterable[str] = DEFAULT_MODELS,
) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    selected_models = tuple(models_to_run)

    for exp_spec in experiment_specs:
        for model_name in selected_models:
            if model_name not in exp_spec.models:
                continue
            model_spec = exp_spec.models[model_name]
            _check_override_conflicts(
                model_name, model_spec.param_grid, exp_spec.train_overrides, "training"
            )
            if model_spec.pass_params_to_testing_phase:
                _check_override_conflicts(
                    model_name, model_spec.param_grid, exp_spec.test_overrides, "testing"
                )
            param_combinations = generate_param_combinations(model_spec.param_grid)
            print(
                f"Experiment {exp_spec.run_name} ({exp_spec.ood_test_type}) - {model_name}:"
                f" {len(param_combinations)} combinations"
            )
            for combination in param_combinations:
                jobs.append(
                    ExperimentJob(
                        run_name=exp_spec.run_name,
                        ood_test_type=exp_spec.ood_test_type,
                        model_name=model_name,
                        param_combination=combination,
                        pass_params_to_testing_phase=model_spec.pass_params_to_testing_phase,
                        train_overrides=exp_spec.train_overrides,
                        test_overrides=exp_spec.test_overrides,
                        weights_frozen=exp_spec.weights_frozen,
                    )
                )

    return jobs


def execute_experiment_job(job: ExperimentJob, job_index: int | None = None, total_jobs: int | None = None):
    seed = job.param_combination.get("seed", 0)
    filtered_params = {k: v for k, v in job.param_combination.items() if k != "seed"}
    combination_key = "_".join([f"{k}-{v}" for k, v in sorted(filtered_params.items())])
    export_path = os.path.join(
        "./exports/contextual_switching_task/experiments", job.run_name, combination_key
    )
    os.makedirs(export_path, exist_ok=True)

    progress_prefix = ""
    if job_index is not None and total_jobs is not None:
        progress_prefix = f"[{job_index + 1}/{total_jobs}] "

    print(
        f"{progress_prefix}Running model: {job.model_name} | run: {job.run_name}"
        f" | params: {job.param_combination}"
    )
    train_logger, test_logger = run_single_experiment(
        job.model_name,
        job.param_combination,
        seed,
        job.weights_frozen,
        job.ood_test_type,
        job.train_overrides,
        job.test_overrides,
        pass_params_to_testing_phase=job.pass_params_to_testing_phase,
    )

    filename = (
        f"results_{job.model_name}_frozen_{job.weights_frozen}_{combination_key}_seed-{seed}.pkl"
    )
    save_results(filename, {"train_logger": train_logger, "test_logger": test_logger}, export_path)
    print(f"Saved results to {export_path} / {filename}\n")


def main() -> None:
    jobs = build_experiment_jobs(EXPERIMENT_SPECS, models_to_run=DEFAULT_MODELS)
    total_jobs = len(jobs)
    if not jobs:
        raise RuntimeError("No experiments to run. Check experiment specification.")
    print(f"Prepared {total_jobs} experiment combination(s).")

    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print(
            "SLURM_ARRAY_TASK_ID not set; running all"
            f" {total_jobs} job(s) sequentially."
        )
        for idx, job in enumerate(jobs):
            execute_experiment_job(job, job_index=idx, total_jobs=total_jobs)
    else:
        task_id = int(task_id_str)
        if task_id < 0 or task_id >= total_jobs:
            raise ValueError(
                f"Task id {task_id} is out of range. There are only {total_jobs} experiments."
            )
        execute_experiment_job(jobs[task_id], job_index=task_id, total_jobs=total_jobs)


if __name__ == "__main__":
    main()


# %% Uncomment below to visualize some results in an interactive session.
# plot_logger_panels(train_logger, train_logger.config, ['behavior'])
#
# for key in test_logger.keys():
#     fig = plot_logger_panels(test_logger[key], train_logger.config, ['behavior'])
#     fig.axes[0].set_title(key)
