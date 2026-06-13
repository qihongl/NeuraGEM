'''
This is meant to also run csw task but restricted to
exploring neuragem model with different hyperparameters.

Then use csw_run_array_neuragem.py script is to plot

'''

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from train_and_infer_functions import run_generalized_tests, predictive_learning
from functions_and_utils import *
from train_and_infer_functions import *
from datasets import *
from functions_and_utils_2 import *
from configs import ContextualSwitchingTaskConfig
from models import *
from datasets import create_datasets_and_loaders


import plot_style
plot_style.set_plot_style()


def save_results(filename, data, export_path):
    os.makedirs(export_path, exist_ok=True)
    with open(os.path.join(export_path, filename), 'wb') as f:
        pickle.dump(data, f)


def load_results(filename, export_path):
    filepath = os.path.join(export_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded results from file: {filename}")
        return data
    else:
        print(f"ERROR: File does NOT exist {filepath}.")
        return None


def generate_param_combinations(param_grid):
    """
    Given a dictionary mapping parameter names to lists of values,
    generate a list of dictionaries representing every combination.
    """
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    return combinations


def run_single_experiment(model_name,
                          param_combination,
                          seed,
                          curriculum,
                          config_overrides,
                          config_overrides_2,
                          pass_params_to_specific_phase=False):
    """
    Runs one experiment for a given model and parameter combination,
    with controlled curriculum phases: passive → interleaved → blocked.
    Uses predictive_learning() and create_datasets() so that a single
    Logger is passed through all phases and accumulates all logs.
    Returns a tuple: (dict of training loggers by phase, testing logger).
    """
    # 1) set up base config and seeds
    config = CSWConfig(experiment_to_run='few_long_blocks')    
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    config.env_seed = seed


    # 2) pull out curriculum‐phase flags and other params
    curriculum = param_combination.get('curriculum', curriculum)
    if curriculum == 'interleaved':
        config.add_interleaved_phase = True
        config.add_blocked_phase = False
    elif curriculum == 'blocked':
        config.add_interleaved_phase = False
        config.add_blocked_phase = True
    elif curriculum == 'interleaved_blocked':
        config.add_interleaved_phase = True
        config.add_blocked_phase = True
        # gonna be controlling these directly here.
        # config.interleaved_phase_length //= 2  # half interleaved
        # config.blocked_phase_length //= 2 if model_name != 'neuragem' else 1200/1000  # half blocked
    elif curriculum == 'blocked_interleaved':
        config.add_interleaved_phase = True
        config.add_blocked_phase = True

    # apply global overrides
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # set all other params up front (except seed)
    for p, v in param_combination.items():
        if p == 'seed': continue
        setattr(config, p, v)

    # model‐specific tweaks
    if model_name in ['rnn', 'mrnn']:
        config.no_of_steps_in_latent_space = 0
    if model_name == 'mrnn':
        # shorten duration, mrnn gets updated for each timestep in its horizon every timestep of the task.
        if not curriculum == 'interleaved_blocked': 
            config.interleaved_phase_length //= 2
            config.blocked_phase_length //= 2
        else:
            config.interleaved_phase_length //= 1.3 # lower because was already lowered above
            config.blocked_phase_length //= 1.2
    # neuragem can override WU_lr inside train_model or via config_overrides

    # Initialize a logger and model (always use RNN_with_latent)
    logger = Logger()
    model = RNN_with_latent(config).to(config.device)
    stored_block_size = config.block_size # stored only because interleaved might change it to 1 temporarily.

    # 3) passive phase (no latent updates)
    if config.add_passive_learning_phase:
        print('Running passive phase')
        config._allow_latent_updates = False # this is a low level config to disable latent updates irrespective of other flags, e.g. LU_lr or no_of_steps_in_latent_space
        config.curriculum_phase = 'passive'
        config.no_of_blocks = int(config.passive_phase_length / config.block_size)
        _,_, dataloader, _ = create_datasets_and_loaders(config)
        predictive_learning(logger, config, dataloader, model,)

        logger = Logger()  # reset logger, nothing interesting in the passive phase

        ##################################### alternative logic to allow for 'blocked_interleaved'
        # 4) interleaved phase (if first)
    if config.add_interleaved_phase and curriculum != 'blocked_interleaved':
        print('Running interleaved phase')
        logger.log_phase('Interleaved\ntraining')
        config.block_size = config.task_length
        config.no_of_blocks = config.interleaved_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        config._allow_latent_updates = config.latent_updates_during_shuffle
        predictive_learning(logger, config, dataloader, model)

    # 5) blocked phase
    if curriculum in ['blocked_interleaved', 'interleaved_blocked']:
        config._allow_latent_updates = True
    else: # allow this to control the latent updates during the experiment if testing only 'blocked' or 'interleaved'
        config._allow_latent_updates = config.latent_updates_during_shuffle
    if config.add_blocked_phase:
        print('Running blocked phase')
        config.block_size = stored_block_size
        logger.log_phase('Blocked\ntraining')
        config.no_of_blocks = config.blocked_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        predictive_learning(logger, config, dataloader, model)

    # 6) interleaved phase (if second for blocked_interleaved)
    if curriculum == 'blocked_interleaved':
        print('Running second interleaved phase')
        logger.log_phase('Interleaved\ntraining\n(2nd)')
        config.block_size = config.task_length
        config.no_of_blocks = config.interleaved_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        config._allow_latent_updates = config.latent_updates_during_shuffle
        predictive_learning(logger, config, dataloader, model)

    # 6) prepare for testing
    if config_overrides_2:
        for k, v in config_overrides_2.items():
            setattr(config, k, v)

    model.config = config
    model.LU_optimizer = model.get_LU_optimizer()
    # bump seed so OOD test differs
    config.env_seed = seed + 1

    # 7) run final tests
    # Save additional info to logger for later analysis
    logger.config = config

    ##############################
    # Testing phase (with frozen weights)
    ##############################
    if model_name == 'mrnn': # because the input horizon is so long it eats up most of the 240 ts testing phase as a quick fix, log in the initial timesteps, before I would exclude them as 'burn-in'
        config.log_initial_burn_in_timesteps = True
    logger.log_phase('Testing\n(W frozen)')
    config.no_of_steps_in_weight_space = 0
    testing_phase_length = 40 * config.task_length
    if curriculum == ['interleaved', 'blocked_interleaved']: # continue testing with whatever was the current curriculum.
        config.block_size = 1 * config.task_length
    elif curriculum in ['blocked', 'interleaved_blocked', ]:
        config.block_size = stored_block_size
    config.no_of_blocks = int(testing_phase_length / config.block_size)
    _, _, dataloader, _ = create_datasets_and_loaders(config)
    predictive_learning(logger, config, dataloader, model)
    config.block_size = stored_block_size

    return logger

# run_name = 'initial_neuragem_runs'
run_name = 'v2_neuragem_runs'
# models = ['rnn', 'mrnn', 'neuragem']
models = ['neuragem',]

# the curricula you want to sweep over:
curricula = [ 'interleaved_blocked',]# 'interleaved',] # 'blocked_interleaved']#'blocked', 

# global overrides
config_overrides = {
    'start_always_on_the_same_block': False,
    'add_passive_learning_phase': False,
}
config_overrides_2 = None

no_of_seeds = 40
#%%
# base params by model and curriculum
base_params = {
    # 'rnn': {
    #     'blocked':             {'seq_len': [18]},
    #     'interleaved':         {'seq_len': [18]},
    #     'interleaved_blocked': {'seq_len': [18]},
    # },
    # 'mrnn': {
    #     'blocked':             {'seq_len': [200]},
    #     'interleaved':         {'seq_len': [200]},
    #     'interleaved_blocked': {'seq_len': [200]},
    # },
    'neuragem': {
        'blocked':             {'blocked_phase_length': [400, 600, 800, 1200], 'latent_updates_during_shuffle': [True, False],},
        'interleaved':         {'interleaved_phase_length': [400, 600, 800, 900], 'latent_updates_during_shuffle': [True, False],},
        'interleaved_blocked': {
            # already run:
            # 'interleaved_phase_length': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, ],
            'interleaved_phase_length': [1200, 1300,  1500, 1600 ],
            # 'blocked_phase_length': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1400],
            # 'interleaved_phase_length': [20, 30, 40, 1400, 1500, 1600],
            'blocked_phase_length': [1200,],
            # 'interleaved_phase_length': [500, 600, 700],
            'latent_updates_during_shuffle': [True, False]
        },
        'blocked_interleaved': {
            'blocked_phase_length': [1200],
            'interleaved_phase_length': [500,600, 700],
            'latent_updates_during_shuffle': [True, False]
        },
    },
}

# BUILD THE LIST OF EXPERIMENTS
experiments = []
for model_name in models:
    for curriculum in curricula:
        # pull the right grid
        param_grid = dict(base_params[model_name][curriculum])
        # always sweep seeds 0–19
        param_grid['seed'] = list(range(no_of_seeds))
        # generate
        for combo in generate_param_combinations(param_grid):
            # inject the curriculum into the combo so run_single_experiment picks it up
            combo['curriculum'] = curriculum
            experiments.append((
                model_name,
                run_name,
                curriculum,
                combo
            ))

# pick one via SLURM_ARRAY_TASK_ID (or 0)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 30))
if task_id >= len(experiments):
    raise ValueError(f"Task id {task_id} is out of range (0..{len(experiments)-1}).")

model_name, run_name, curriculum, param_combination = experiments[task_id]
print(f'Running experiment: {task_id} out of a total of {len(experiments)} experiments to run', )
seed = param_combination['seed']

#%%
# make a folder name excluding seed
filtered = {k: v for k, v in param_combination.items() if k != 'seed'}
combination_key = "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))
export_path = os.path.join('./exports/csw/experiments', run_name, combination_key)
os.makedirs(export_path, exist_ok=True)

print(f"Running {model_name} | curriculum={curriculum} | params={param_combination}")
logger = run_single_experiment(
    model_name,
    param_combination,
    seed,
    curriculum,
    config_overrides,
    config_overrides_2
)

filename = f"results_{model_name}_{combination_key}_seed-{seed}.pkl"
save_results(filename, logger, export_path)
print(f"Saved → {os.path.join(export_path, filename)}")

#%%
plot_logger = False
if plot_logger:
    panel_order = ['corrects', 'latent_2d', 'gradients']
    fig = plot_logger_panels(
        logger, logger.config, panel_order, subplot_height=1.5, annotate_phases='corrects')