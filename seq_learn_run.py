''' This runs the seqlearning task to explore lower seq_len. 
Recently I found that lowering seq_len to 6 or 10 is possible with a small increase to LU_lr from 0.1
to 0.2.. also can use only 1 LU step. I will explore here integrating that in the main experiments. 
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
from seq_learn_config import run_name, get_base_params, apply_neuragem_overrides


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
    config = seq_learnConfig(experiment_to_run='few_long_blocks')    
    if model_name == 'neuragem':
        # config.blocked_phase_length = 1500 # there is an override on this. 
        apply_neuragem_overrides(config, param_combination)


    torch.manual_seed(seed)
    np.random.seed(seed)
    config.env_seed = seed

    # apply global overrides
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # set all other params up front (except seed)
    for p, v in param_combination.items():
        if p == 'seed': continue
        setattr(config, p, v)

    # 2) pull out curriculum‐phase flags and other params
    curriculum = param_combination.get('curriculum', curriculum)
    if curriculum == 'interleaved':
        config.add_interleaved_phase = True
        config.add_blocked_phase = False
    elif curriculum == 'blocked':
        config.add_interleaved_phase = False
        config.add_blocked_phase = True
    elif curriculum in ['interleaved_blocked', 'blocked_interleaved']:
        config.add_interleaved_phase = True
        config.add_blocked_phase = True
        # config.blocked_phase_length //= 2 if model_name!='neuragem' else 1200  # half blocked

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
        if curriculum in ['interleaved_blocked', 'blocked_interleaved']: 
            config.interleaved_phase_length = int(config.interleaved_phase_length / 1.3) # dividing by 2 not enough for mrnn to learn either of these tasks
            config.blocked_phase_length = int(config.blocked_phase_length / 1.2)
            
        else:
            config.interleaved_phase_length //= 2
            config.blocked_phase_length //= 2

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
    # Testing phase (Changed to random testing to match the human experiment)
    ##############################
    testing_curriculum = 'random' # 'same_as_last_curriculum' or 'random'
    if model_name == 'mrnn': # because the input horizon is so long it eats up most of the 240 ts testing phase as a quick fix, log in the initial timesteps, before I would exclude them as 'burn-in'
        config.log_initial_burn_in_timesteps = True
    logger.log_phase('Testing\n(W frozen)')
    config.no_of_steps_in_weight_space = 0
    testing_phase_length = 40 * config.task_length
    if testing_curriculum == 'same_as_last_curriculum':
        print(f"Testing with the same curriculum as training: {curriculum}")
        if curriculum == ['interleaved', 'blocked_interleaved']: # continue testing with whatever was the current curriculum.
            config.block_size = 1 * config.task_length
        elif curriculum in ['blocked', 'interleaved_blocked', ]:
            config.block_size = stored_block_size
    else:       
        print(f"Testing with a random curriculum (different from training)")
        config.shuffle_or_interleave = 'random' # Contexts will appear randomly during testing instead of in interleaved
        config.block_size = 1 * config.task_length # sample context every trial

        # Adjustments to NeuraGEM to accommodate random testing curriculum as described in Methods 3.2
        if model_name == 'neuragem':
            config.LU_lr = 0.4 # from 0.2
            config.no_of_steps_in_latent_space = 5
            config.l2_loss = 0.0004 # from 0.00004


    config.no_of_blocks = int(testing_phase_length / config.block_size)
    _, _, dataloader, _ = create_datasets_and_loaders(config)
    predictive_learning(logger, config, dataloader, model)
    config.block_size = stored_block_size

    return logger

if __name__ == "__main__":
    # models = ['rnn', 'mrnn', 'neuragem']
    # models = ['rnn', 'neuragem']
    # models = ['mrnn',]
    models = ['neuragem',]

    # the curricula you want to sweep over:
    # curricula = ['blocked', 'interleaved', 'interleaved_blocked', ]
    # 'blocked_interleaved' ]
    curricula = ['interleaved']
    # curricula = ['interleaved_blocked', 'blocked', 'interleaved', ]

    # global overrides
    config_overrides = {
        'start_always_on_the_same_block': False,
        'add_passive_learning_phase': False,
    }
    config_overrides_2 = None

    no_of_seeds = 20  # number of seeds to sweep over

    # base params by model and curriculum
    base_params = get_base_params()
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
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if task_id >= len(experiments):
        raise ValueError(f"Task id {task_id} is out of range (0..{len(experiments)-1}).")

    model_name, run_name, curriculum, param_combination = experiments[task_id]
    print(f'Running experiment: {task_id} out of a total of {len(experiments)} experiments to run', )
    seed = param_combination['seed']

    # make a folder name excluding seed
    filtered = {k: v for k, v in param_combination.items() if k != 'seed'}
    combination_key = "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))
    export_path = os.path.join('./exports/seq_learn/experiments', run_name, combination_key)
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

    plot_logger = True
    if plot_logger:
        panel_order = ['corrects', 'latent_2d', 'gradients']
        fig = plot_logger_panels(
            logger, logger.config, panel_order, subplot_height=1.5, annotate_phases='corrects')
