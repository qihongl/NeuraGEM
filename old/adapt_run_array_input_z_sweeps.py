'''
I readapted the code to run the various OOD tests for the contextual switching task.
This reuses the OOD_means script and params to now run input z sweeps.
The main focus on to describe the responses of the model across different 
z and input values. I use it to explore the model's behavior
with multiplicative gating vs additive input.

'''
#%%
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from torch.utils.data import DataLoader
from train_and_infer_functions import train_model, run_generalized_tests
from functions_and_utils import plot_logger_panels, explore_data_container
from functions_and_utils_2 import *
from configs import ContextualSwitchingTaskConfig

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


def run_single_experiment(model_name, param_combination, seed, weights_frozen,
                          ood_test_type, config_overrides, config_overrides_2,
                          pass_params_to_testing_phase=False):
    """
    Runs one experiment for a given model and parameter combination.
    Returns a tuple: (training logger, testing logger dictionary).
    """
    # Set up configuration
    config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
    for param, value in param_combination.items():
        if not pass_params_to_testing_phase:
            if param == "seed":
                continue
            setattr(config, param, value)
        config.use_mul_gating = not config.use_add_gating  # use multiplicative gating if not using additive gating
        config.input_size = 3 if config.use_add_gating else 1  # input plus latent dim. Should only be 3 for add gating else 1
        # config.post_gating = True # use to test multiplicative post gating instead. 
        # config.pre_gating = not config.post_gating
        
    # Apply additional configuration overrides
    if config_overrides is not None:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    # Model-specific configuration updates
    if model_name == 'neuragem':
        # config.WU_lr = 1e-4
        pass
    else:  # for rnn
        config.no_of_steps_in_latent_space = 0

    # Train the model
            
    train_logger, model, config, _ = train_model(config, seed=seed, run_test_phase=config.run_test_phase,)
    if config.run_test_phase: 
        config.no_of_steps_in_latent_space = 1  # reset to 1 because last testing turned it off.
    
    # If parameters should be passed to the testing phase, update config here
    if pass_params_to_testing_phase:
        for param, value in param_combination.items():
            if param == "seed":
                continue
            setattr(config, param, value)
    if config_overrides_2 is not None:
        for key, value in config_overrides_2.items():
            setattr(config, key, value)
    model.config = config
    model.LU_optimizer = model.get_LU_optimizer()  # update optimizer after config changes

    # Run tests (using generalized tests here)
    # config.env_seed should change so that the ood test is not the same as the training seed
    config.env_seed = seed + 1 # This had no effect. Keeping just in case a model ever memorizes training data.
    if config.run_input_sweep:
        test_loggers_by_z = {}
        config.no_of_steps_in_latent_space = 0 # Z is set manually in the input sweep test.
        model.config.no_of_steps_in_latent_space = 0 # Z is set manually in the input sweep test.
        # Run input sweep tests for different z values
        z_values = np.linspace(-2, 2, 10).round(1)
        for z in z_values:
            latent = torch.zeros((config.batch_size, config.seq_len, config.latent_dims[0]), dtype=torch.float32).to(config.device)
            latent[:, :, 0] = z
            latent[:, :, 1] = -z

            model.set_latent(latent)
            print(f"Running input sweep test with z={z}")
            test_logger = run_generalized_tests(model, config, weights_frozen=weights_frozen, test_type=ood_test_type)
            test_loggers_by_z[z] = test_logger
        test_logger = test_loggers_by_z    
    elif config.run_lr_sweep:
        # Run learning rate sweep tests
        lr_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        test_loggers_by_lr = {}
        for lr in lr_values:
            model.LU_optimizer.param_groups[0]['lr'] = lr
            print(f"Running learning rate sweep test with lr={lr}")
            test_logger = run_generalized_tests(model, config, weights_frozen=weights_frozen, test_type=ood_test_type)
            test_loggers_by_lr[lr] = test_logger
        test_logger = test_loggers_by_lr

    else:
        test_logger = run_generalized_tests(model, config, weights_frozen=weights_frozen, test_type=ood_test_type)

    return train_logger, test_logger


# Define experiment configuration for each model
# #############################################################################
#### This can run either sweeping over z or lr_z to switch use the config overrides  
# 'run_input_sweep': False,'run_lr_sweep': True,
#################################################################################

run_name= 'pca_ood_means_final_check' 
# run_name= 'pca_ood_means_post_gating' 


ood_test_type = 'ood_means'
# ood_test_type = 'block_size'

# Select which models to run. You can choose one or both.
models = ['rnn', 'neuragem']

if run_name== 'pca_ood_means' :

    model_configs = {
        'rnn': {
            'run_name': run_name,
            'ood_test_type': ood_test_type,
            'param_grid': {
                'default_std': [0.1,  0.3, 0.4],
                'seq_len': [5, 50],
                'WU_lr': [1e-3],
                'seed': list(range(20)) # 3 * 2 * 20 = 120
            },
            'pass_params_to_testing_phase': False
        },
        'neuragem': {
            'run_name': run_name,
            'ood_test_type': ood_test_type,
            'param_grid': {
                'default_std': [0.1, 0.3, 0.4],
                'WU_lr': [1e-4, 1e-3],
                'l2_loss': [ 0.0008,],
                # 'l2_loss': [0.0008], #[0.00005, 0.0001, 0.0009],
                'seed': list(range(20)), # 3 * 2 *  4 * 20 = 480
                'use_add_gating': [True, False], # True for additive gating, False for multiplicative gating
            },
            'pass_params_to_testing_phase': False
        }
    }

    config_overrides = {
    # 'default_std': 0.1,
    'blocked_phase_length': 1000,
    'start_always_on_the_same_block': False,
    # 'WU_lr': 1e-3,  # note: neuragem-specific update is done in run_single_experiment
    # 'l2_loss': 0.0008,
    'run_input_sweep': True,
    'run_lr_sweep': False,
    'log_hidden_states': True, # Log hidden states for analysis
    'log_end_weights': True, # Log end weights for analysis

    }
else:
    model_configs = {
        'rnn': {
            'run_name': run_name,
            'ood_test_type': ood_test_type,
            'param_grid': {
                'default_std': [0.1,  0.3, 0.4],
                'seq_len': [5, 50],
                'WU_lr': [1e-3],
                'seed': list(range(20)) # 3 * 2 * 20 = 120
            },
            'pass_params_to_testing_phase': False
        },
        'neuragem': {
            'run_name': run_name,
            'ood_test_type': ood_test_type,
            'param_grid': {
                'default_std': [0.1, 0.3, 0.4],
                'WU_lr': [1e-4],
                'l2_loss': [ 0.0008,],
                # 'l2_loss': [0.0008], #[0.00005, 0.0001, 0.0009],
                'seed': list(range(20)), # 3 * 1 *  4 * 20 = 240
                'use_add_gating': [True, False], # True for additive gating, False for multiplicative gating
            },
            'pass_params_to_testing_phase': False
        }
    }


    # Global config overrides that apply to both models.
    # These overrides should not contain any key that is in the model-specific param_grid (except 'seed').
    config_overrides = {
        # 'default_std': 0.1,
        'blocked_phase_length': 1000,#4000,
        # 'WU_lr': 1e-3,  # note: neuragem-specific update is done in run_single_experiment
        # 'l2_loss': 0.0008,
        'run_input_sweep': False,
        'run_lr_sweep': False,
        'run_block_size_and_means_sweep': True,  # Set to True to run block size and means sweep
        'run_test_phase': True,  # Run the test phase after training
        'log_weights': True, # logg all weights during training!
        'log_hidden_states': True, # Log hidden states for analysis
        'log_end_weights': True, # Log end weights for analysis

    }

config_overrides_2 = {'l2_loss': 0.0001,}

no_of_experiments_to_run = 0
for model_name in models:
    mconf = model_configs[model_name]
    param_combinations = generate_param_combinations(mconf['param_grid'])
    no_of_experiments_to_run += len(param_combinations)
print(f"Total number of experiments to run: {no_of_experiments_to_run}")
#%%
weights_frozen = True

# Build experiments list for Slurm array compatibility.
# Each experiment is a tuple: (model_name, run_name, ood_test_type, pass_params_to_testing_phase, param_combination)
experiments = []
for model_name in models:
    mconf = model_configs[model_name]
    # Check that none of the keys in param_grid (except 'seed') appear in config_overrides.
    config_overrides_to_process = config_overrides_2 if mconf['pass_params_to_testing_phase'] else config_overrides
    for param in mconf['param_grid']:
        if config_overrides_to_process is not None:
            if param != "seed" and param in config_overrides_to_process:
                raise ValueError(f"{param} is in config_overrides. Please remove it for model {model_name}.")
    param_combinations = generate_param_combinations(mconf['param_grid'])
    print (f"Model: {model_name}, param_combinations: {len(param_combinations)}")
    for param_combination in param_combinations:
        experiments.append((model_name,
                            mconf['run_name'],
                            mconf['ood_test_type'],
                            mconf['pass_params_to_testing_phase'],
                            param_combination))

# Use the SLURM array task ID to select the experiment.
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
if task_id >= len(experiments):
    raise ValueError(f"Task id {task_id} is out of range. There are only {len(experiments)} experiments.")

model_name, run_name, ood_test_type, pass_params_to_testing_phase, param_combination = experiments[task_id]
seed = param_combination.get("seed", 0)
# Create folder name excluding seed
filtered_params = {k: v for k, v in param_combination.items() if k != "seed"}
combination_key = "_".join([f"{k}-{v}" for k, v in sorted(filtered_params.items())])
export_path = os.path.join('./exports/contextual_switching_task/experiments',
                            run_name,
                            combination_key)
os.makedirs(export_path, exist_ok=True)

print(f"Running model: {model_name} with parameters: {param_combination}")
train_logger, test_logger = run_single_experiment(
    model_name, param_combination, seed, weights_frozen, ood_test_type,
    config_overrides, config_overrides_2, pass_params_to_testing_phase=pass_params_to_testing_phase,
)

filename = f'results_{model_name}_frozen_{weights_frozen}_{combination_key}_seed-{seed}.pkl'
save_results(filename, {"train_logger": train_logger, "test_logger": test_logger}, export_path)
print(f"Saved results to folder {export_path} \nfilename: {filename}")

#%% Uncomment below to visualize some results in an interactive session.
# plot_logger_panels(train_logger, train_logger.config, ['behavior'])
#
# for key in test_logger.keys():
#     fig = plot_logger_panels(test_logger[key], train_logger.config, ['behavior'])
#     fig.axes[0].set_title(key)
