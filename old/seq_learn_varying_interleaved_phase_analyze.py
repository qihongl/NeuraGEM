'''
This script digs into the interleaved_blocked and blocked_interleaved runs of NeuraGem and explores the parameter space more thoroughly.
It analyzes the results of the runs from csw_run_array_neuragem.py
Make sure the base_parameters grid is the same in both files
if changed so that it load the same experiments. 
Its main function is to produce the sweep over 
interleaved_phase_length and blocked_phase_length for the NeuraGem model.
'''

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from train_and_infer_functions import run_generalized_tests, predictive_learning
from functions_and_utils import *
from train_and_infer_functions import *
from datasets import *
from functions_and_utils_2 import *
from configs import SeqLearnConfig
from models import *
from datasets import create_datasets_and_loaders
from seq_learn_functions_and_utils import *
from collections import defaultdict

import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

def save_results(filename, data, export_path):
    os.makedirs(export_path, exist_ok=True)
    with open(os.path.join(export_path, filename), 'wb') as f:
        pickle.dump(data, f)

def generate_param_combinations(param_grid):
    """
    Given a dictionary mapping parameter names to lists of values,
    generate a list of dictionaries representing every combination.
    """
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    return combinations



#%%
if __name__ == "__main__":
    # models = ['rnn', 'mrnn', 'neuragem']
    models = ['neuragem',]

    # curricula want to explore
    # curricula = [ 'interleaved', 'interleaved_blocked', 'blocked_interleaved'] # 'blocked',
    curricula = [ 'interleaved', 'interleaved_blocked', 'blocked_interleaved'] # 'blocked',
    

    # global overrides
    config_overrides = {
        'start_always_on_the_same_block': False,
        'add_passive_learning_phase': False,
    }
    config_overrides_2 = None

    run_name = 'v2_neuragem_runs'

    if run_name == 'initial_neuragem_runs':
        no_of_seeds = 20 
        # base params by model and curriculum
        base_params = {

            'neuragem': {
                # 'blocked':             {'blocked_phase_length': [1200], 'latent_updates_during_shuffle': [True, False],},
                'interleaved':         {'interleaved_phase_length': [800], 'latent_updates_during_shuffle': [True, False],},
                'interleaved_blocked': {
                    'blocked_phase_length': [500, 600, 700, 800, 900, 1200],
                    # 'blocked_phase_length': [1000,1100,1400, ], # these also available. 
                    'interleaved_phase_length': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
                    # also blocked phase 1200 ahs these interleaved phase available +[20, 30, 40, 1400, 1500, 1600],

                    'latent_updates_during_shuffle': [True, False]
                },
                'blocked_interleaved': {
                    'blocked_phase_length': [1200],
                    'interleaved_phase_length': [400,500,600],
                    'latent_updates_during_shuffle': [True, False]
                },
            },
        }
    elif run_name == 'v2_neuragem_runs': # this is when it became clear to focus on blocked len 1200 and inter 500. Running more seeds
        no_of_seeds = 40 
        curricula = [  'interleaved_blocked', ] # 'blocked',
        base_params = {
            'neuragem': {
                'interleaved':         {'interleaved_phase_length': [800], 'latent_updates_during_shuffle': [True, False],},
                'interleaved_blocked': {
                    'blocked_phase_length': [1200],
                    'interleaved_phase_length': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
                    'latent_updates_during_shuffle': [True, False]
                },
                'blocked_interleaved': {
                    'interleaved_phase_length': [400,500,600],
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
            # always sweep seeds 
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
    # Collect results into lists for DataFrame construction
    results_list = []
    all_loggers = defaultdict(dict)
    missing_loggers = defaultdict(list)  # curriculum -> list of (combination_key, seed)

    for exp_id in range(len(experiments)):
        model_name, run_name, curriculum, param_combination = experiments[exp_id]
        seed = param_combination['seed']

        # make a folder name excluding seed
        filtered = {k: v for k, v in param_combination.items() if k != 'seed'}
        combination_key = "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))
        export_path = os.path.join('./exports/csw/experiments', run_name, combination_key)
        os.makedirs(export_path, exist_ok=True)

        # Only load the logger for the current seed
        filename = f"results_{model_name}_{combination_key}_seed-{seed}.pkl"
        logger = load_results(filename, export_path)
        if logger is None:
            missing_loggers[curriculum].append((combination_key, seed))
            continue
        # compute score
        score = compute_testing_score(logger, alpha=0.5)
        # Extract relevant parameters for DataFrame
        latent_updates = param_combination.get('latent_updates_during_shuffle', False)
        # record both phase lengths separately
        results_list.append({
            'curriculum': curriculum,
            'latent_updates': latent_updates,
            'blocked_phase_length': param_combination.get('blocked_phase_length', np.nan),
            'interleaved_phase_length': param_combination.get('interleaved_phase_length', np.nan),
            'seed': seed,
            'score': score,
            'combination_key': combination_key,
        })

    for curriculum, missing in missing_loggers.items():
        print(f"Missing loggers for curriculum '{curriculum}': {len(missing)} runs not found.")

    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    print(results_df.head())

#%%
# Violin plot of scores for a single interleaved phase length
# Example usage:
if 'blocked' in results_df['curriculum'].unique():
    plot_violin_for_param(results_df, 'blocked', param_name='phase_length', param_value=1200)

if 'interleaved' in results_df['curriculum'].unique():
    plot_violin_for_param(results_df, 'interleaved', param_name='interleaved_phase_length', param_value=800)

# Violin plots for all interleaved phase lengths, stacked vertically
# interleaved_lengths =  [100, 300, 800, 1000, 1300]
# not sure why the below comes up with more values than is available 
interleaved_lengths =  results_df['interleaved_phase_length'].dropna().unique()

# fix blocked phase length for these plots
fixed_blocked_phase_length = 1200
df_ib = results_df[
    (results_df['curriculum'] == 'interleaved_blocked') &
    (results_df['blocked_phase_length'] == fixed_blocked_phase_length)
]

n_plots = len(interleaved_lengths)
fig, axes = plt.subplots(
    n_plots,
    1,
    figsize=(cs.panel_small_size[0], cs.panel_small_size[1] * n_plots),
    sharex=True
)

# Only plot for interleaved_phase_length values that are present in df_ib
available_lengths = []
missing_lengths = []
for pl in interleaved_lengths:
    if ((df_ib['interleaved_phase_length'] == pl).any()):
        available_lengths.append(pl)
    else:
        missing_lengths.append(pl)

if missing_lengths:
    print(f"Warning: The following interleaved_phase_length values are not present in the filtered DataFrame and will be skipped: {missing_lengths}")

if len(available_lengths) < len(interleaved_lengths):
    print(f"Plotting only {len(available_lengths)} out of {len(interleaved_lengths)} requested interleaved_phase_length values.")

# Adjust axes if fewer available
if len(available_lengths) < len(axes):
    # Hide unused axes if any
    for ax in axes[len(available_lengths):]:
        ax.set_visible(False)
    axes = axes[:len(available_lengths)]
available_lengths = sorted(available_lengths)  # Sort for consistent order in plots
for ax, pl in zip(axes, available_lengths):
    plot_violin_for_param(
        df_ib,
        curriculum='interleaved_blocked',
        param_name='interleaved_phase_length',
        param_value=pl,
        ax=ax,
        title=f'b length={fixed_blocked_phase_length}, inter={pl}'
    )


#%%

# === New analysis: accuracy vs interleaved_phase_length at fixed blocked_phase_length ===
import scipy.stats as st

# filter for interleaved_blocked
data_ib = results_df[results_df['curriculum'] == 'interleaved_blocked']
# get the list of blocked_phase_length values from your config
blocked_lengths = base_params['neuragem']['interleaved_blocked']['blocked_phase_length']

for blocked_len in blocked_lengths:
    subset = data_ib[data_ib['blocked_phase_length'] == blocked_len]
    if subset.empty:
        continue

    # compute mean and SEM per interleaved_phase_length & latent_updates
    stats = (
        subset
        .groupby(['interleaved_phase_length', 'latent_updates'])['score']
        .agg(['mean', lambda x: st.sem(x, nan_policy='omit')])
        .rename(columns={'<lambda_0>':'sem'})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=cs.panel_small_size)

    # raw points for latent_updates=True
    # raw = subset[subset['latent_updates'] == True]
    # ax.scatter(
    #     raw['interleaved_phase_length'],
    #     raw['score'],
    #     color='grey',
    #     alpha=0.5,
    #     s=5,
    #     label='seeds (latent_updates=True)'
    # )

    # plot mean ± SEM for both latent_updates settings
    for lu, col in zip([False, True], [cs.neuragem, 'grey']):
        grp = stats[stats['latent_updates'] == lu]
        ax.plot(
            grp['interleaved_phase_length'],
            grp['mean'],
            color=col,
            marker='o',
            markersize=cs.marker_size,
            label=f'{'NeuraGEM' if lu else 'Z lesioned'}'
        )
        ax.fill_between(
            grp['interleaved_phase_length'],
            grp['mean'] - grp['sem'],
            grp['mean'] + grp['sem'],
            color=col,
            alpha=0.3,
            rasterized=True,
        )

    # ax.set_title(f"interleaved_blocked (blocked={blocked_len})", fontsize=8)
    ax.set_xlabel('Interleaved phase length')
    ax.set_ylabel('Prediction accuracy')
    # ax.set_ylim(0.4, 1.05)
    ax.legend( frameon=True, fontsize=6, bbox_to_anchor=(.30, 0.8))
    #save fig
    folder_name = f'interleaved_blocked_blocked-{blocked_len}'
    export_path = os.path.join('./exports/csw/experiments', run_name, folder_name)
    os.makedirs(export_path, exist_ok=True)
    fig.savefig(os.path.join(export_path, f'interleaved_blocked_blocked-{blocked_len}.pdf'), dpi=300)
    print(f'file saved to {os.path.join(export_path, f"interleaved_blocked_blocked-{blocked_len}.pdf")}')

#%%
# === Blocked then interleaved analysis: Violin plots for blocked_interleaved curriculum ===
import matplotlib.pyplot as plt

# collect unique interleaved_phase_length values for blocked_interleaved
bi_lengths = sorted(
    results_df[results_df['curriculum'] == 'blocked_interleaved']['interleaved_phase_length'].dropna().unique()
)

if bi_lengths:
    n = len(bi_lengths)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(cs.panel_small_size[0], cs.panel_small_size[1] * n),
        sharex=True
    )
    if n == 1:
        axes = [axes]
    for ax, length in zip(axes, bi_lengths):
        plot_violin_for_param(
            results_df,
            curriculum='blocked_interleaved',
            param_name='interleaved_phase_length',
            param_value=length,
            ax=ax,
            title=f'blocked_interleaved (interleaved={length})'
        )
    plt.tight_layout()
    plt.show()
else:
    print("No data found for 'blocked_interleaved' violin plots.")

#%% This finds two examples of runs above and below average score for a given curriculum and parameters
# plots the runs to examine

# Specify parameters
selected_params = {
    'curriculum': 'interleaved_blocked',
    'blocked_phase_length': 800,
    'interleaved_phase_length': 1000,
    'latent_updates_during_shuffle': True
}
model_name = 'neuragem'

# Load and split loggers
comb_key, avg_score, loggers, above_seeds, below_seeds = load_and_split_loggers(
    run_name, model_name, selected_params
)
print(f"Average score for {comb_key}: {avg_score:.3f}")
print("Seeds above average:", above_seeds)
print("Seeds below average:", below_seeds)

# Plot first above- and below-average runs
if above_seeds:
    fig = plot_logger_panels(
        loggers[above_seeds[0]]['logger'],
        loggers[above_seeds[0]]['logger'].config,
        ['corrects','latent_2d','gradients'],
        subplot_height=1.5,
        annotate_phases='corrects'
    )
if below_seeds:
    fig = plot_logger_panels(
        loggers[below_seeds[0]]['logger'],
        loggers[below_seeds[0]]['logger'].config,
        ['corrects','latent_2d','gradients'],
        subplot_height=1.5,
        annotate_phases='corrects'
    )
    corrects_by_transitions = plot_corrects_by_transition(
        loggers[below_seeds[0]]['logger'], get_corrects_and_trial_starts
    )