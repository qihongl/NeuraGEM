"""
csw_analyze_results.py

1)  Load all saved `Logger` objects from 
    ./exports/contextual_switching_task/experiments/{run_name}/…
2)  Compute/testing peri‐switch errors (mean+SEM) per model & curriculum.
3)  Call plot_logger_panels to visualize the latent 2D space for neuragem.
"""
#%%
if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy import stats
# suppress all future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from functions_and_utils import explore_data_container, plot_logger_panels, get_corrects_and_trial_starts, plot_switches_from_logger
from functions_and_utils_2 import calculate_error
from seq_learn_functions_and_utils import *
from functions_adaptation_dynamics_analysis import *

import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from collections import defaultdict
from itertools import product
import matplotlib.gridspec as gridspec

from seq_learn_config import run_name, export_base_path, get_base_params

transitions_to_use = ['T5/6'] # To compute accuracy, it reflects learning the underlying computation.
# %% THIS CODE IS ADAPTED FROM THE ORIGINAL PAPER REPOSITORY. 
## Load human data
# From the Collab authors provided at: https://colab.research.google.com/github/PrincetonCompMemLab/csw_paper_final/blob/master/generate_paper_figures.ipynb#scrollTo=kmnNJROHRq7f&uniqifier=1

numeric_only = True

ALL_CONDITIONS = ['blocked',
 'interleaved',
 'blocked_rep',
 'interleaved_rep',
 'explicit_interleaved',
 'inserted_early',
 'inserted_middle',
 'inserted_late',
 'inserted_early_rep',
 'inserted_middle_rep',
 'inserted_late_rep']

## load and save
dfD = {}
for cond in ALL_CONDITIONS:
  for thresh in [0.9,0]:
    fname = f"{cond}_thresh{int(thresh*100)}.csv"
    df = pd.read_csv(f'https://raw.githubusercontent.com/PrincetonCompMemLab/blocked_training_facilitates_learning/master/data/human/{fname}')
    dfD[cond,thresh] = df
human_df = pd.concat(dfD,names=['condition','thresh'])

# additional columns
human_df.loc[:,'score'] = human_df.correct_response
human_df.loc[:,'response_node'] = human_df.apply(lambda r: [r.false_tonode,r.true_tonode][r.correct_response],axis=1)
node2stateD = {
  "BEGIN":0,
  "LOCNODEB":1,
  "LOCNODEC":2,
  "NODE11":3,
  "NODE12":4,
  "NODE21":5,
  "NODE22":6,
  "NODE31":7,
  "NODE32":8,
  "END":9
}


###########################
##########################
def load_results(filename, export_path):
    filepath = os.path.join(export_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            # print(f"Loading results from file: {filepath}")
            data = pickle.load(f)
        # print(f"Loaded results from file: {filename}")
        return data
    else:
        return None


def generate_param_combinations(param_grid):
    """
    Given a dictionary mapping parameter names to lists of values,
    generate a list of dictionaries representing every combination.
    """
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    return combinations

def summarize_linregress(x, y):
    """
    Run OLS linear regression via scipy.stats.linregress and compute t-stat for the slope.
    Returns a dict of stats for caption reporting.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    n = len(x)
    df = n - 2
    if np.isclose(abs(r_value), 1.0):
        t_stat = np.inf
    else:
        t_stat = r_value * np.sqrt(df / (1.0 - r_value**2))
    return {
        "n": n,
        "df": df,
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "t": t_stat,
        "p": p_value,
        "stderr": std_err,
    }

def format_p_value(p_value):
    if p_value < 1e-4:
        return r"$p < 10^{-4}$"
    if p_value < 1e-3:
        return r"$p < 10^{-3}$"
    return f"$p = {p_value:.3f}$"

def write_regression_stats(out_path, latex_path, stats_rows):
    """
    Write regression stats to a TSV and a LaTeX snippet for figure captions.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("label\tn\tdf\tslope\tintercept\tr\tt\tp\tstderr\n")
        for row in stats_rows:
            stats_dict = row["stats"]
            f.write(
                f"{row['label']}\t{stats_dict['n']}\t{stats_dict['df']}\t"
                f"{stats_dict['slope']:.6f}\t{stats_dict['intercept']:.6f}\t"
                f"{stats_dict['r']:.6f}\t{stats_dict['t']:.6f}\t"
                f"{stats_dict['p']:.6g}\t{stats_dict['stderr']:.6f}\n"
            )

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% OLS linear regression (scipy.stats.linregress), two-sided t-test on slope\n")
        for row in stats_rows:
            stats_dict = row["stats"]
            p_str = format_p_value(stats_dict["p"])
            t_str = (
                "inf"
                if np.isinf(stats_dict["t"])
                else f"{stats_dict['t']:.2f}"
            )
            f.write(
                f"{row['label']} (OLS linear regression, two-sided t-test on slope, "
                f"n = {stats_dict['n']}): "
                f"slope = {stats_dict['slope']:.3f}, "
                f"$r = {stats_dict['r']:.2f}$, "
                f"$t({stats_dict['df']}) = {t_str}$, "
                f"{p_str}.\n"
            )

# ------------------------------------------------------------
# Experiment identifiers and base params shared across scripts
base_params = get_base_params()

# ------------------------------------------------------------
# LOAD ALL CURRICULA FOR EACH MODEL & EXPERIMENT
# ------------------------------------------------------------
#%%

# base_models drives loading; models is used downstream for plotting
base_models   = ['rnn', 'mrnn', 'neuragem']
curricula     = ['blocked', 'interleaved', 'interleaved_blocked', ]#'blocked_interleaved']
models        = ['rnn', 'mrnn', 'neuragem', 'neuragem_z_lesioned']
# Now all_loggers [...] [...] [...] is a list of Logger objects per combination_key
all_loggers   = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
file_counts   = defaultdict(lambda: defaultdict(lambda: {'found': 0, 'missing': 0}))
missing_files = []

for base_model in base_models:
    for curriculum in curricula:
        # pull the right grid
        param_grid = dict(base_params[base_model][curriculum])
        # always sweep seeds 0–19
        param_grid['seed'] = list(range(20))

        for param_combination in generate_param_combinations(param_grid):
            param_combination['curriculum'] = curriculum
            seed = param_combination.pop('seed')

            # decide model label & drop latent flag from the key if neuragem
            if base_model == 'neuragem':
                latent_flag = param_combination.get('latent_updates_during_shuffle')
                model_label = 'neuragem' if latent_flag else 'neuragem_z_lesioned'
            else:
                model_label = base_model

            # build combination key (no seed)
            filtered = param_combination.copy()
            combination_key = "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))

            export_path = os.path.join('./exports/seq_learn/experiments', run_name, combination_key)
            filename    = f"results_{base_model}_{combination_key}_seed-{seed}.pkl"
            logger      = load_results(filename, export_path)

            if logger is not None:
                all_loggers[curriculum][model_label][combination_key].append(logger)
                file_counts[model_label][curriculum]['found'] += 1
            else:
                file_counts[model_label][curriculum]['missing'] += 1
                missing_files.append((model_label, curriculum, combination_key, seed))

# Print summary
print("\n=== Logger file summary ===")
for m in models:
    for curriculum in curricula:
        found   = file_counts[m][curriculum]['found']
        missing = file_counts[m][curriculum]['missing']
        print(f"{m} | {curriculum}: found {found}, missing {missing}")

if missing_files:
    print("\nMissing files (model, curriculum, combination_key, seed):")
    for entry in missing_files:
        print(entry)

#%%
# ------------------------------------------------------------
# Preprocess human data ONCE and cache for plotting
# ------------------------------------------------------------
# This block preprocesses and stores human data for each curriculum
# so plotting can be repeated without recomputing.

# Map curriculum to human data plotting parameters
curriculum_to_human = {
    'interleaved_blocked':  {'condL': ['inserted_late'],    'labels': ['Interleaved Blocked']},
    'blocked_interleaved':  {'condL': ['inserted_early'],   'labels': ['Blocked Interleaved']},
    'blocked':              {'condL': ['blocked'],          'labels': ['Blocked']},
    'interleaved':          {'condL': ['interleaved'],      'labels': ['Interleaved']},
}

# Precompute and cache processed human dataframes for each curriculum
human_plot_data = {}

for curriculum, params in curriculum_to_human.items():
    condL = params['condL']
    labels = params['labels']
    thresh = 0.9
    # Add columns if not already present
    if 'rfc_int' not in human_df.columns:
        human_df.loc[:, 'rfc_int'] = (human_df.true_rfc.str.split('-').str[-1].str.lower() == 'jungle').astype('int')
    if 'response_node_int' not in human_df.columns:
        human_df.loc[:, 'response_node_int'] = human_df.apply(lambda r: node2stateD[r.response_node], axis=1)
    df_plt = human_df.query(f"condition==@condL&thresh==@thresh").reset_index()
    dftest_plt = df_plt.query("block==4")
    grouped = dftest_plt.groupby(['subjnum', 'condition']).mean(numeric_only=numeric_only).reset_index()
    human_plot_data[curriculum] = {'grouped': grouped, 'labels': labels, 'condL': condL}

def plot_human_data_by_condition_cached(curriculum, ax=None, width = 0.5):
    data = human_plot_data[curriculum]
    grouped = data['grouped']
    labels = data['labels']
    condL = data['condL']
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=cs.panel_small_size)
    g = sns.violinplot(
        data=grouped,
        x='condition', y='score', hue_order='condition', ax=ax, order=condL,
        width=width,  # Slim the violins
        inner=None,
        linewidth=0.7,  # Remove inner markings
    )
    sns.stripplot(
        data=grouped,
        x='condition', y='score', ax=ax, order=condL,
        color='black', size=cs.marker_size, alpha=0.5, jitter=True
    )
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Prediction accuracy')
    ax.set_xlabel('')
    ax.set_xticklabels(['Humans'])
    ax.set_ylim(0.3, 1.3)


# ------------------------------------------------------------
# ONE FIGURE PER CURRICULUM, ALL THREE MODELS, neuragem tested with and without latent updates
# ------------------------------------------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


for curriculum in curricula:
# for curriculum in curricula:
    # Create a 1x4 grid, but use axes[0] for schematic (leave empty for now)
    # Stack latent trace and regression vertically in axes[2] and axes[3]
    fig, axes = plt.subplots(1, 5,
                       figsize=(cs.panel_small_size[0]*5,
                                cs.panel_small_size[1]))

    models_to_plot = ['rnn', 'mrnn', 'neuragem',]# 'neuragem_z_lesioned']

    all_scores = []
    all_models = []
    for model in models_to_plot:
        runs = []
        for key, loggers in all_loggers[curriculum][model].items():
            runs = loggers
            scs = [compute_testing_score(l, transitions_to_use=transitions_to_use)
                   for l in runs]
            all_scores.extend(scs)
            all_models.extend([model]*len(scs))

    # axes[0]: schematic (curriculum-specific)
    ax_schematic = axes[0]
    draw_curriculum_schematic(ax_schematic, curriculum)

    # axes[1]: human data
    ax = axes[1]
    plot_human_data_by_condition_cached(curriculum, ax=ax, width= 0.4)

    # axes[2]: model accuracy violin/strip
    ax = axes[2]
    df = pd.DataFrame({'model': all_models, 'accuracy': all_scores})
    df['accuracy'].replace([np.inf, -np.inf], np.nan, inplace=True)
    palette = [getattr(cs, m, 'gray') for m in models_to_plot]
    sns.violinplot(x='model', y='accuracy', data=df,
                   order=models_to_plot, ax=ax,
                   palette=palette, scale='width', width=0.4,
                   inner=None,
                   linewidth=0.7,)  # Remove inner markings
    sns.stripplot(x='model', y='accuracy', data=df,
                  order=models_to_plot, ax=ax,
                  color='black', size=cs.marker_size,
                  alpha=0.5, jitter=True)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(0.3, 1.3)
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_xticklabels([r"RNN$^{\mathrm{short}}$", r"RNN$^{\mathrm{long}}$", 'NG',])# 'NG\nZ-Lesioned'])

    # axes[3]: vertical stack of latent trace and regression boxplot
    # We'll use inset_axes to create two vertical panels inside axes[3]

    # ax_latent = inset_axes(axes[3], width="100%", height="40%", loc='upper center', borderpad=.12)
    # ax_reg = inset_axes(axes[3], width="100%", height="40%", loc='lower center', borderpad=.12)
    ax_latent = axes[3]
    ax_reg = axes[4]
    runs_true = [l for k, l in all_loggers[curriculum]['neuragem'].items()
                 if "latent_updates_during_shuffle-True" in k]
    runs_true = runs_true[0] if runs_true else []
    if runs_true:
        ex = runs_true[1] # just pick one run to visualize the latent space and switches; the regression will be computed across all runs
        li = np.concatenate(ex.latent_values, axis=0).reshape(-1, ex.latent_values[0].shape[-1])
        ax_latent.plot(li, '-', linewidth=0.5, alpha=0.8)
        plot_switches_from_logger(ax_latent, ex, ex.config, use_ll=False, alpha =0.2, alpha_interleaved=0.2)
        ax_latent.set_xlabel('Time steps')
        ax_latent.set_ylabel('Z')
        if curriculum == 'interleaved_blocked':
            t_block = next(t for name, t in ex.phases if name.startswith('Blocked'))
            ax_latent.axvline(t_block, color='red', linestyle='--', alpha=0.5)
            ax_latent.set_xlim(t_block-100, t_block+140)
        else:
            t_test = next(t for name, t in ex.phases if name.startswith('Testing'))
            ax_latent.set_xlim(1000, 1200)

    curriculum_phase_to_regress = {
        'blocked': 'blocked',
        'interleaved': 'interleaved',
        'interleaved_blocked': 'blocked',
        'blocked_interleaved': 'interleaved'
    }
    cid_b, rnd_b = compute_phase_betas(runs_true, curriculum_phase_to_regress[curriculum])
    plot_cid_rnd_single_phase(cid_b, rnd_b,
                              curriculum_phase_to_regress[curriculum],
                              ax=ax_reg, orient='v')
    ## regress betas for neuragem and save them for later plotting in cid_above and cid_below and rnd_above and rnd_below
    if curriculum == 'interleaved_blocked':
        scores = [compute_testing_score(l, transitions_to_use=transitions_to_use)
                for l in runs_true]
        # Separate runs_true into above-average and below-average groups
        avg_score = np.mean(scores)
        runs_above_avg = [l for l, s in zip(runs_true, scores) if s > avg_score]
        runs_below_avg = [l for l, s in zip(runs_true, scores) if s <= avg_score]
        # Compute betas for both groups
        cid_above, rnd_above = compute_phase_betas(runs_above_avg, curriculum_phase_to_regress[curriculum])
        cid_below, rnd_below = compute_phase_betas(runs_below_avg, curriculum_phase_to_regress[curriculum])
    
    squished = False
    ax = axes[1]
    pos = ax.get_position()
    if squished:
        ax.set_position((pos.x0+0.053, pos.y0, pos.width * 0.43, pos.height))
    else:
        ax.set_position((pos.x0+0.08, pos.y0, pos.width * 0.43, pos.height))

    ax = axes[2]
    pos = ax.get_position()
    if squished:
        ax.set_position((pos.x0-0.01, pos.y0, pos.width * 1.3, pos.height))
    else:
        ax.set_position((pos.x0-0.0, pos.y0, pos.width * 1.3, pos.height))

    ax = axes[3]
    pos = ax.get_position()
    if squished: 
        ax.set_position((pos.x0 + 0.045, pos.y0+0.07, pos.width+0.013 , pos.height* 0.5))
    else:
        ax.set_position((pos.x0 + 0.09, pos.y0+0.07, pos.width+0.013 , pos.height* 0.5))
    # ax.set_position((pos.x0 + 0.2, pos.y0, pos.width , pos.height))
    ax.yaxis.set_label_coords(-0.18, 0.5)

    ax = axes[4]
    pos = ax.get_position()
    if squished:
        ax.set_position((pos.x0 + 0.25, pos.y0, pos.width *0.5, pos.height))
    else:
        ax.set_position((pos.x0 + 0.18, pos.y0, pos.width *0.5, pos.height))

    save_folder = f'./exports/seq_learn/{run_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    fname = f"all_models_latent_{curriculum}.pdf"
    save_path = os.path.join(save_folder, fname)
    plt.savefig(save_path, transparent=True, bbox_inches='tight', facecolor='white')
    # plt.close(fig)
    print(f"Saved figure: {save_path}")

############################################################################
############################################################################


#%%
show_specific_plots = True
if show_specific_plots:
    seed_eg = 2
    _model = 'neuragem'
    # _model = 'rnn'
    _curriculum = 'interleaved_blocked'
    # _curriculum = 'interleaved'
    # _curriculum = 'blocked'
    # pick a key with latent_updates-True
    keys, loggers = next(iter(all_loggers[_curriculum][_model].items()))
    logger = loggers[seed_eg]
    print('seq_len:', logger.config.seq_len)
    panel_order = ['corrects', 'latent_2d', 'gradients']
    fig = plot_logger_panels(logger, logger.config, panel_order,
                                subplot_height=.8, annotate_phases='corrects')
