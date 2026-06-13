"""
csw_analyze_results.py

1)  Load all saved `Logger` objects from 
    ./exports/contextual_switching_task/experiments/{run_name}/…
2)  Compute/testing peri‐switch errors (mean+SEM) per model & curriculum.
3)  Call plot_logger_panels to visualize the latent 2D space for neuragem.
"""
#%%
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
import re
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

def parse_param_from_key(combination_key, param_name):
    token = f"{param_name}-"
    idx = combination_key.rfind(token)
    if idx == -1:
        return None
    start = idx + len(token)
    end = combination_key.find('_', start)
    if end == -1:
        end = len(combination_key)
    return combination_key[start:end]

def betas_scatter_local(t_rnd_betas, t_cid_betas, ax: mpl.axes._axes.Axes = None, title=None):
    """
    Scatter plot of T_rnd vs T_cid betas with regression line.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 2))
    reg_color = 'tab:red'
    ax.scatter(t_rnd_betas, t_cid_betas, s=20, c='grey', marker='x')
    if len(t_rnd_betas) >= 2:
        slope, intercept, _, _, _ = stats.linregress(t_rnd_betas, t_cid_betas)
        x_vals = np.asarray(t_rnd_betas)
        order = np.argsort(x_vals)
        ax.plot(x_vals[order], intercept + slope * x_vals[order], reg_color)
    ax.set_xlabel('T$_{rnd}$ Beta')
    ax.set_ylabel('T$_{cid}$ Beta')
    if title:
        ax.set_title(title, fontsize=8)

# ------------------------------------------------------------
# Experiment identifiers
# run_name = 'initial_runs'
# run_name = 'controlling_phase_lengths_runs'
# run_name = 'interleaved_vs_interleaved_blocked'
run_name = 'random_ctx'
export_base_path = f'./exports/csw/experiments/{run_name}/'

# base params by model and curriculum
base_params = {
        'neuragem': {
            'interleaved': {
                'blocked_phase_length': [1200],
                'interleaved_phase_length': [7000],
                'latent_updates_during_shuffle': [True],
                'shuffle_or_interleave': ['interleave', 'shuffle'],
                'random_transition_shuffle_or_interleave': ['interleave', 'shuffle'],
            },
            'interleaved_blocked': {
                'blocked_phase_length': [2500],
                'interleaved_phase_length': [700],
                'latent_updates_during_shuffle': [True, False]            },
        },
    }

# ------------------------------------------------------------
# LOAD ALL CURRICULA FOR EACH MODEL & EXPERIMENT
# ------------------------------------------------------------
#%%

# base_models drives loading; models is used downstream for plotting
base_models   = ['neuragem']
curricula     = ['interleaved', 'interleaved_blocked']
models        = ['neuragem', 'neuragem_z_lesioned']
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
                if latent_flag is None:
                    model_label = 'neuragem'
                else:
                    model_label = 'neuragem' if latent_flag else 'neuragem_z_lesioned'
            else:
                model_label = base_model

            # build combination key (no seed)
            filtered = param_combination.copy()
            combination_key = "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))

            export_path = os.path.join('./exports/csw/experiments', run_name, combination_key)
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



def get_interleaved_condition_runs(all_loggers):
    curriculum = 'interleaved'
    model = 'neuragem'
    condition_runs = []
    for combination_key, loggers in all_loggers[curriculum][model].items():
        if 'latent_updates_during_shuffle-False' in combination_key:
            continue
        shuffle_mode = parse_param_from_key(combination_key, 'shuffle_or_interleave')
        transition_mode = parse_param_from_key(combination_key, 'random_transition_shuffle_or_interleave')
        if shuffle_mode is None or transition_mode is None:
            continue
        if shuffle_mode != transition_mode:
            continue
        label = f"shuffle={shuffle_mode}, transition={transition_mode}"
        condition_runs.append((combination_key, label, loggers))
    return sorted(condition_runs, key=lambda entry: entry[1])

def write_interleaved_stats_report(interleaved_conditions, run_name):
    stats_rows = []
    for _, label, runs in interleaved_conditions:
        cid_b, rnd_b = compute_phase_betas(runs, 'interleaved')
        stats_rows.append({
            "label": f"{label} (T_rnd vs T_cid)",
            "stats": summarize_linregress(rnd_b, cid_b),
        })

    if not stats_rows:
        return

    save_folder = f'./exports/csw/{run_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    stats_out = os.path.join(save_folder, "interleaved_regression_stats.tsv")
    latex_out = os.path.join(save_folder, "interleaved_regression_stats_caption.txt")
    write_regression_stats(stats_out, latex_out, stats_rows)
    print(f"Wrote interleaved regression stats: {stats_out}")
    print(f"Wrote interleaved caption snippet: {latex_out}")


# ------------------------------------------------------------
# ONE FIGURE PER CURRICULUM, 
# ------------------------------------------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


interleaved_conditions = get_interleaved_condition_runs(all_loggers)
write_interleaved_stats_report(interleaved_conditions, run_name)

plot_configs = []
for curriculum in curricula:
    if curriculum == 'interleaved':
        for combination_key, label, runs in interleaved_conditions:
            plot_configs.append((curriculum, runs, label, combination_key))
    else:
        runs_true = [l for k, l in all_loggers[curriculum]['neuragem'].items()
                     if "latent_updates_during_shuffle-True" in k]
        runs_true = runs_true[0] if runs_true else []
        plot_configs.append((curriculum, runs_true, None, None))

for curriculum, runs_true, condition_label, condition_key in plot_configs:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(cs.panel_small_size[0]*3.5, cs.panel_small_size[1]),
        gridspec_kw={'width_ratios': [1.0, 0.5, .4, .7]},
    )

    models_to_plot = [ 'neuragem',]# 'neuragem_z_lesioned']

    all_scores = []
    all_models = []
    for model in models_to_plot:
        if condition_key:
            runs = runs_true
            scs = [compute_testing_score(l, transitions_to_use=['T5/6'])
                   for l in runs]
            all_scores.extend(scs)
            all_models.extend([model]*len(scs))
        else:
            for key, loggers in all_loggers[curriculum][model].items():
                runs = loggers
                scs = [compute_testing_score(l, transitions_to_use=['T5/6'])
                       for l in runs]
                all_scores.extend(scs)
                all_models.extend([model]*len(scs))

    # axes[0]: schematic (curriculum-specific)
    ax_schematic = axes[0]
    shuffle_mode = None
    if condition_key:
        shuffle_mode = parse_param_from_key(condition_key, 'shuffle_or_interleave')
        transition_mode = parse_param_from_key(condition_key, 'random_transition_shuffle_or_interleave')
    draw_curriculum_schematic(ax_schematic, curriculum, Tcid_mode=shuffle_mode, Trnd_mode=transition_mode)

    # axes[1]: model accuracy violin/strip
    ax = axes[1]
    df = pd.DataFrame({'model': all_models, 'accuracy': all_scores})
    df['accuracy'].replace([np.inf, -np.inf], np.nan, inplace=True)
    palette = [getattr(cs, m, 'gray') for m in models_to_plot]
    sns.violinplot(x='model', y='accuracy', data=df,
                   order=models_to_plot, ax=ax,
                   palette=palette, scale='width', width=0.3, inner=None, linewidth=0.7),
    sns.stripplot(x='model', y='accuracy', data=df,
                  order=models_to_plot, ax=ax,
                  color='black', size=cs.marker_size,
                  alpha=0.5, jitter=True)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(0.3, 1.3)
    ax.set_ylabel("Prediction accuracy")
    # ax.set_yticklabels([])
    ax.set_xlabel("")
    label_map = {'rnn': 'RNN', 'mrnn': 'MRNN', 'neuragem': 'NG', 'neuragem_z_lesioned': 'NG\nZ-Lesioned'}
    xticklabels = [label_map.get(m, m) for m in models_to_plot]
    ax.set_xticklabels(xticklabels)

    ax_reg = axes[2]

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
    ax_scatter = axes[3]
    scatter_title = condition_label if condition_label else None
    betas_scatter_local(rnd_b, cid_b, ax=ax_scatter, title=None)
    

    ## regress betas for neuragem and save them for later plotting in cid_above and cid_below and rnd_above and rnd_below
    if curriculum == 'interleaved_blocked':
        scores = [compute_testing_score(l, transitions_to_use=['T5/6',])
                for l in runs_true]
        # Separate runs_true into above-average and below-average groups
        avg_score = np.mean(scores)
        runs_above_avg = [l for l, s in zip(runs_true, scores) if s > avg_score]
        runs_below_avg = [l for l, s in zip(runs_true, scores) if s <= avg_score]
        # Compute betas for both groups
        cid_above, rnd_above = compute_phase_betas(runs_above_avg, curriculum_phase_to_regress[curriculum])
        cid_below, rnd_below = compute_phase_betas(runs_below_avg, curriculum_phase_to_regress[curriculum])
    
    # fig.tight_layout()
    save_folder = f'./exports/csw/{run_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    if condition_key:
        fname = f"all_models_latent_{curriculum}_{condition_key}.pdf"
    else:
        fname = f"all_models_latent_{curriculum}.pdf"
    save_path = os.path.join(save_folder, fname)
    # if save_path longer than 255 characters, truncate the middle of the filename
    if len(save_path) > 255:
        base, ext = os.path.splitext(save_path)
        save_path = base[:100] + '...' + base[-100:] + ext
    plt.savefig(save_path, transparent=True, bbox_inches='tight', facecolor='white')
    # plt.close(fig)
    print(f"Saved figure: {save_path}")


############################################################################
############################################################################
#%% ## 'neuragem_z_lesioned' plotting using inset_axes for vertical stacking
############################################################################
############################################################################
############################################################################
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
curriculum: str = 'interleaved_blocked'

# Create figure with 4 subplots in a row
panel_width: float = cs.panel_small_size[0] * 3.0
panel_height: float = cs.panel_small_size[1] * 1

fig, axes = plt.subplots(1, 4, figsize=(panel_width, panel_height), gridspec_kw={'width_ratios': [1, 0.7, 1, 1]})

# axes[0]: Above/below average regression plots side by side
ax_reg_above = axes[0]
plot_cid_rnd_single_phase(cid_above, rnd_above,
                          curriculum_phase_to_regress['interleaved_blocked'],
                          ax=ax_reg_above, orient='v')
ax_reg_above.set_title("> avg", fontsize=8, fontweight='bold')

# axes[1]: Below avg regression plot (shares y-axis with axes[0])
ax_reg_below = axes[1]
plot_cid_rnd_single_phase(cid_below, rnd_below,
                          curriculum_phase_to_regress['interleaved_blocked'],
                          ax=ax_reg_below, orient='v')
ax_reg_below.set_title("< avg", fontsize=8, fontweight='bold')

# Share y-axis between regression plots and remove y-axis from right plot
ax_reg_below.sharey(ax_reg_above)
ax_reg_below.set_ylabel("")
ax_reg_below.tick_params(left=False, labelleft=False)
ax_reg_below.spines['left'].set_visible(False)
ax_reg_below.set_xlabel('')
ax_reg_below.text(-0, 1.3, 'Accuracy', fontsize=7, ha='center', va='center', transform=ax_reg_below.transAxes)


# axes[1]: violinplot of neuragem_z_lesioned performance
ax_acc = axes[2]
model = 'neuragem_z_lesioned'
runs = []
for key, loggers in all_loggers[curriculum][model].items():
    runs.extend(loggers)
scs = [compute_testing_score(l, transitions_to_use=['T5/6',]) for l in runs]
df = pd.DataFrame({'accuracy': scs})
sns.violinplot(y='accuracy', data=df, ax=ax_acc, color=getattr(cs, model, 'gray'), width=0.3, inner=None, linewidth=0.7)
sns.stripplot(y='accuracy', data=df, ax=ax_acc, color='black', size=cs.marker_size, alpha=0.5, jitter=True)
ax_acc.axhline(0.5, color='black', linestyle='--', alpha=0.5)
ax_acc.set_ylim(0.3, 1.3)
ax_acc.set_ylabel("Prediction accuracy")
ax_acc.set_xlabel("")
ax_acc.set_xticklabels(['NG Z inactive\nduring interleaved'])



# axes[2]: latent trace for neuragem_z_lesioned (half height, centered)
ax_latent = axes[3]
runs_false = [l for k, l in all_loggers[curriculum][model].items()
                if "latent_updates_during_shuffle-False" in k]
if runs_false:
    runs_false = runs_false[0]  # extra list dim
    ex = runs_false[0]
    li = np.concatenate(ex.latent_values, axis=0).reshape(-1, ex.latent_values[0].shape[-1])
    ax_latent.plot(li, '-', linewidth=0.5, alpha=0.8)
    plot_switches_from_logger(ax_latent, ex, ex.config, use_ll=False, alpha=0.2, alpha_interleaved=0.2)
    ax_latent.set_xlabel('Time steps')
    ax_latent.set_ylabel('Z')
    # mark transition
    if curriculum == 'interleaved_blocked':
        t_block = next(t for name, t in ex.phases if name.startswith('Blocked'))
        ax_latent.axvline(t_block, color='red', linestyle='--', alpha=0.5)
        ax_latent.set_xlim([t_block-70, t_block+150])
    else:
        t_test = next(t for name, t in ex.phases if name.startswith('Testing'))
        ax_latent.set_xlim([t_test-200, t_test])
# Adjust width of regression plots to 0.4 of original
pos_above = ax_reg_above.get_position()

ax_reg_above.set_position([pos_above.x0, pos_above.y0, pos_above.width * 0.6, pos_above.height])
# move the xlabel to the right by 0.1

ax_reg_above.set_xlabel('Predictor', fontsize=7)
ax_reg_above.xaxis.label.set_position((1.1, -0.22))


pos_below = ax_reg_below.get_position()
ax_reg_below.set_position([pos_below.x0-0.091, pos_below.y0, pos_below.width , pos_below.height]) 
# Adjust subplot positions to keep proportions
# axes[1] (violin)
ax = axes[2] # accuracy
pos = ax.get_position()
ax.set_position([pos.x0 + 0.06, pos.y0, pos.width * 0.7, pos.height])

ax = axes[3]  # Latent
pos = ax.get_position()
ax.set_position([pos.x0 + 0.1, pos.y0+0.2,pos.width*1.4, pos.height*0.5])

# save figure as pdf
save_folder = f'./exports/csw/{run_name}/analysis/'
os.makedirs(save_folder, exist_ok=True)
fname = f"neuragem_z_lesioned_latent_{curriculum}.pdf"
save_path = os.path.join(save_folder, fname)
plt.savefig(save_path, transparent=True, bbox_inches='tight', facecolor='white')
print(f"Saved figure: {save_path}")

#%%    
# Detailed regression analysis
#%Regression analysis for neuragem z-lesioned comparison
_selected_model = 'neuragem'
_curriculum = 'interleaved_blocked'
# all runs with latent_updates=True
key, runs_true = next(iter(all_loggers[_curriculum][_selected_model].items()))
end_accuracies = [compute_testing_score(l) for l in runs_true]
# compute betas
inter_cid, inter_rnd = compute_phase_betas(runs_true, 'interleaved')
block_cid, block_rnd = compute_phase_betas(runs_true, 'blocked')
# plot
# fig = plot_cid_rnd_single_phase(inter_cid, inter_rnd, 'interleaved')
# fig = plot_cid_rnd_single_phase(block_cid, block_rnd, 'blocked')

from configs import SeqLearnConfig
config = SeqLearnConfig(experiment_to_run='few_long_blocks')
fig2 = plot_statistical_relationships(inter_rnd, block_rnd, block_cid, end_accuracies)
pdf2 = f'{config.export_path}{config.dataset_name}_scatter.pdf'
plt.savefig(pdf2, format='pdf', bbox_inches='tight', transparent= True)
print(pdf2)

save_folder = f'./exports/csw/{run_name}/analysis/'
os.makedirs(save_folder, exist_ok=True)
stats_rows = [
    {
        "label": "Blocked T_{ran} beta vs end accuracy",
        "stats": summarize_linregress(block_rnd, end_accuracies),
    },
    {
        "label": "Blocked T_{TID} beta vs end accuracy",
        "stats": summarize_linregress(block_cid, end_accuracies),
    },
    {
        "label": "Interleaved T_{ran} vs blocked T_{ran}",
        "stats": summarize_linregress(inter_rnd, block_rnd),
    },
    {
        "label": "Interleaved T_{ran} vs blocked T_{TID}",
        "stats": summarize_linregress(inter_rnd, block_cid),
    },
]
stats_out = os.path.join(save_folder, "regression_stats.tsv")
latex_out = os.path.join(save_folder, "regression_stats_caption.txt")
write_regression_stats(stats_out, latex_out, stats_rows)
print(f"Wrote regression stats: {stats_out}")
print(f"Wrote caption snippet: {latex_out}")


show_specific_plots = False
if show_specific_plots:
    seed_eg = 2
    _model = 'neuragem'
    _curriculum = 'interleaved_blocked'
    # pick a key with latent_updates-True
    keys, loggers = next(iter(all_loggers[_curriculum][_model].items()))
    logger = loggers[seed_eg]
    panel_order = ['corrects', 'latent_2d', 'gradients']
    fig = plot_logger_panels(logger, logger.config, panel_order,
                                subplot_height=1.5, annotate_phases='corrects')
