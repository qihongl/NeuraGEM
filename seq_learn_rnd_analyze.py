"""
seq_learn_analyze_results.py

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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
# suppress all future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from functions_and_utils import plot_logger_panels, get_corrects_and_trial_starts, plot_switches_from_logger
from seq_learn_functions_and_utils import *
from functions_adaptation_dynamics_analysis import *

import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from collections import defaultdict
from itertools import product

from seq_learn_config import run_name, export_base_path, get_base_params

# transitions_to_use=['T1/2',]
transitions_to_use=['T5/6',]
TEST_PHASE_SAME = 'Testing\n(W frozen)'
TEST_PHASE_RANDOM = 'Testing\n(with random context)'
TEST_CONTEXT_TO_PHASE = {
        'same curr': TEST_PHASE_SAME,
        'rnd curr': TEST_PHASE_RANDOM,
}
# Default used in advanced neuragem analyses (can switch to 'rnd curr')
ANALYSIS_TEST_CONTEXT = 'same curr'


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


def get_phase_window(logger, phase_name):
    """
    Return [start, end) timestep window for the requested phase name.
    end=None means phase continues to end of logger timeline.
    """
    if not hasattr(logger, 'phases') or logger.phases is None:
        return None, None
    for idx, (name, start_t) in enumerate(logger.phases):
        if name == phase_name:
            end_t = logger.phases[idx + 1][1] if (idx + 1) < len(logger.phases) else None
            return start_t, end_t
    return None, None


def compute_testing_score_for_phase(logger, phase_name, alpha=0.5, transitions_to_use=('T1/2',)):
    """
    Compute testing score for a specific phase using the same logic as the old
    compute_testing_score(), but restricted to the requested phase window.
    """
    start_t, end_t = get_phase_window(logger, phase_name)
    if start_t is None:
        return np.nan

    # Keep old behavior: no transitions_to_use kwarg here
    corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)

    corrects = np.asarray(corrects)
    both_starts = np.asarray(both_starts)

    if corrects.size == 0:
        return np.nan

    # Old transition-focused scoring behavior
    focused = np.full_like(corrects, 0.5, dtype=float)
    for tr in transitions_to_use:
        if tr not in transitions:
            continue
        idx = both_starts + transitions[tr]
        idx = idx[(idx >= 0) & (idx < len(corrects))]
        focused[idx] = corrects[idx]

    # Exponential moving average (same as old helper)
    run_avg = []
    prev = 0.0
    for v in focused:
        if v == 0.5:
            run_avg.append(prev)
        else:
            prev = (1 - alpha) * prev + alpha * v
            run_avg.append(prev)
    run_avg = np.asarray(run_avg, dtype=float)

    # Restrict to requested phase
    if end_t is None:
        return np.nanmean(run_avg[start_t:])
    return np.nanmean(run_avg[start_t:end_t])

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

# ------------------------------------------------------------
# ONE FIGURE PER CURRICULUM:
#  - left: same-curriculum testing
#  - right: random-context testing
# ------------------------------------------------------------
for curriculum in curricula:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cs.panel_small_size[0] * 2.4, cs.panel_small_size[1] * 1.2),
        sharey=True,
    )

    models_to_plot = ['rnn', 'mrnn', 'neuragem']
    model_name_labels = ['RNN', 'MRNN', 'NG']

    rows = []
    for model in models_to_plot:
        for key, loggers in all_loggers[curriculum][model].items():
            for seed_idx, logger in enumerate(loggers):
                same_score = compute_testing_score_for_phase(
                    logger,
                    TEST_PHASE_SAME,
                    transitions_to_use=transitions_to_use,
                )
                rnd_score = compute_testing_score_for_phase(
                    logger,
                    TEST_PHASE_RANDOM,
                    transitions_to_use=transitions_to_use,
                )
                rows.extend([
                    {
                        'curriculum': curriculum,
                        'model': model,
                        'combination_key': key,
                        'seed': seed_idx,
                        'test_context': 'same curr',
                        'accuracy': same_score,
                    },
                    {
                        'curriculum': curriculum,
                        'model': model,
                        'combination_key': key,
                        'seed': seed_idx,
                        'test_context': 'rnd curr',
                        'accuracy': rnd_score,
                    },
                ])

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No rows to plot for curriculum={curriculum}")
        continue

    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    palette = [getattr(cs, m, 'gray') for m in models_to_plot]

    for ax, context_name in zip(axes, ['same curr', 'rnd curr']):
        dfi = df[df['test_context'] == context_name]
        sns.violinplot(
            x='model',
            y='accuracy',
            data=dfi,
            order=models_to_plot,
            ax=ax,
            palette=palette,
            # scale='width',
            width=0.5,
            **cs.old_violin_defaults,
            inner=None,
        )
        sns.stripplot(
            x='model',
            y='accuracy',
            data=dfi,
            order=models_to_plot,
            ax=ax,
            color='black',
            size=cs.marker_size,
            alpha=0.45,
            jitter=True,
        )
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_ylim(0.3, 1.2)
        ax.set_xlabel('')
        ax.set_xticklabels(model_name_labels)
        ax.set_title(context_name)

    axes[0].set_ylabel('Prediction accuracy')
    axes[1].set_ylabel('')
    fig.suptitle(f'{curriculum}', y=1.0)
    fig.tight_layout()

    save_folder = f'./exports/seq_learn/{run_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    fname = f"model_accuracy_by_test_context_{curriculum}.pdf"
    save_path = os.path.join(save_folder, fname)
    plt.savefig(save_path, transparent=True, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {save_path}")

############################################################################
############################################################################

#%% Neuragem latent + regression panels (phase-selectable score splitting)
for curriculum in curricula:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cs.panel_small_size[0] * 1.6, cs.panel_small_size[1]*0.8),
    )
    ax_latent, ax_reg = axes

    runs_true = [l for k, l in all_loggers[curriculum]['neuragem'].items()
                 if "latent_updates_during_shuffle-True" in k]
    runs_true = runs_true[0] if runs_true else []
    if not runs_true:
        plt.close(fig)
        continue

    ex = runs_true[0]
    li = np.concatenate(ex.latent_values, axis=0).reshape(-1, ex.latent_values[0].shape[-1])
    ax_latent.plot(li, '-', linewidth=0.5, alpha=0.8)
    plot_switches_from_logger(ax_latent, ex, ex.config, use_ll=False, alpha=0.2, alpha_interleaved=0.2)
    ax_latent.set_xlabel('Time steps')
    ax_latent.set_ylabel('Z')
    if curriculum == 'interleaved_blocked':
        t_block = next(t for name, t in ex.phases if name.startswith('Blocked'))
        ax_latent.axvline(t_block, color='red', linestyle='--', alpha=0.5)
        ax_latent.set_xlim(t_block - 100, t_block + 140)
    else:
        t_test = next(t for name, t in ex.phases if name.startswith('Testing'))
        ax_latent.set_xlim(t_test - 200, t_test)

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

    # regress betas for neuragem and save them for later plotting
    split_phase_name = TEST_CONTEXT_TO_PHASE[ANALYSIS_TEST_CONTEXT]
    if curriculum == 'interleaved_blocked':
        scores = [
            compute_testing_score_for_phase(
                l,
                split_phase_name,
                transitions_to_use=transitions_to_use,
            )
            for l in runs_true
        ]
        scores = np.asarray(scores, dtype=float)
        valid_mask = np.isfinite(scores)
        valid_runs = [r for r, keep in zip(runs_true, valid_mask) if keep]
        valid_scores = scores[valid_mask]

        if len(valid_runs) == 0:
            print(f"No valid neuragem scores for split in curriculum={curriculum}")
            continue

        # Separate runs_true into above-average and below-average groups
        avg_score = np.mean(valid_scores)
        runs_above_avg = [l for l, s in zip(valid_runs, valid_scores) if s > avg_score]
        runs_below_avg = [l for l, s in zip(valid_runs, valid_scores) if s <= avg_score]
        # Compute betas for both groups
        cid_above, rnd_above = compute_phase_betas(runs_above_avg, curriculum_phase_to_regress[curriculum])
        cid_below, rnd_below = compute_phase_betas(runs_below_avg, curriculum_phase_to_regress[curriculum])
    fig.tight_layout()

    save_folder = f'./exports/seq_learn/{run_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    fname = f"neuragem_latent_regression_{curriculum}_{ANALYSIS_TEST_CONTEXT.replace(' ', '_')}.pdf"
    save_path = os.path.join(save_folder, fname)
    plt.savefig(save_path, transparent=True, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {save_path}")


#%%
show_specific_plots = False
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
