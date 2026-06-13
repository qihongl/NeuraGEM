'''
Produces learning rate summary plot. Reads experiments run by cst_run_generalization_experiments.py
Need to change the ood_test_type and run_name at the top to point to the right experiments. 
Run with ood_test_type 'ood_means' for novel means plot, 'ood_stds' for novel stds plot, and 'block_size' for block size plot.
 
'''
if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
#%%
import os
import glob
import pickle
from collections import defaultdict

import numpy as np

from Bayesian_obs_generalization import run_bayesian_generalization
from configs import ContextualSwitchingTaskConfig
from functions_and_utils import *
from functions_and_utils_2 import *
from functions_adaptation_dynamics_analysis import *
import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

if __name__ == "__main__":

    # run_name, ood_test_type = 'new_runs_stds', 'ood_stds'
    # run_name, ood_test_type = 'new_runs_means', 'ood_means' # this is for trying diff std values. But remember to turn off scaling later for diff stds
    run_name, ood_test_type = 'new_runs_50_means', 'ood_means' # this is for trying diff std values. But remember to turn off scaling later for diff stds
    # run_name, ood_test_type = 'new_runs_block_size', 'block_size'
    no_of_seeds = 50
    start_seed = 0
    add_stats_test = False
    selected_models = ["rnn_seq_len_5", "rnn_seq_len_50", "neuragem", ]
    # selected_models = ["rnn_seq_len_5", "rnn_seq_len_50", "bayesian" ]
    # selected_models = ["rnn_seq_len_5", "rnn_seq_len_50",  ]
    if ood_test_type in ['block_size', 'ood_stds']:
        ng_l2_loss = 0.0001 
    else:  # this is to explore new_runs_means which has diff l2 and stds (turn off std scaling)
        # ng_l2_loss = 0.002 #[0.0001,0.0002,0.0004,0.0006,0.0008,],
        ng_l2_loss = 0.0008 #[0.0001,0.0002,0.0004,0.0006,0.0008,],
    # Define the base export folder where experiments were saved.
    export_base_path = f'./exports/contextual_switching_task/experiments/{run_name}/'

    # Define a list of comparison experiments.
    # For each experiment you specify:
    #   - a human-readable name,
    #   - parameters (as a dict) for RNN with seq_len 5,
    #   - parameters (as a dict) for RNN with seq_len 50,
    #   - parameters (as a dict) for NeuraGEM (e.g. LU_lr and l2_loss).
    # You can add additional parameters as needed.  
    comparison_experiments = [
    {
        'name': 'exp_std0.3',
        'rnn_params_5': {'seq_len': 5, 'WU_lr': 1e-3, 'default_std': 0.3},
        'rnn_params_50': {'seq_len': 50, 'WU_lr': 1e-3, 'default_std': 0.3},
        'neuragem_params': {'l2_loss': ng_l2_loss, 'WU_lr': 1e-4, 'default_std': 0.3},
        'bayesian_params': {'default_std': 0.3}
        # "bayesian_params": {"default_std": 0.3, "observation_noise_mode": {"default": "fixed", "ood_stds": "ground_truth"}},
    },
    {
        'name': 'exp_std0.4',
        'rnn_params_5': {'seq_len': 5, 'WU_lr': 1e-3, 'default_std': 0.4},
        'rnn_params_50': {'seq_len': 50, 'WU_lr': 1e-3, 'default_std': 0.4},
        'neuragem_params': { 'l2_loss': ng_l2_loss, 'WU_lr': 1e-4, 'default_std': 0.4},
        'bayesian_params': {'default_std': 0.4}
    },
    ]

    # This dictionary will store the loaded loggers for each experiment and model.
    # The keys for each experiment are 'rnn_seq_len_5', 'rnn_seq_len_50', and 'neuragem'.
    # Each entry is now a list over all seeds.
    comparison_loggers = defaultdict(dict)

    for exp in comparison_experiments:
        exp_name = exp['name']
        # For RNN experiments, build a base key string using all parameter combinations.
        # Sort the parameters alphabetically before building the key.
        rnn_key_5 = "_".join([f"{key}-{value}" for key, value in sorted(exp['rnn_params_5'].items())])
        rnn_key_50 = "_".join([f"{key}-{value}" for key, value in sorted(exp['rnn_params_50'].items())])
        folder_rnn_5 = os.path.join(export_base_path, rnn_key_5)
        folder_rnn_50 = os.path.join(export_base_path, rnn_key_50)

        # For NeuraGEM experiments, build the key using all parameter combinations.
        # Sort the parameters alphabetically before building the key.
        neuragem_key = "_".join([f"{key}-{value}" for key, value in sorted(exp['neuragem_params'].items())])
        folder_neuragem = os.path.join(export_base_path, neuragem_key)

        # Use glob to load all seed files matching the new naming convention.
        filename_pattern_rnn_5 = os.path.join(folder_rnn_5, f"results_rnn_frozen_True_{rnn_key_5}_seed-*.pkl")
        filename_pattern_rnn_50 = os.path.join(folder_rnn_50, f"results_rnn_frozen_True_{rnn_key_50}_seed-*.pkl")
        filename_pattern_neuragem = os.path.join(folder_neuragem, f"results_neuragem_frozen_True_{neuragem_key}_seed-*.pkl")

        # Load RNN (seq_len 5) loggers
        files_rnn_5 = glob.glob(filename_pattern_rnn_5)[start_seed:no_of_seeds]
        if not files_rnn_5:
            print(f"Failed to load RNN (seq_len 5) loggers for experiment {exp_name} from {folder_rnn_5}")
        else:
            data_list = []
            for file in files_rnn_5:
                try:
                    with open(file, 'rb') as f:
                        data_list.append(pickle.load(f))
                except EOFError:
                    print(f"EOFError encountered while loading file {file}, skipping.")
                except Exception as e:
                    print(f"Error encountered while loading file {file}: {e}, skipping.")
            comparison_loggers[exp_name]['rnn_seq_len_5'] = data_list
            print(f"Loaded {len(data_list)} RNN (seq_len 5) logger files for experiment {exp_name}: {rnn_key_5}")

        # Load RNN (seq_len 50) loggers
        files_rnn_50 = glob.glob(filename_pattern_rnn_50)[start_seed:no_of_seeds]
        if not files_rnn_50:
            print(f"Failed to load RNN (seq_len 50) loggers for experiment {exp_name} from {folder_rnn_50}")
        else:
            data_list = []
            for file in files_rnn_50:
                with open(file, 'rb') as f:
                    data_list.append(pickle.load(f))
            comparison_loggers[exp_name]['rnn_seq_len_50'] = data_list
            print(f"Loaded {len(data_list)} RNN (seq_len 50) logger files for experiment {exp_name}: {rnn_key_50}")

        # Load NeuraGEM loggers
        files_neuragem = glob.glob(filename_pattern_neuragem)[start_seed:no_of_seeds]
        if not files_neuragem:
            print(f"Failed to load NeuraGEM loggers for experiment {exp_name} from {folder_neuragem}")
        else:
            data_list = []
            for file in files_neuragem:
                with open(file, 'rb') as f:
                    data_list.append(pickle.load(f))
            comparison_loggers[exp_name]['neuragem'] = data_list
            print(f"Loaded {len(data_list)} NeuraGEM logger files for experiment {exp_name}: {neuragem_key}")

        bayesian_params = exp.get('bayesian_params')
        if bayesian_params:
            bayesian_config = ContextualSwitchingTaskConfig("figure")
            for key, value in bayesian_params.items():
                if hasattr(bayesian_config, key):
                    setattr(bayesian_config, key, value)
            bayesian_obs_mode = "ground_truth" if ood_test_type == 'ood_stds' else "fixed"
            bayesian_payloads = []
            for seed_idx in range(no_of_seeds):
                try:
                    bayesian_artifacts = run_bayesian_generalization(
                        base_config=bayesian_config,
                        seed=seed_idx,
                        hazard_rate=None,
                        test_types=(ood_test_type,),
                        observation_noise_mode=bayesian_obs_mode,
                        save_artifacts=False,
                    )
                except Exception as e:
                    print(f"Failed to run Bayesian observer for experiment {exp_name}, seed {seed_idx}: {e}")
                    continue
                test_loggers = bayesian_artifacts.generalization_loggers.get(ood_test_type, {})
                if not test_loggers:
                    continue
                bayesian_payloads.append({'test_logger': test_loggers})
            if bayesian_payloads:
                comparison_loggers[exp_name]['bayesian'] = bayesian_payloads
                print(f"Generated {len(bayesian_payloads)} Bayesian observer runs for experiment {exp_name}")
            else:
                print(f"Failed to generate Bayesian observer loggers for experiment {exp_name}")

    print("Comparison loggers loaded for experiments:")
    for exp_name, loggers in comparison_loggers.items():
        print(f"Experiment {exp_name}: models loaded: {list(loggers.keys())}")
    

#%% 

error_type = 'abs_from_mean'
pre_window = 3
post_window = 30

#############################################
#### Plot adaptation curves and training loss.
#############################################
# selected_exp = "exp1"
exp_dict = {'ood_means': 'exp_std0.3', 'ood_stds': 'exp_std0.3', 'block_size': 'exp_std0.4'}
selected_exp = exp_dict[ood_test_type]


# ----- Testing Loggers Plot and Summary Collection -----
summary_test_mse = {}      # Store (time_avg_error, time_sem_error) for each model and OOD
summary_learning_rates = {} 
summary_slopes = {}        # Store full arrays of avg_slopes for each model & OOD

for model_name in selected_models:
    seed_loggers = comparison_loggers[selected_exp].get(model_name)
    if seed_loggers is None:
        print(f"Loggers for {model_name} in experiment {selected_exp} not found.")
        continue

    # group test loggers by OOD value
    ood_logger_dict = {}
    for seed in seed_loggers:
        test_loggers = seed.get('test_logger', {})
        if not test_loggers:
            print(f"Test logger not found in one of the seeds for {model_name}.")
            continue
        for ood_value, logger_obj in test_loggers.items():
            ood_logger_dict.setdefault(ood_value, []).append(logger_obj)


    summary_test_mse[model_name] = {}
    summary_learning_rates[model_name] = {}
####
    
    for ood_value, logger_list in ood_logger_dict.items():
        ####
        debug_latents = False
        if debug_latents:
            fig, axes = plt.subplots(len(logger_list), figsize = [2, len(logger_list)])
            for (i, log) in enumerate(logger_list):
                axes[i].plot(np.array(log.llcids).squeeze())
        ####        
        segments_list = []
        for logger in logger_list:

            segs = extract_switch_centered_segments(logger, pre_window=pre_window, 

                post_window=post_window, phases_to_include=None)
            segments_list.append(segs)
        # Flatten segments from all loggers.
        if segments_list:
            segments = [seg for sublist in segments_list for seg in sublist]
        else:
            segments = []
        
        # Compute rolling estimates.
        unique_times, avg_slopes, std_slopes = compute_rolling_lr_estimates(segments
                    , filter_outside_values= False, filter_outliers=True if ood_test_type != 'ood_means' else False, clip_outside_values=True if ood_test_type != 'ood_means' else False)
        # mean_pre_switch_lr, std_pre_switch_lr = extract_pre_switch_lr(avg_slopes, 15, k=5)
        avg_slopes = avg_slopes.squeeze()
            

        color = (
            cs.short_horizon_rnn if model_name == "rnn_seq_len_5"
            else cs.long_horizon_rnn if model_name == "rnn_seq_len_50"
            else cs.neuragem if model_name == "neuragem"
            else cs.bayesian
        )
        label = (
            "RNN (h=5)" if model_name == "rnn_seq_len_5"
            else "RNN (h=50)" if model_name == "rnn_seq_len_50"
            else "NeuraGEM" if model_name == "neuragem"
            else "Bayesian"
        )
        plot_individual_runs = False
        if plot_individual_runs:
            fig, ax = plt.subplots(figsize=cs.panel_small_size)
            
            plot_rolling_learning_rate(unique_times, avg_slopes, std_slopes, ax=ax, color=color, window_size = 1)
            ax.set_title(f'{label} OOD: {ood_value}')
            ax.axvline(0, linestyle="--", color="k", alpha=0.5)

        # save summary lr values
        if ood_test_type == 'ood_means':
            exclude_first_n_times = 10
            time_steps_span_to_include = 5
            times_to_include = np.where((unique_times > exclude_first_n_times) & (unique_times > 0))[0]
        elif ood_test_type == 'ood_stds':
            # skip pre_window and first 10 time points and take only 5 points after that
            exclude_first_n_times = 10
            time_steps_span_to_include = 5
            times_to_include = np.where((unique_times > exclude_first_n_times) & (unique_times > 0))[0]
        elif ood_test_type == 'block_size':
            exclude_first_n_times = 5 # be more conservative because some blocks are really short.
            time_steps_span_to_include = 5
            times_to_include = np.where((unique_times > exclude_first_n_times) & (unique_times > 0))[0]

    
        unique_times = unique_times[times_to_include][:time_steps_span_to_include]
        avg_slopes = avg_slopes[times_to_include][:time_steps_span_to_include]
        std_slopes = std_slopes[times_to_include][:time_steps_span_to_include]
        use_sem_per_data_point = False
        time_avg_slopes = np.mean(avg_slopes)
        if use_sem_per_data_point:
            time_avg_sem = np.mean(std_slopes)
        else:
            time_avg_sem = np.std(avg_slopes) / np.sqrt(len(avg_slopes))                
        summary_learning_rates[model_name][ood_value] = (time_avg_slopes, time_avg_sem)

# ------------ plot learning rates ------------
fig, ax = plt.subplots(figsize=cs.panel_small_size)
if ood_test_type == 'ood_stds': # novel stds plot need not be this wide. 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

colors = {
    "rnn_seq_len_5": cs.short_horizon_rnn,
    "rnn_seq_len_50": cs.long_horizon_rnn,
    "neuragem": cs.neuragem,
    "bayesian": cs.bayesian,
}
model_labels = {
    "rnn_seq_len_5": "RNN",
    "rnn_seq_len_50": "MetaRNN",
    "neuragem": "NeuraGEM",
    "bayesian": "Bayesian",
}
for model in selected_models:
    model_summary = summary_learning_rates.get(model)
    if not model_summary:
        print(f"No learning rate summary for {model}, skipping plot.")
        continue
    ood_vals = sorted(model_summary.keys())
    if not ood_vals:
        continue
    lr_vals, sem_vals = [], []
    for ood in ood_vals:
        time_avg_lr, time_sem_lr = model_summary[ood]
        lr_vals.append(time_avg_lr)
        sem_vals.append(time_sem_lr)
    lr_vals_arr = np.array(lr_vals)
    sem_vals_arr = np.array(sem_vals)

    is_bayesian = model == "bayesian"
    ax.plot(
        ood_vals,
        lr_vals_arr,
        linestyle="--" if is_bayesian else "-",
        marker="" if is_bayesian else "o",
        label=model_labels[model],
        color=colors[model],
        linewidth=0.8,
        markersize=2,
    )
    ax.fill_between(ood_vals, lr_vals_arr - sem_vals_arr, lr_vals_arr + sem_vals_arr,
                    color=colors[model], alpha=0.2)

# --- Statistical test across OOD = 0.5 between RNN h=5 and RNN h=50 ---
import scipy.stats as stats


if add_stats_test:
    # OOD points to test (keep as a list for possible future expansion)
    test_oods = [0.5]
    # choose models to compare
    m1, m2 = "rnn_seq_len_5", "rnn_seq_len_50"
    if 'neuragem' in selected_models:
        m1, m2 = "neuragem", "rnn_seq_len_50"

    # Effective sample size per curve equals number of time points used to compute the SEM band
    # In current code, time_steps_span_to_include = 5 for all ood_test_type branches
    n_eff_map = {'ood_means': 5, 'ood_stds': 5, 'block_size': 5}
    n1 = n2 = n_eff_map.get(ood_test_type, 5)

    def welch_from_means_sem(m1_mean, m1_sem, n1, m2_mean, m2_sem, n2):
        # Convert SEMs back to variances
        s1_sq = (m1_sem ** 2) * n1
        s2_sq = (m2_sem ** 2) * n2
        denom = np.sqrt(s1_sq / n1 + s2_sq / n2)
        if denom == 0:
            return np.nan, np.nan, np.nan
        tstat = (m1_mean - m2_mean) / denom
        v1 = s1_sq / n1
        v2 = s2_sq / n2
        num = (v1 + v2) ** 2
        den = (v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)
        df = num / den if den > 0 else np.nan
        pval = 2 * stats.t.sf(np.abs(tstat), df) if np.isfinite(df) else np.nan
        return tstat, df, pval

    last_pval = None
    for o in test_oods:
        if (m1 in summary_learning_rates and o in summary_learning_rates[m1]
                and m2 in summary_learning_rates and o in summary_learning_rates[m2]):
            mean1, sem1 = summary_learning_rates[m1][o]
            mean2, sem2 = summary_learning_rates[m2][o]
            tstat, df, pval = welch_from_means_sem(mean1, sem1, n1, mean2, sem2, n2)
            last_pval = pval
            print(f"Welch test {m1} vs {m2} @ OOD {o}: t={tstat:.3f}, df={df:.1f}, p={pval:.3g}, n_eff=({n1},{n2})")
        else:
            print(f"Missing data for stats at OOD {o}")

    # annotate significance on the plot (single position if one OOD)
    if last_pval is not None and np.isfinite(last_pval):
        x0 = float(np.mean(test_oods))
        y1 = summary_learning_rates[m1][test_oods[0]][0]
        y2 = summary_learning_rates[m2][test_oods[0]][0]
        ax.plot([x0, x0], [y1, y2], color='k', lw=0.75)
        ax.plot([min(test_oods), max(test_oods)], [y1, y1], color='k', lw=.5)
        ax.plot([min(test_oods), max(test_oods)], [y2, y2], color='k', lw=.5)
        stars = '***' if last_pval < 1e-3 else '**' if last_pval < 1e-2 else '*' if last_pval < 5e-2 else 'n/s'
        ax.text(x0, max(y1, y2) + 0.02, stars, ha='center', va='bottom', color='k', fontsize=10)


if ood_test_type == 'ood_means':
    iid_vals=(0.2, 0.8)
elif ood_test_type == 'ood_stds':
    iid_vals=[0.3]
elif ood_test_type == 'block_size':
    iid_vals=[30]
if ood_test_type == 'ood_means':
    ax.set_xlabel("Observations mean")
elif ood_test_type == 'ood_stds':
    ax.set_xlabel("Observations std")
    ax.set_xlim(left=0.25, right=0.61)
elif ood_test_type == 'block_size':
    ax.set_xlabel("Block size")
else:
    ax.set_xlabel('OOD value')
for iid_val in iid_vals:
    ax.axvline(iid_val, linestyle="--", color=cs.iid_data, alpha=0.5, linewidth=2)
ax.set_ylabel("Time-Avg Learning Rate")
ax.set_title(f"{selected_exp}", fontsize=6)
# ax.legend(fontsize=6)
plot_path = os.path.join(export_base_path, f"test_summary_{ood_test_type}_peri_switch_learning_rates_{selected_exp}.pdf")
fig.savefig(plot_path)
print(f"Saved  to: {plot_path}")

#%%
