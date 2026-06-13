'''
THis loads specific experiments and first looks at the activity patterns during training
Produces heatmaps of activity variance, means, or norms across training blocks.
It sorts the heatmaps by the last or penultimate block and compares the two.

Then it quantifies the angular similarity between the last two blocks of activity patterns
across multiple seeds for statstical analysis. Produces a final plot with the cosine similarity
THen finally tries to quantify the weights from the input layer to the hidden layer
and compares the weights for the two latent units (z1 and z2) across the last two blocks.

'''
try:  # safer IPython auto-reload setup
    from IPython import get_ipython  # type: ignore
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic('load_ext', 'autoreload')
        _ip.run_line_magic('autoreload', '2')
except Exception:
    pass
#%%
import os, glob, pickle
from functions_and_utils import *
from functions_and_utils_2 import *
from functions_adaptation_dynamics_analysis import *
from collections import defaultdict
import numpy as np
from scipy.stats import ttest_ind, sem
from scipy.stats import ttest_rel, ttest_1samp  # added for paired and one-sample tests
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

import seaborn as sns
import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

import matplotlib as mpl
import copy

mpl.rcParams['figure.dpi'] = 150

#######################
####### RUns available 
#######################
# run_name, ood_test_type = 'input_z_sweeps', 'ood_means'
# run_name, ood_test_type = 'input_z_sweeps_train1000', 'ood_means'
# run_name, ood_test_type = 'input_z_sweeps_additive', 'ood_means'
# I ran z_sweeps again but passed additive vs. multiplicative as a parameter just like in the lr_sweep. Goal is to pot them side by side
# run_name= 'z_sweep_train_4000' 

''' there is a question of whether this was run with training len 4000 or not 
next step.. run post gating multiplicative form and compare the results to the pre gating that was run here by default.
steps: change run name
    then add use post_gating true
    use pre gating faluse
config.pre_gating = False
config.post_gating = True
'''

''' To run this use adapt_run_array_input_z_sweep.py which can be run on slurm using ./submit_adapt_job.sh 359 input_z_sweeps
RUn it with these param grids and config_overrides:
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
        'blocked_phase_length': 4000,
        'start_always_on_the_same_block': False,
        # 'WU_lr': 1e-3,  # note: neuragem-specific update is done in run_single_experiment
        # 'l2_loss': 0.0008,
        'run_input_sweep': True,
        'run_lr_sweep': False,
        'log_hidden_states': True, # Log hidden states for analysis
        'log_end_weights': True, # Log end weights for analysis

    }

'''

## FOR Fig in paper, run this twice. Once with compute_CCGP = True for the CCGP plot and once with it False for the proper population analysis using less noisy inputs.

## New exp simply running OOD_means using this code base, but the diff is I log the hidden states for PCA analysis
# run_name= 'pca_ood_means' 
run_name= 'pca_ood_means_final_check'

compute_CCGP = True  # whether to compute cross-condition generalization performance (CCGP) for context
# std = 0.1  # PCA is more human interpretable at this level but 
std = 0.3 if compute_CCGP else 0.1  # ccgp will not work with std .1 because no data points from one mean would every be closer to the other mean (this is used for the cross condition.)

# Define the base export folder where experiments were saved.
export_base_path = f'./exports/contextual_switching_task/experiments/{run_name}/'

# Define a list of comparison experiments.
# For each experiment you specify:
#   - a human-readable name,
#   - parameters (as a dict) for RNN with seq_len 5,
#   - parameters (as a dict) for RNN with seq_len 50,
#   - parameters (as a dict) for NeuraGEM (e.g. LU_lr and l2_loss).
# You can add additional parameters as needed.  
ng_l2_loss = 0.0008 #[0.0001,0.0002,0.0004,0.0006,0.0008, 0.001,0.0015,0.002,0.004,]
comparison_experiments = [
    {
        'name': 'mul',
        'rnn_params_5': {'seq_len': 5, 'WU_lr': 1e-3, 'default_std': std},
        'rnn_params_50': {'seq_len': 50, 'WU_lr': 1e-3, 'default_std': std},
        'neuragem_params': {'l2_loss': ng_l2_loss, 'WU_lr': 1e-4, 'default_std': std,
                                'use_add_gating': False }  # True for additive gating, False for multiplicative gating
    },
    {
        'name': 'add',
        'rnn_params_5': {'seq_len': 5, 'WU_lr': 1e-3, 'default_std': std},
        'rnn_params_50': {'seq_len': 50, 'WU_lr': 1e-3, 'default_std': std},
        'neuragem_params': {'l2_loss': ng_l2_loss, 'WU_lr': 1e-4, 'default_std': std,
                                'use_add_gating': True}  # True for additive gating, False for multiplicative gating
    }
]

# This dictionary will store the loaded loggers for each experiment and model.
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
    files_rnn_5 = glob.glob(filename_pattern_rnn_5)
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
    files_rnn_50 = glob.glob(filename_pattern_rnn_50)
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
    files_neuragem = glob.glob(filename_pattern_neuragem)
    if not files_neuragem:
        print(f"Failed to load NeuraGEM loggers for experiment {exp_name} from {folder_neuragem}")
    else:
        data_list = []
        for file in files_neuragem:
            with open(file, 'rb') as f:
                try:
                    data_list.append(pickle.load(f))
                except EOFError:
                    print(f"EOFError encountered while loading file {file}, skipping.")
        comparison_loggers[exp_name]['neuragem'] = data_list
        print(f"Loaded {len(data_list)} NeuraGEM logger files for experiment {exp_name}: {neuragem_key}")

print("Comparison loggers loaded for experiments:")
for exp_name, loggers in comparison_loggers.items():
    print(f"Experiment {exp_name}: models loaded: {list(loggers.keys())}")
    
#%% 
#############################################
# choose which metric to display ('var' or 'norm')
seed_idx = 1  # pick one of the seeds to analyze. There is 20
# metric = 'norm'
# metric = 'var'
metric = 'mean'
segment_min_size = 0  # minimum segment size to compute metrics
ignore_first_timesteps = 4 # ignore first 10 timesteps in each segment, because RNN takes time to adapt

# Number of context blocks to include in analysis (always the first k blocks)
num_blocks_to_include = 15

include_OOD_mean_in_PCA = False  # whether to include OOD mean in PCA analysis
include_OOD_mean_in_computing_PCA_components = True  # whether to include OOD mean in computing PCA components

# define block_step: stride over starting blocks when forming adjacent block pairs
# e.g. block_step=1 -> (0,1),(1,2),(2,3)... ; block_step=2 -> (0,1),(2,3),(4,5)...
block_step = 1

# Indices of two blocks to compare for angular similarity (negative allowed)
block_idx_a, block_idx_b = -2, -1

selected_models = ["neuragem", "rnn_seq_len_5", "rnn_seq_len_50"]

results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # results[exp][model][metric]
# Loop through each experiment and model and compute metrics
fig_sim, axes_sims = plt.subplots(2,2, sharey=True, figsize = [4,4])

for selected_exp in [ce['name'] for ce in comparison_experiments]:
    for model_name in selected_models:
        seed_loggers = comparison_loggers[selected_exp].get(model_name, [])
        train_loggers = [s.get('train_logger') for s in seed_loggers if s.get('train_logger')]
        test_loggers = [s.get('test_logger') for s in seed_loggers if s.get('test_logger')]

        # Restrict to the first k context blocks of training data
        train_loggers = [copy.deepcopy(l) for l in train_loggers]
        for logger in train_loggers:
            if logger.config.log_initial_burn_in_timesteps and len(logger.inputs) > len(logger.llcids):
              logger.inputs = logger.inputs[logger.config.seq_len-1:]  # remove initial burn-in

            llcids = np.array(logger.llcids).squeeze()
            if llcids.size == 0:
                continue

            boundaries = np.where(np.diff(llcids) != 0)[0] + 1
            starts = np.r_[0, boundaries]
            ends = np.r_[boundaries, len(llcids)]
            n_blocks_available = len(starts)
            n_blocks_keep = min(num_blocks_to_include, n_blocks_available)
            end_idx = ends[n_blocks_keep - 1]

            logger.inputs = logger.inputs[:end_idx]
            logger.predicted_outputs = logger.predicted_outputs[:end_idx]
            logger.llcids = logger.llcids[:end_idx]
            logger.hlcids = logger.hlcids[:end_idx]
            if hasattr(logger, 'hidden_states'):
                logger.hidden_states = logger.hidden_states[:end_idx]

        for seed_idx, logger in enumerate(train_loggers):
            hs = np.array(logger.hidden_states).squeeze()    # (T, units)
            context_ids = np.array(logger.llcids).squeeze()       # (T,)

            boundaries = np.where(np.diff(context_ids) != 0)[0] + 1
            starts = np.r_[0, boundaries]
            ends   = np.r_[boundaries, len(context_ids)]
            num_blocks = len(starts)

            sel = list(zip(starts, ends))

            # compute per‐block metrics
            block_metrics = []

            for s, e in sel:
                seg = hs[s:e]
                if ignore_first_timesteps > 0:
                    if (e - s) <= ignore_first_timesteps or (e - s - ignore_first_timesteps) < segment_min_size:
                        continue  # skip segment if not enough timesteps after ignoring
                    seg = seg[ignore_first_timesteps:]
                
                if len(seg) < segment_min_size:
                    continue  # skip segment if not enough timesteps
                block_metrics.append({
                    'var':  np.var(seg, axis=0),
                    'norm': np.linalg.norm(seg, axis=0) / np.sqrt(len(seg)),  # scale by sqrt of segment length
                    'mean': np.mean(seg, axis=0),
                })
            n_blocks = len(block_metrics)

            # --- Compute similarity/angle across adjacent block pairs, sampling starts every block_step ---
            similarity_vec = []
            angle_vec      = []
            for i in range(0, n_blocks - 1, block_step):
                va = block_metrics[i][metric]
                vb = block_metrics[i + 1][metric]
                if np.linalg.norm(va) < 1e-8 or np.linalg.norm(vb) < 1e-8 or len(va) < 2:
                    cos_sim = np.nan; angle = np.nan
                else:
                    cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    angle   = np.degrees(np.arccos(cos_sim))
                similarity_vec.append(cos_sim)
                angle_vec.append(angle)

            # store per‐seed vectors and summary statistics
            results[selected_exp][model_name]['cos_sim_vec'].append(similarity_vec)
            results[selected_exp][model_name]['angle_vec'].append(angle_vec)
            results[selected_exp][model_name]['cos_sim_mean'].append(
                np.nanmean(similarity_vec) if similarity_vec else np.nan
            )
            results[selected_exp][model_name]['angle_mean'].append(
                np.nanmean(angle_vec) if angle_vec else np.nan
            )

            # --- Weight analysis: only for NeuraGEM, always use all units ---
            if 'input_layer_weights' in logger.others.keys():
                w = logger.others['input_layer_weights'] # these are the input layer weights for the NG add model. The first two dims are Z to hidden, ignore the last dim which is input to hidden.

                if 'P' in logger.others.keys(): # those are only logged for mul NG. So their presence indicates this is a mul gating run
                    p = logger.others['P']  # P is the multiplicative gates layer, shape (z dim, hiddens)

                if w.shape[1] < 2: # for the mul NG case, replace w with P. 
                    w = p.T  # transpose to match the shape (hiddens, z dim)

                # compute mean weights for module z1 and z2 (these are z -> hidden weights, not input->latent)
                wa_mean = np.mean(w[:, 0])
                wb_mean = np.mean(w[:, 1])
            else:
                wa_mean = np.nan
                wb_mean = np.nan
                print(f"{model_name}: No input layer weights found for seed {seed_idx}.")
            
            # Store results for this seed
            results[selected_exp][model_name]['wa_mean'].append(wa_mean)
            results[selected_exp][model_name]['wb_mean'].append(wb_mean)
            results[selected_exp][model_name]['wh_mean'].append(max([wa_mean, wb_mean]))
            results[selected_exp][model_name]['wl_mean'].append(min([wa_mean, wb_mean]))
            corr_val = np.corrcoef(va, vb)[0, 1] if len(va) > 1 else np.nan
            results[selected_exp][model_name]['corr'].append(corr_val)

            if compute_CCGP:
                logger_ccgp = train_loggers[seed_idx-1]
                if logger_ccgp.config.log_initial_burn_in_timesteps and len(logger_ccgp.inputs) > len(logger_ccgp.llcids):
                    logger_ccgp.inputs = logger_ccgp.inputs[logger_ccgp.config.seq_len-1:]
                inputs = np.array(logger_ccgp.inputs).squeeze()
                context_labels = np.array(logger_ccgp.llcids).squeeze()
                hidden_states = np.array(logger_ccgp.hidden_states).squeeze()
                hidden_states = hidden_states[-len(inputs):]

                condition_labels = (inputs >= 0.5).astype(int)
                train_mask = (condition_labels == 0)
                test_mask  = (condition_labels == 1)

                X_train, y_train = hidden_states[train_mask], context_labels[train_mask]
                X_test,  y_test  = hidden_states[test_mask],  context_labels[test_mask]

                y_train = np.isclose(y_train, 0.8).astype(int)
                y_test  = np.isclose(y_test, 0.8).astype(int)

                clf = LinearSVC(max_iter=10000)  # removed dual='auto' for compatibility
                clf.fit(X_train, y_train)
                acc1 = clf.score(X_test, y_test)

                clf2 = LinearSVC(max_iter=10000)
                clf2.fit(X_test, y_test)
                acc2 = clf2.score(X_train, y_train)

                ccgp = 0.5 * (acc1 + acc2)
                results[selected_exp][model_name]['ccgp'].append(ccgp)

        # Report mean ± SEM across seeds
        for key in ('angle_mean', 'wa_mean', 'wb_mean', 'cos_sim_mean'):
            arr = np.array(results[selected_exp][model_name][key])
            if arr.size == 0:
                print(f"{model_name}: no data for {key}")
                continue
            m = arr.mean()
            s = sem(arr)
            unit = '°' if key == 'angle_mean' else ''
            label = {
                'angle_mean': 'angle',
                'wa_mean': 'mean w(z1)',
                'wb_mean': 'mean w(z2)',
                'cos_sim_mean': 'cosine similarity'
            }[key]
            print(f"{model_name}: {label} across seeds = {m:.2f}{unit} ± {s:.2f}{unit}")

        #####################################################################################################
        # PCA only on the last seed
        #####################################################################################################
        # Inputs: from a single logger
        hidden_states = np.array(logger.hidden_states).squeeze()  # (T, H)
        context_ids = np.array(logger.llcids).squeeze()           # (T,)
        T = hidden_states.shape[0]

        if include_OOD_mean_in_PCA:
            selected_mean = 0.5
            testing_logger = test_loggers[seed_idx][selected_mean]
            
            if logger.config.log_initial_burn_in_timesteps and len(testing_logger.inputs) > len(testing_logger.llcids):
                testing_logger.inputs = testing_logger.inputs[testing_logger.config.seq_len-1:]
            hidden_states_testing = np.array(testing_logger.hidden_states).squeeze()  # (T, H)
            context_ids_testing = np.array(testing_logger.llcids).squeeze()           # (T,)
            

        # Convert context to 0 (low) and 1 (high) for coloring
        context_bin = np.isclose(context_ids, 0.8).astype(int)

        # Run PCA
        pca = PCA(n_components=2)
        hidden_pca = pca.fit_transform(hidden_states)  # (T, 2)
        if include_OOD_mean_in_PCA:
            hidden_states = np.concatenate((hidden_states, hidden_states_testing), axis=0)  # (T + T_test, H)
            context_ids = np.concatenate((context_ids, context_ids_testing), axis=0)
        if include_OOD_mean_in_computing_PCA_components:
            hidden_pca = pca.fit_transform(hidden_states)
        else:
            hidden_pca = pca.transform(hidden_states)  # (T, 2)
            
        # store PCA outputs (append to lists to satisfy list default)
        results[selected_exp][model_name]['hidden_pca'].append(hidden_pca)
        results[selected_exp][model_name]['context_ids'].append(context_ids)

    
        # Plot the temporal evolution of similarity over training
        ax = axes_sims.flatten()[selected_models.index(model_name)]
        # Pad or truncate all vectors to the same length for stacking
        cos_sim_vecs = results[selected_exp][model_name]['cos_sim_vec']
        min_len = min(len(v) for v in cos_sim_vecs)
        sims = np.stack([v[:min_len] for v in cos_sim_vecs])
        color_dict = {'rnn_seq_len_5': cs.short_horizon_rnn, 
                      'rnn_seq_len_50': cs.long_horizon_rnn, 
                      'neuragem': cs.neuragem}
        if model_name == 'neuragem' and selected_exp == 'add':
            color_dict[model_name] = cs.neuragem_additive
        elif model_name == 'neuragem' and selected_exp == 'mul':
            color_dict[model_name] = cs.neuragem
        ax.plot(sims.mean(0), label=f'{model_name} {selected_exp}', color=color_dict[model_name], linewidth=0.5)
        ax.legend()
        #####################################################################################################
        ############## # Heat maps of activity for one example logger #######################################
        #####################################################################################################
        # --- Active‐neuron heatmaps side by side ---
        # pick seed_idx seed for hidden states
        logger_ref = train_loggers[seed_idx]
        if logger_ref.config.log_hidden_states is False:
            print(f"{model_name}: No hidden states logged for seed {seed_idx}. Turn on config.log_hidden_states maybe?")
            raise ValueError("No hidden states logged for this seed.")
            
        hs = np.array(logger_ref.hidden_states).squeeze()    # (T, units)
        context_ids = np.array(logger_ref.llcids).squeeze()       # (T,)
        
        boundaries = np.where(np.diff(context_ids) != 0)[0] + 1
        starts = np.r_[0, boundaries]
        ends   = np.r_[boundaries, len(context_ids)]
        num_blocks = len(starts)
        # select all blocks (data already trimmed above)
        sel = list(zip(starts, ends))
        # compute per‐block metrics
        block_metrics = []

        for s, e in sel:
            seg = hs[s:e]
            if ignore_first_timesteps > 0:
                if (e - s) <= ignore_first_timesteps or (e - s - ignore_first_timesteps) < segment_min_size:
                    continue  # skip segment if not enough timesteps after ignoring
                seg = seg[ignore_first_timesteps:]
            
            if len(seg) < segment_min_size:
                continue  # skip segment if not enough timesteps
            block_metrics.append({
                'var':  np.var(seg, axis=0),
                'norm': np.linalg.norm(seg, axis=0) / np.sqrt(len(seg)),  # scale by sqrt of segment length
                'mean': np.mean(seg, axis=0),
            })
        n_blocks = len(block_metrics)

        # helper to build data matrix for a given ref
        def build_data_mat(ref_key):
            if ref_key == 'last':
                idx = n_blocks - 1
            elif ref_key == 'penultimate':
                idx = n_blocks - 2
            else:
                raise ValueError
            order = np.argsort(block_metrics[idx][metric])[::-1]
            mat = np.vstack([block_metrics[i][metric][order] for i in range(n_blocks)])
            return mat, idx

        mat_last, idx_last = build_data_mat('last')
        mat_pen, idx_pen = build_data_mat('penultimate')

        if False: # deactivate heatmap plots
            fig, axes = plt.subplots(1, 2, figsize=(3 * cs.panel_small_size[0], cs.panel_small_size[1]))
            sns.heatmap(mat_last.T, cmap='viridis', ax=axes[0], cbar_kws={'label': metric})
            axes[0].set_title(f"{model_name}", fontsize=6)
            axes[0].set_xlabel('Training block')
            axes[0].set_ylabel('Units (sorted by last)')

            sns.heatmap(mat_pen.T, cmap='viridis', ax=axes[1], cbar_kws={'label': metric})
            axes[1].set_xlabel('Training block')
            axes[1].set_ylabel('Units (sorted by penultimate)')

        # heatmap_path = os.path.join(
        #     export_base_path,
        #     f'unit_{metric}_heatmaps_{selected_exp}_{model_name}.pdf'
        # )
        # fig.savefig(heatmap_path)
        # print(f"Saved side‐by‐side heatmaps ({metric}) for {model_name} to: {heatmap_path}")




################################################################################
######################## PLOTS########################################
###############################################################################

# ─── Revised combined summary figure ───
# gather metrics
cos_means = [
    np.mean(results['mul']['rnn_seq_len_5']['cos_sim_mean']),
    np.mean(results['mul']['rnn_seq_len_50']['cos_sim_mean']),
    np.mean(results['mul']['neuragem']['cos_sim_mean']),
    np.mean(results['add']['neuragem']['cos_sim_mean']),
]
cos_sems = [
    sem(results['mul']['rnn_seq_len_5']['cos_sim_mean']),
    sem(results['mul']['rnn_seq_len_50']['cos_sim_mean']),
    sem(results['mul']['neuragem']['cos_sim_mean']),
    sem(results['add']['neuragem']['cos_sim_mean']),
]
wa_means = [
    np.mean(results['mul']['neuragem']['wh_mean']),
    np.mean(results['add']['neuragem']['wh_mean']),
]
wb_means = [
    np.mean(results['mul']['neuragem']['wl_mean']),
    np.mean(results['add']['neuragem']['wl_mean']),
]
wa_sems = [sem(results['mul']['neuragem']['wh_mean']), sem(results['add']['neuragem']['wh_mean'])]
wb_sems = [sem(results['mul']['neuragem']['wl_mean']), sem(results['add']['neuragem']['wb_mean'])]
ccgp_means = [
    np.mean(results['mul']['rnn_seq_len_5']['ccgp']),
    np.mean(results['mul']['rnn_seq_len_50']['ccgp']),
    np.mean(results['mul']['neuragem']['ccgp']),
    np.mean(results['add']['neuragem']['ccgp']),
]
ccgp_sems = [
    sem(results['mul']['rnn_seq_len_5']['ccgp']),
    sem(results['mul']['rnn_seq_len_50']['ccgp']),
    sem(results['mul']['neuragem']['ccgp']),
    sem(results['add']['neuragem']['ccgp']),
]

# build 1×4 panels
fig, axes = plt.subplots(1, 4,
    figsize=(4 * cs.panel_small_size[0], cs.panel_small_size[1]),
    sharey=False
)

colors=[cs.short_horizon_rnn, cs.long_horizon_rnn, cs.neuragem, cs.neuragem_additive, ]
# 1) Cosine similarity
labels = ["RNN", "MRNN", "NG (mul)", "NG (add)", ]
axes[0].bar(
    labels, cos_means, yerr=cos_sems,
    capsize=4, width=0.4, alpha=0.8,
    color=colors,
)
# horizontal line at RNN h=5 mean
axes[0].axhline(cos_means[0],
    color=cs.short_horizon_rnn, linestyle='--', linewidth=1
)
axes[0].set_ylim(0.4, 1.0)
axes[0].tick_params(axis='x', rotation=30)

# 2) NG mul weights
x = np.arange(2)
w = 0.35
axes[1].bar(
    x, [wa_means[0], wb_means[0]], w,
    yerr=[wa_sems[0], wb_sems[0]],
    capsize=4, color=['#555555', '#aaaaaa']
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(["w(z1)","w(z2)"])
axes[1].set_ylabel("mean weight")
axes[2].set_xlabel("NG Mul")
# 3) NG add weights
axes[2].bar(
    x, [wa_means[1], wb_means[1]], w,
    yerr=[wa_sems[1], wb_sems[1]],
    capsize=4, color=['#555555', '#aaaaaa']
)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["w(z1)","w(z2)"])
axes[2].set_xlabel("NG Add")

# 4) CCGP
axes[3].bar(
    labels, ccgp_means, yerr=ccgp_sems,
    capsize=4, width=0.4, alpha=0.8,
    color=colors,
)
axes[3].axhline(0.5,
    color='grey', linestyle='--', linewidth=1
)
# Label the chance line
xmax = axes[3].get_xlim()[1]
axes[3].text(
    xmax + 0.02, 0.51, 'chance',
    color='grey', ha='right', va='bottom',
    fontsize=6, clip_on=False
)
axes[3].set_ylabel("CCGP")
axes[3].set_ylim(0.4, .80)
axes[3].tick_params(axis='x', rotation=30)

# ===================== Statistical tests & annotations ===================== #

def p_to_star(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def annotate_line(ax, x1, x2, y, h, text, fontsize=6):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k', linewidth=0.6)
    ax.text((x1+x2)/2, y+h+0.005, text, ha='center', va='bottom', fontsize=fontsize)

def annotate_bar_star(ax, x, y, text, fontsize=6):
    ax.text(x, y, text, ha='center', va='bottom', fontsize=fontsize)

# Collect raw per-seed arrays
cos_rnn      = np.array(results['mul']['rnn_seq_len_5']['cos_sim_mean'], dtype=float)
cos_mrnn     = np.array(results['mul']['rnn_seq_len_50']['cos_sim_mean'], dtype=float)
cos_ng_mul   = np.array(results['mul']['neuragem']['cos_sim_mean'], dtype=float)
cos_ng_add   = np.array(results['add']['neuragem']['cos_sim_mean'], dtype=float)

# Remove NaNs
cos_rnn    = cos_rnn[~np.isnan(cos_rnn)]
cos_mrnn   = cos_mrnn[~np.isnan(cos_mrnn)]
cos_ng_mul = cos_ng_mul[~np.isnan(cos_ng_mul)]
cos_ng_add = cos_ng_add[~np.isnan(cos_ng_add)]

# Cosine similarity comparisons vs baseline RNN (Welch t-tests)
comparisons_cos = {
    'RNN vs MRNN': (cos_rnn, cos_mrnn),
    'RNN vs NG (mul)': (cos_rnn, cos_ng_mul),
    'RNN vs NG (add)': (cos_rnn, cos_ng_add),
}
cos_stats_strings = []
ax_cos = axes[0]
max_cos_height = max(cos_means[i] + cos_sems[i] for i in range(len(cos_means)))
line_y_start = max_cos_height + 0.015
line_step = 0.05
for idx, (label_cmp, (a, b)) in enumerate(comparisons_cos.items(), start=1):
    if len(a) > 1 and len(b) > 1:
        t_stat, p_val = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        stars = p_to_star(p_val)
        cos_stats_strings.append(f"{label_cmp}: t={t_stat:.2f}, p={p_val:.2e} ({stars})")
        x_target = idx  # bars: 0,1,2,3
        annotate_line(ax_cos, 0, x_target, line_y_start + (idx-1)*line_step, 0.012, stars)
    else:
        cos_stats_strings.append(f"{label_cmp}: insufficient data")

# Weight differences (paired) within each gating type
wz1_mul = np.array(results['mul']['neuragem']['wh_mean'], dtype=float)
wz2_mul = np.array(results['mul']['neuragem']['wl_mean'], dtype=float)
wz1_add = np.array(results['add']['neuragem']['wh_mean'], dtype=float)
wz2_add = np.array(results['add']['neuragem']['wl_mean'], dtype=float)

mask_mul = ~np.isnan(wz1_mul) & ~np.isnan(wz2_mul)
mask_add = ~np.isnan(wz1_add) & ~np.isnan(wz2_add)
wz1_mul_c, wz2_mul_c = wz1_mul[mask_mul], wz2_mul[mask_mul]
wz1_add_c, wz2_add_c = wz1_add[mask_add], wz2_add[mask_add]

ax_w_mul = axes[1]
ax_w_add = axes[2]
weight_stats_strings = []
if len(wz1_mul_c) > 1:
    t_stat_mul, p_val_mul = ttest_rel(wz1_mul_c, wz2_mul_c, nan_policy='omit')
    stars_mul = p_to_star(p_val_mul)
    annotate_line(ax_w_mul, 0, 1, max(wa_means[0]+wa_sems[0], wb_means[0]+wb_sems[0]) + 0.01, 0.01, stars_mul)
    weight_stats_strings.append(f"NG (mul) w(z1) vs w(z2): t={t_stat_mul:.2f}, p={p_val_mul:.2e} ({stars_mul})")
else:
    weight_stats_strings.append("NG (mul) w(z1) vs w(z2): insufficient data")
if len(wz1_add_c) > 1:
    t_stat_add, p_val_add = ttest_rel(wz1_add_c, wz2_add_c, nan_policy='omit')
    stars_add = p_to_star(p_val_add)
    annotate_line(ax_w_add, 0, 1, max(wa_means[1]+wa_sems[1], wb_means[1]+wb_sems[1]) + 0.01, 0.01, stars_add)
    weight_stats_strings.append(f"NG (add) w(z1) vs w(z2): t={t_stat_add:.2f}, p={p_val_add:.2e} ({stars_add})")
else:
    weight_stats_strings.append("NG (add) w(z1) vs w(z2): insufficient data")

# Between-gating comparisons for each module (independent tests)
if len(wz1_mul_c) > 1 and len(wz1_add_c) > 1:
    t_stat_wz1, p_val_wz1 = ttest_ind(wz1_mul_c, wz1_add_c, equal_var=False, nan_policy='omit')
    t_stat_wz2, p_val_wz2 = ttest_ind(wz2_mul_c, wz2_add_c, equal_var=False, nan_policy='omit')
    weight_stats_strings.append(f"w(z1) mul vs add: t={t_stat_wz1:.2f}, p={p_val_wz1:.2e} ({p_to_star(p_val_wz1)})")
    weight_stats_strings.append(f"w(z2) mul vs add: t={t_stat_wz2:.2f}, p={p_val_wz2:.2e} ({p_to_star(p_val_wz2)})")

# CCGP tests vs chance and vs baseline RNN
ccgp_rnn    = np.array(results['mul']['rnn_seq_len_5']['ccgp'], dtype=float)
ccgp_mrnn   = np.array(results['mul']['rnn_seq_len_50']['ccgp'], dtype=float)
ccgp_ng_mul = np.array(results['mul']['neuragem']['ccgp'], dtype=float)
ccgp_ng_add = np.array(results['add']['neuragem']['ccgp'], dtype=float)
ccgp_arrays = [ccgp_rnn, ccgp_mrnn, ccgp_ng_mul, ccgp_ng_add]
ccgp_arrays = [a[~np.isnan(a)] for a in ccgp_arrays]
ccgp_stats_strings = []
ax_ccgp = axes[3]
for i, arr in enumerate(ccgp_arrays):
    if len(arr) > 1:
        t_stat, p_val = ttest_1samp(arr, 0.5, nan_policy='omit')
        stars = p_to_star(p_val)
        y_bar = ccgp_means[i] + ccgp_sems[i] + 0.015
        annotate_bar_star(ax_ccgp, i, y_bar, stars)
        ccgp_stats_strings.append(f"{labels[i]} vs chance: t={t_stat:.2f}, p={p_val:.2e} ({stars})")
    else:
        ccgp_stats_strings.append(f"{labels[i]} vs chance: insufficient data")

for i, (lab, arr) in enumerate(zip(labels[1:], ccgp_arrays[1:]), start=1):
    if len(ccgp_rnn) > 1 and len(arr) > 1:
        t_stat, p_val = ttest_ind(ccgp_rnn, arr, equal_var=False, nan_policy='omit')
        ccgp_stats_strings.append(f"RNN vs {labels[i]} CCGP: t={t_stat:.2f}, p={p_val:.2e} ({p_to_star(p_val)})")

print("\n=== Statistical tests (uncorrected p-values) ===")
print("Cosine similarity:")
for s in cos_stats_strings: print("  "+s)
print("Weights (z -> hidden):")
for s in weight_stats_strings: print("  "+s)
print("CCGP:")
for s in ccgp_stats_strings: print("  "+s)
print("NOTE: No multiple-comparisons correction applied.")

stats_summary = {
    'cosine': cos_stats_strings,
    'weights': weight_stats_strings,
    'ccgp': ccgp_stats_strings
}

legend_stats_text = " | ".join(cos_stats_strings + weight_stats_strings + ccgp_stats_strings)
# =================== End statistical annotations =================== #

ax = axes[0]
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 1.14, pos.height])  

ax = axes[1]
pos = ax.get_position()
ax.set_position([pos.x0+ 0.08, pos.y0, pos.width * .5, pos.height])  

ax = axes[2]
pos = ax.get_position()
ax.set_position([pos.x0+0.04, pos.y0, pos.width * .5, pos.height])  

ax = axes[3]
pos = ax.get_position()
ax.set_position([pos.x0+0.05, pos.y0, pos.width * 1.14, pos.height])  


# fig.tight_layout()
out_path = os.path.join(export_base_path, f"neural_ng_add_mul_comparison_{'CCGP' if compute_CCGP else ''}.pdf")
fig.savefig(out_path, bbox_inches='tight', transparent=True)
print(f"Saved summary figure to: {out_path}")
print("\nLegend suggestions (copy as needed):")
for line in cos_stats_strings + weight_stats_strings + ccgp_stats_strings:
    print(line)
