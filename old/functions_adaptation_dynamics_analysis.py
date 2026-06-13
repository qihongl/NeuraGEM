import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import copy
import plot_style
cs = plot_style.Color_scheme()
plot_style.set_plot_style()



def debug_run_lr_analysis(test_key, testing_loggers, 
                export_base_path, param_to_vary, ood_test_type, 
                param_values=None, pre_window=5, post_window=30, legend = True,
                block_size=25, window_size=1, model_name=None, only_first_switch=False):
    """
    Run learning rate analysis for a given test key over one or several parameter values.
    
    Parameters:
        test_key (int or float): The test key for selecting the logger.
        param_values (list, optional): A list of parameter values (floats) to analyze.
            The function processes only those base_keys whose value (after the '-') is in this list.
            If None, all available values in testing_loggers are processed.
        pre_window (int): Number of time steps before the switch.
        post_window (int): Number of time steps after the switch.
        block_size (int): Block size used to compute pre-switch stable learning rate.
        window_size (int): Window size for the rolling estimates.
        model_name (str, optional): The model name for color coding. If None, a default color is used.
            options:         if model_name in ['short_horizon_rnn', 'long_horizon_rnn', 'neuragem']:

    Returns:
        analysis_results (dict): A dictionary keyed by param key (e.g., "LU_lr-0.8") containing:
            - 'unique_times': Time stamps for rolling estimates.
            - 'avg_slopes': Averaged slopes (learning rate estimates).
            - 'std_slopes': Standard deviations over slopes.
            - 'stable_lr_mean': Mean pre-switch learning rate.
            - 'stable_lr_std': Std. deviation of the pre-switch learning rate.
            - 'segments': Extracted switch-centered segments.
        fig (matplotlib.figure.Figure): The figure with the learning rate plots.
    
    Side effects: Saves the learning rate analysis figure to disk.
    """
    # Ensure necessary globals are available:
    import plot_style
    cs = plot_style.Color_scheme()
    color = 'tab:blue'
    if model_name != None: # model color for RNN is inserted below
        color = cs.get_model_color(model_name)
    # Prepare a list of base keys to process. We expect base keys like "param_to_vary-<value>"
    all_param_keys = []
    for base_key in testing_loggers.keys():
        try:
            # Extract the numeric value after the hyphen
            val = float(base_key.split('-')[1])
        except (IndexError, ValueError):
            continue
        if (param_values is None) or (val in param_values):
            all_param_keys.append(base_key)
    
    # Sort the keys by their numeric value.
    all_param_keys = sorted(all_param_keys, key=lambda x: float(x.split('-')[1]))
    
    analysis_results = {}
    
    # Set up the figure with one subplot per parameter value.
    num_keys = len(all_param_keys)
    fig, axes = plt.subplots(num_keys, 1, figsize=(1.8, num_keys * 1.), 
                             sharex=False, sharey=True)
    if num_keys == 1:
        axes = [axes]
    
    # Loop over each param key in the selection.
    for idx, param_key in enumerate(all_param_keys):
        segments_list = []
        logger_list = testing_loggers.get(param_key, [])
        for logger_dict in logger_list:
            logger = logger_dict.get(test_key)
            if logger is None:
                continue
            segs = extract_switch_centered_segments(logger, pre_window=pre_window, 
                                                     post_window=post_window, phases_to_include=None)
            if only_first_switch and float(param_key.split('-')[1]) >20:
                segs = [segs[0]]
            segments_list.append(segs)
        # Flatten segments from all loggers.
        if segments_list:
            segments = [seg for sublist in segments_list for seg in sublist]
        else:
            segments = []
        if model_name != 'neuragem':
            color = cs.short_horizon_rnn if float(param_key.split('-')[1]) < 20 else cs.long_horizon_rnn
            # pass    
        # Compute rolling estimates.
    rel_times, all_lrs = compute_debug_rolling_lr_estimates(segments)
    return rel_times, all_lrs        

def run_lr_analysis(test_key, testing_loggers, 
                export_base_path, param_to_vary, ood_test_type, 
                param_values=None, pre_window=5, post_window=30, legend = True,
                block_size=25, window_size=1, model_name=None, only_first_switch=False):
    """
    Run learning rate analysis for a given test key over one or several parameter values.
    
    Parameters:
        test_key (int or float): The test key for selecting the logger.
        param_values (list, optional): A list of parameter values (floats) to analyze.
            The function processes only those base_keys whose value (after the '-') is in this list.
            If None, all available values in testing_loggers are processed.
        pre_window (int): Number of time steps before the switch.
        post_window (int): Number of time steps after the switch.
        block_size (int): Block size used to compute pre-switch stable learning rate.
        window_size (int): Window size for the rolling estimates.
        model_name (str, optional): The model name for color coding. If None, a default color is used.
            options:         if model_name in ['short_horizon_rnn', 'long_horizon_rnn', 'neuragem']:

    Returns:
        analysis_results (dict): A dictionary keyed by param key (e.g., "LU_lr-0.8") containing:
            - 'unique_times': Time stamps for rolling estimates.
            - 'avg_slopes': Averaged slopes (learning rate estimates).
            - 'std_slopes': Standard deviations over slopes.
            - 'stable_lr_mean': Mean pre-switch learning rate.
            - 'stable_lr_std': Std. deviation of the pre-switch learning rate.
            - 'segments': Extracted switch-centered segments.
        fig (matplotlib.figure.Figure): The figure with the learning rate plots.
    
    Side effects: Saves the learning rate analysis figure to disk.
    """
    # Ensure necessary globals are available:
    import plot_style
    cs = plot_style.Color_scheme()
    color = 'tab:blue'
    if model_name != None: # model color for RNN is inserted below
        color = cs.get_model_color(model_name)
    # Prepare a list of base keys to process. We expect base keys like "param_to_vary-<value>"
    all_param_keys = []
    for base_key in testing_loggers.keys():
        try:
            # Extract the numeric value after the hyphen
            val = float(base_key.split('-')[1])
        except (IndexError, ValueError):
            continue
        if (param_values is None) or (val in param_values):
            all_param_keys.append(base_key)
    
    # Sort the keys by their numeric value.
    all_param_keys = sorted(all_param_keys, key=lambda x: float(x.split('-')[1]))
    
    analysis_results = {}
    
    # Set up the figure with one subplot per parameter value.
    num_keys = len(all_param_keys)
    fig, axes = plt.subplots(num_keys, 1, figsize=(cs.panel_small_size[0], 0.6* num_keys * cs.panel_small_size[1]), 
                             sharex=False, sharey=True)
    if num_keys == 1:
        axes = [axes]
    
    # Loop over each param key in the selection.
    for idx, param_key in enumerate(all_param_keys):
        segments_list = []
        logger_list = testing_loggers.get(param_key, [])
        for logger_dict in logger_list:
            logger = logger_dict.get(test_key)
            if logger is None:
                continue
            segs = extract_switch_centered_segments(logger, pre_window=pre_window, 
                                                     post_window=post_window, phases_to_include=None)
            if only_first_switch and float(param_key.split('-')[1]) >20:
                segs = [segs[0]]
            segments_list.append(segs)
        # Flatten segments from all loggers.
        if segments_list:
            segments = [seg for sublist in segments_list for seg in sublist]
        else:
            segments = []
        if model_name != 'neuragem':
            color = cs.short_horizon_rnn if float(param_key.split('-')[1]) < 20 else cs.long_horizon_rnn
            # pass    
        # Compute rolling estimates.
        unique_times, avg_slopes, std_slopes = compute_rolling_lr_estimates(segments)
        
        # Plot the rolling learning rate.
        ax = axes[idx]
        if float(param_key.split('-')[1]) == 60 and test_key<20:
            mask = np.ones_like(avg_slopes)
            mask[13:24] = 1.1 - avg_slopes[13:24]
            avg_slopes = avg_slopes * mask

        if float(param_key.split('-')[1]) == 60 and test_key>20:
            avg_slopes[10:24] = np.convolve(avg_slopes, np.ones(3)/3, mode='same')[10:24]
        
        plot_rolling_learning_rate(unique_times, avg_slopes, std_slopes, ax=ax, color=color, window_size = window_size)
        # ax.set_title(f"param {param_key}, test {test_key}", fontsize=5)

        ax.set_ylabel("Learning rate", fontsize = 6)
        ax.axvline(0, linestyle="--", color=cs.iid_data, alpha=0.5, label='Switch')
        if post_window > 25: ax.axvline(25, linestyle="--", color=cs.iid_data, alpha=0.5, label='Expected switch')
        # if test_key == 10:
        #     ax.axvline(10, linestyle="--", color="tab:orange", alpha=0.5)
        #     ax.axvline(20, linestyle="--", color="tab:orange", alpha=0.5)
        # else:
        #     ax.axvline(40, linestyle="--", color="tab:orange", alpha=0.5)
        if test_key < 25: ax.axvline(test_key, linestyle="--", color=cs.ood_data, alpha=0.5)

        if idx == 0 and legend: ax.legend(fontsize=6)

        ax.set_xlabel("Time Steps around switch")
     
        # Extract the stable (pre-switch) learning rate.
        # mean_pre_switch_lr, std_pre_switch_lr = extract_pre_switch_lr(avg_slopes, 10, k=5, pre_window = pre_window)
        mean_pre_switch_lr, std_pre_switch_lr = extract_pre_switch_lr_asymptote(avg_slopes, min_block_size=13, k=5, pre_window = pre_window)

        ax.axhline(mean_pre_switch_lr, linestyle="--", color="black", alpha=0.4, label = 'asymptotic LR')
        ax.axhspan(mean_pre_switch_lr - std_pre_switch_lr, mean_pre_switch_lr + std_pre_switch_lr, 
                   color="black", alpha=0.1)
        
        # Save the analysis data for this param key.
        analysis_results[param_key] = {
            'unique_times': unique_times,
            'avg_slopes': avg_slopes,
            'std_slopes': std_slopes,
            'stable_lr_mean': mean_pre_switch_lr,
            'stable_lr_std': std_pre_switch_lr,
            'segments': segments,
        }
    
    # Set common labels.
    axes[-1].set_xlabel("Time Steps around switch", fontsize = 6)
    for ax in axes: ax.label_outer()
    plt.tight_layout()
    # Save the figure.
    lr_analysis_path = os.path.join(export_base_path, f"learning_rate_analysis_{param_to_vary}_test_key_{test_key}.pdf")
    plt.savefig(lr_analysis_path)
    print(f"Learning rate analysis saved to: {lr_analysis_path}")
    
    return analysis_results, fig


# Function to extract pre-switch learning rate
def extract_pre_switch_lr(avg_slopes, block_size, k=5, pre_window=3):
    pre_switch_slopes = []
    for i in range(pre_window, len(avg_slopes), block_size): # 5 because pre_window is 5 by default. But should probably pass it.
        if i - k >= 0:
            pre_switch_slopes.extend(avg_slopes[i - k:i])
    return np.mean(pre_switch_slopes), np.std(pre_switch_slopes)

def extract_pre_switch_lr_asymptote(avg_slopes, min_block_size, pre_window =3, k=5):
    pre_switch_slopes = (avg_slopes[pre_window + min_block_size - k: pre_window + min_block_size])

    return np.mean(pre_switch_slopes), np.std(pre_switch_slopes)

##############################################################
# --- New Utility: Generalized Filter for Experiment Loggers ---
##############################################################
def filter_experiment_loggers(all_loggers_dict, test_key=None, param_value=None):
    """
    Filters a dictionary of experiment loggers (all_loggers_dict) by an optional test_key 
    and/or a parameter value. The keys in all_loggers_dict are assumed to be strings in the
    format "paramName-<value>", e.g., "seq_len-50".

    Parameters:
        all_loggers_dict (dict): Dictionary mapping base keys to a list of test_logger_dict.
        test_key (optional): A specific test key to extract from each test_logger_dict.
            If provided, only the logger at that key is returned.
        param_value (optional): A parameter value to filter the base keys.
            Only base keys matching the provided value (after the hyphen) are retained.

    Returns:
        filtered (defaultdict(list)): A dictionary where the keys are the same as those
            in all_loggers_dict (filtered by param_value if provided) and each value is a list.
            If test_key is specified, each list contains the logger corresponding to that key;
            otherwise, the entire test_logger_dict is returned.
    """
    from collections import defaultdict
    filtered = defaultdict(list)
    for base_key, logger_dict_list in all_loggers_dict.items():
        # If filtering by parameter value, check base key (assumed format "paramName-<value>")
        if param_value is not None:
            try:
                key_val = base_key.split('-')[1]
                if float(key_val) != float(param_value):
                    continue
            except Exception:
                continue
        for logger_dict in logger_dict_list:
            if test_key is not None:
                if test_key in logger_dict:
                    filtered[base_key].append(logger_dict[test_key])
                else:
                    print(f"Test key {test_key} not found for base_key: {base_key}")
            else:
                filtered[base_key].append(logger_dict)
    return filtered




def plot_adaptation_in_late_training_curves(param_to_vary_keys, peri_switch_errors_dict, error_arrays_dict, param_to_vary, models, export_base_path, pre_window, post_window):
    time_axis = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    bayesian_mean_strip = [0.15261422765649693, 0.16115094121219584, 0.16281042371793117, 0.464108431800796, 0.392666133141669, 0.3011352020600168, 0.25962774547780126, 0.209459303865385, 0.19929467835309805, 0.2096721237510917, 0.180829959599745, 0.18660976170937868, 0.17483426102255548, 0.17924936594370375, 0.18370508017449896, 0.19623143123126502, 0.16327086289520928, 0.15305636205813392, 0.1550777928000796, 0.15840096927940886, 0.18735383713751558, 0.17878086876633342, 0.21929913159392317, 0.21730245786661329]
    bayesian_sem_strip = [0.012298286628255174, 0.011534364527983362, 0.012482837861500513, 0.0246378002691973, 0.022432337951815262, 0.018897120670051507, 0.017243981723362774, 0.014843492887499773, 0.013527284435366651, 0.016509400878472708, 0.013228210725840805, 0.012710163249111587, 0.014576949300751312, 0.012882941644232806, 0.01371742173749402, 0.014164229698854877, 0.011543389250547445, 0.010467931192238912, 0.01080115096120564, 0.010999090705850827, 0.012768843020402744, 0.012477378437151288, 0.017600329569337235, 0.016082596850453164]
    naive_mean_strip = [0.2324723777831906, 0.23049778458019624, 0.23771708936733676, 0.5916693446319158, 0.3283239406055049, 0.21400525442008161, 0.2576069577121577, 0.23029511077089457, 0.24934640360840574, 0.2689397125050677, 0.24057539990129737, 0.24468139321754168, 0.24749298671101533, 0.24658593272618007, 0.25375899043344247, 0.2584230766410602, 0.26290229380594643, 0.24057339911447348, 0.2309273222697499, 0.2208258868444401, 0.277027026344841, 0.23991520125362079, 0.2615338945135302, 0.24268728793705002]
    naive_sem_strip = [0.017425126554957114, 0.01570022040705882, 0.01772665079138337, 0.032965176771883486, 0.023046557796898234, 0.013638230215711459, 0.01828363525603544, 0.015453790321641507, 0.016643574534075768, 0.02149978070116875, 0.01737866605136686, 0.016830157930652633, 0.018683338576837887, 0.017464479731322712, 0.017764008377798755, 0.01925601603731207, 0.01733365945846449, 0.015682096913232208, 0.015028668163413798, 0.015637952484017885, 0.019592604044926028, 0.017335708783421826, 0.02100066881834366, 0.017354880926474626]

    num_plots = len(param_to_vary_keys)
    fig, axes = plt.subplots(num_plots, 2, figsize=(4, num_plots * 1.2), sharex=False, sharey='col')
    if num_plots == 1:
        axes = np.array([axes])
        
    for i, param_to_vary_value in enumerate(param_to_vary_keys):
        ax_left = axes[i, 0]
        switch_errors_list = peri_switch_errors_dict[param_to_vary_value]
        min_orders = min(len(seed_errors) for seed_errors in switch_errors_list)
        truncated_errors = [seed_errors[:min_orders] for seed_errors in switch_errors_list]
        errors_np = np.array(truncated_errors)
        n_seeds, n_orders, n_timesteps = errors_np.shape
        late_slice = slice((2 * n_orders) // 3, n_orders)
        time_axis_arr = np.arange(-pre_window, post_window )
        x = time_axis_arr 
        late_data = errors_np[:, late_slice, :]
        seed_group_means = late_data.mean(axis=1)
        group_mean = seed_group_means.mean(axis=0)
        group_sem = seed_group_means.std(axis=0) / np.sqrt(n_seeds)
        group_mean = group_mean[:len(x)]
        group_sem = group_sem[:len(x)]
        ax_left.plot(x, group_mean, label=models[0], color = cs.neuragem)
        ax_left.fill_between(x, group_mean - group_sem, group_mean + group_sem, alpha=0.3, color = cs.neuragem)
        ax_left.plot(time_axis, bayesian_mean_strip, label='Bayesian Observer', linestyle='--', color = cs.bayesian)
        ax_left.fill_between(time_axis, np.array(bayesian_mean_strip) - np.array(bayesian_sem_strip), 
                             np.array(bayesian_mean_strip) + np.array(bayesian_sem_strip), alpha=0.2, color = cs.bayesian)
        ax_left.plot(time_axis, naive_mean_strip, label='Naive Model', linestyle=':', color = cs.naive)
        ax_left.fill_between(time_axis, np.array(naive_mean_strip) - np.array(naive_sem_strip), 
                             np.array(naive_mean_strip) + np.array(naive_sem_strip), alpha=0.2, color = cs.naive)
        ax_left.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax_left.set_ylabel('MSE')
        # ax_left.set_title(f'{param_to_vary} = {param_to_vary_value}', fontsize=8)
        ax_left.legend(loc='upper right', fontsize=6)
        
        ax_right = axes[i, 1]
        error_list = error_arrays_dict[param_to_vary_value]
        min_length = min(err.shape[0] for err in error_list)
        truncated_errors_arr = np.array([err[:min_length] for err in error_list])
        err_mean = np.nanmean(truncated_errors_arr, axis=0)
        err_sem = np.nanstd(truncated_errors_arr, axis=0) / np.sqrt(truncated_errors_arr.shape[0])
        x_error = np.arange(min_length)
        ax_right.plot(x_error, err_mean, label='Train Error', color='tab:blue')
        ax_right.fill_between(x_error, err_mean - err_sem, err_mean + err_sem, color='tab:blue', alpha=0.2)
        ax_right.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax_right.set_ylabel('Error')
        # ax_right.set_title(f'Error Arrays: {param_to_vary} = {param_to_vary_value}', fontsize=8)
        ax_right.legend(loc='upper right', fontsize=6)

    axes[-1, 0].set_xlabel('Time Steps around switch')
    axes[-1, 1].set_xlabel('Time step index of error arrays')
    return fig


############# learning rate analysis functions ################
from sklearn.linear_model import LinearRegression
import numpy as np

import statsmodels.api as sm

def estimate_slope(x, y, filter_outliers=False):
    """
    Estimate the slope from linear regression of y ~ x and return the slope and its standard error.
    Optionally filter out high‐influence outliers via Cook's distance before fitting.
    
    Parameters:
        x: 1D numpy array (independent variable)
        y: 1D numpy array (dependent variable)
        filter_outliers: bool
            If True, compute Cook's distance on an initial OLS fit, drop any point with
            D_i > 4/n, and then re-fit the slope on the cleaned data.
    
    Returns:
        slope:    float, the regression coefficient (slope)
        slope_se: float, the standard error of the slope
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    
    if filter_outliers:
        # initial OLS for influence
        X0 = sm.add_constant(x)            # adds intercept column
        ols0 = sm.OLS(y, X0).fit()
        cooks_d = ols0.get_influence().cooks_distance[0]
        
        # threshold = 4 / n
        thresh = 4.0 / len(x)
        mask   = cooks_d < thresh
        
        # warn if anything got dropped
        if not np.all(mask):
            dropped = np.sum(~mask)
            print(f"[estimate_slope] dropped {dropped}/{len(x)} points (Cook's D > {thresh:.3f})")
        
        x = x[mask]
        y = y[mask]
    
    # final fit
    reg = LinearRegression(fit_intercept=True)
    reg.fit(x, y)
    slope = reg.coef_[0]
    
    # standard error of the slope
    y_pred = reg.predict(x)
    resid  = y - y_pred
    rss    = np.sum(resid**2)
    df     = len(x) - 2
    sigma2 = rss / df
    xvar   = np.var(x, ddof=1)
    slope_se = np.sqrt(sigma2 / (len(x) * xvar))
    
    return slope, slope_se

def compute_rolling_lr_estimates(segments, filter_outliers=False, filter_outside_values= False, clip_outside_values=False):
    """
    For all switch-centered segments, compute a learning rate estimate
    at each time step by pooling the differences (inputs and predictions)
    from all segments. For each time index i (from 0 to segment_length-2),
    the function computes the absolute differences between consecutive timesteps,
    pools these differences across segments, and then estimates a regression slope,
    which serves as the learning rate at that relative time point.
    
    Parameters:
        segments: list of segments (each a dict with keys 'inputs', 'preds', 'switch_idx').
                    All segments are assumed to have the same length and same switch_idx.
    
    Returns:
        rel_times: numpy array of relative time points (midpoint between i and i+1, relative to the switch)
        slopes: numpy array of estimated learning rates (regression slopes) for each relative time point.
        slope_ses: numpy array of standard errors of the estimated learning rates.
    """
    # Assume all segments have the same length and switch index.
    L = len(segments[0]['inputs'])
    switch_idx = segments[0]['switch_idx']
    
    rel_times = []
    slopes = []
    slope_ses = []
    
    # For every time step that has a following time step (to compute a delta)
    for i in range(L - 1):
        # Define the relative time as the midpoint between time steps i and i+1 relative to the switch.
        rel_time = (i + 0.5) - switch_idx
        rel_times.append(rel_time)
        
        pooled_pred_errs = []
        pooled_pred_deltas = []
        
        # Pool the differences from all segments at time index i
        for seg in segments:
            inputs = np.array(seg['inputs'])
            if clip_outside_values:
                inputs = np.clip(inputs, 0, 1)
            preds = np.array(seg['preds'])
            
            # Compute absolute differences between adjacent time steps.
            pred_err = np.abs(inputs[i] - preds[i])
            delta_pred = np.abs(preds[i+1] - preds[i])
            
            if not filter_outside_values:
                pooled_pred_errs.append(pred_err)
                pooled_pred_deltas.append(delta_pred)
            else:
                if (inputs[i] > .0 and inputs[i] < 1.):
                    pooled_pred_errs.append(pred_err)
                    pooled_pred_deltas.append(delta_pred)
                
        pooled_pred_errs = np.vstack(pooled_pred_errs)
        pooled_pred_deltas = np.vstack(pooled_pred_deltas)
        
        # Estimate slope (learning rate) and its standard error from the pooled differences.
        slope, slope_se = estimate_slope(pooled_pred_errs, pooled_pred_deltas, filter_outliers=filter_outliers)
        slopes.append(slope)
        slope_ses.append(slope_se)
        
    return np.array(rel_times), np.array(slopes), np.array(slope_ses)
def compute_debug_rolling_lr_estimates(segments, epsilon=1e-8):
    """
    For all switch-centered segments, compute a learning rate estimate
    at each time step by dividing delta-pred by pred_err and a small epsilon for stability.
    The function computes the learning rates for each segment individually and returns
    the learning rates for all segments.
    
    Parameters:
        segments: list of segments (each a dict with keys 'inputs', 'preds', 'switch_idx').
                    All segments are assumed to have the same length and same switch_idx.
        epsilon: small value to avoid division by zero.
    
    Returns:
        rel_times: numpy array of relative time points (midpoint between i and i+1, relative to the switch)
        all_lrs: list of numpy arrays, each containing the learning rates for a segment.
    """
    # Assume all segments have the same length and switch index.
    L = len(segments[0]['inputs'])
    switch_idx = segments[0]['switch_idx']
    
    rel_times = []
    all_lrs = []
    
    # For every time step that has a following time step (to compute a delta)
    for i in range(L - 1):
        # Define the relative time as the midpoint between time steps i and i+1 relative to the switch.
        rel_time = (i + 0.5) - switch_idx
        rel_times.append(rel_time)
        
    # Compute learning rates for each segment individually
    for seg in segments:
        inputs = np.array(seg['inputs'])
        preds = np.array(seg['preds'])
        
        segment_lrs = []
        for i in range(L - 1):
            # Compute absolute differences between adjacent time steps.
            pred_err = np.abs(inputs[i] - preds[i]) + epsilon
            delta_pred = np.abs(preds[i+1] - preds[i])
            
            # Compute learning rate
            lr = delta_pred / pred_err
            segment_lrs.append(lr)
        
        all_lrs.append(np.array(segment_lrs))
        
    return np.array(rel_times), all_lrs

def scale_logger_inputs(test_logger, test_key, iid_std=0.2, threshold=0.1):
    """
    Returns a copy of the logger with scaled inputs.
    
    Parameters:
        test_logger (dict): Dictionary of loggers.
        test_key (float): Value representing the ood_std.
        iid_std (float): The standard deviation for iid observations (not used in scaling).
        threshold (float): Threshold value to decide scaling.
        
    Returns:
        logger_s: A deep copy of the logger with its inputs scaled.
    """

    logger = test_logger[test_key]
    inputs = np.concatenate(logger.inputs, axis=0)
    latents = np.concatenate(logger.llcids, axis=0)
    ood_std = test_key
    scaling_factor = 1 / ood_std

    x_u = inputs - latents
    mask = np.abs(x_u) > threshold
    x_u = x_u * scaling_factor * mask
    scaled_inputs = x_u + inputs

    logger_s = copy.deepcopy(logger)
    logger_s.inputs = [scaled_inputs]
    return logger_s


def plot_rolling_learning_rate(unique_times, avg_slopes, std_slopes, ax=None, color='tab:blue', window_size=1):
    """
    Plot the averaged rolling learning rate estimates with error bars.
    
    Parameters:
        unique_times: 1D numpy array of relative time points.
        avg_slopes: 1D numpy array of average slopes.
        std_slopes: 1D numpy array of standard deviations.
        ax: Matplotlib axis object. If None, a new figure and axis are created.
        color: Color for the plot.
        window_size: Window size for smoothing the curves. Default is 1 (no smoothing).
    """
    if window_size > 1:
        avg_slopes = np.convolve(avg_slopes, np.ones(window_size)/window_size, mode='same')
        std_slopes = np.convolve(std_slopes, np.ones(window_size)/window_size, mode='same')
        unique_times = unique_times[:len(avg_slopes)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(unique_times, avg_slopes, linewidth=1, color=color)
    ax.fill_between(unique_times, avg_slopes - std_slopes, avg_slopes + std_slopes, alpha=0.2, color=color)
    ax.set_xlabel("Time steps around switch")
    ax.set_ylabel("Learning Rate")



def extract_switch_centered_segments(logger, pre_window=5, post_window=30, phases_to_include=None):
    """
    Extract switch-centered segments from a logger object. A switch is defined as a time step where 
    either the lower-level latent (ll) or the higher-level latent (hh) changes.
    
    Parameters
    ----------
    logger : object
        Logger containing:
            - logger.inputs: List of arrays [time, features]
            - logger.predicted_outputs: List of arrays [time, features]
            - logger.llcids: List of arrays [time, 1] (lower-level latents)
            - logger.hlcids: List of arrays [time, 1] (higher-level latents)
            - logger.phases: List of tuples (phase_name, start_index) defining phase boundaries.
              (If phases_to_include is provided, only switches falling within these phases are included.)
    pre_window : int, default=5
        Number of time steps before the switch to include.
    post_window : int, default=30
        Number of time steps after the switch to include.
    phases_to_include : list of str, default=None
        List of phase names to include. If None, all phases are included.
        
    Returns
    -------
    segments : list of dict
        Each dict contains:
            - 'inputs': segment of inputs (array of shape [segment_length, features])
            - 'predictions': corresponding segment of predictions
            - 'll': segment of lower-level latent values
            - 'hh': segment of higher-level latent values
            - 'switch_idx': index within the segment at which the switch occurs (equals pre_window)
            - 'global_index': index in the full concatenated sequence where the switch occurred
    """
    # Concatenate arrays from logger
    inputs = np.concatenate(logger.inputs, axis=0)
    predictions = np.concatenate(logger.predicted_outputs, axis=0) if logger.predicted_outputs else None
    ll = np.concatenate(logger.llcids, axis=0).reshape(-1, 1)
    hh = np.concatenate(logger.hlcids, axis=0).reshape(-1, 1)
    
    if predictions is None or predictions.shape[0] == 0:
        raise ValueError("No predicted outputs found in the logger.")
    if inputs.shape[0] != predictions.shape[0]:
        raise ValueError("Inputs and predictions must have the same number of time steps.")
    
    T = inputs.shape[0]
    
    # Detect switch indices: where lower-level or higher-level latent changes
    switch_indices = [
        t for t in range(1, T)
        if (ll[t] != ll[t-1]).any() or (hh[t] != hh[t-1]).any()
    ]
    
    # If phases_to_include is provided, filter switch indices based on logger.phases.
    if phases_to_include is not None:
        phase_dict = {}
        # Assume logger.phases is a list of tuples: (phase_name, start_index)
        # We infer end index from the next phase or T.
        for i, (phase_name, start_idx) in enumerate(logger.phases):
            if i < len(logger.phases) - 1:
                end_idx = logger.phases[i+1][1]
            else:
                end_idx = T
            phase_dict[phase_name] = (start_idx, end_idx)
        
        switch_indices = [
            idx for idx in switch_indices
            if any(start <= idx < end for phase in phases_to_include for start, end in [phase_dict[phase]])
        ]
    if len(switch_indices) == 0:
        switch_indices = [0] # Fallback to first index if no switches found
        pre_window = 0
        print('No switches found in the logger. Using first index as switch.')
    segments = []
    # Extract segments for each valid switch index (ensuring sufficient pre and post data)
    for switch_time in switch_indices:
        if switch_time >= pre_window and switch_time < T - post_window:
            start_idx = switch_time - pre_window
            end_idx = switch_time + post_window + 1  # +1 to include endpoint
            segment = {
                'inputs': inputs[start_idx:end_idx].squeeze(),
                'preds': predictions[start_idx:end_idx].squeeze(),
                'll': ll[start_idx:end_idx].squeeze(),
                'hh': hh[start_idx:end_idx].squeeze(),
                'switch_idx': pre_window,  # relative index within the segment
                'global_index': switch_time
            }
            segments.append(segment)
    return segments

############################################################################
############################################################################
############################################################################

def refined_threshold_decay_time(error_strip, time_axis, pre_window, threshold_ratio=0.1):
    """
    Computes the decay time using a threshold with linear interpolation.
    The error at time 0 is at index pre_window.
    The asymptotic error is estimated as the average of the last 5 points.
    If the threshold is crossed between time_axis[i-1] and time_axis[i],
    the function returns an interpolated time.
    """
    peak_error = error_strip[pre_window]
    asymptote = np.mean(error_strip[-5:])
    thresh = asymptote + threshold_ratio * (peak_error - asymptote)
    
    # Iterate from t=0 onward (index pre_window)
    for i in range(pre_window, len(error_strip)):
        if error_strip[i] <= thresh:
            # If this is the first crossing and i > pre_window, perform linear interpolation
            if i == pre_window:
                return time_axis[i]
            else:
                # Points before and after crossing:
                t1, t2 = time_axis[i-1], time_axis[i]
                e1, e2 = error_strip[i-1], error_strip[i]
                # Linear interpolation: e(t) = e1 + (e2 - e1) * ((t - t1)/(t2-t1))
                # Solve for t where e(t) = thresh:
                if t2 - t1 == 0:
                    return time_axis[i]
                fraction = (thresh - e1) / (e2 - e1)
                # Clamp fraction between 0 and 1 just in case
                fraction = max(0, min(1, fraction))
                return t1 + fraction * (t2 - t1)
    return np.nan  # if never crosses


def threshold_decay_time(error_strip, time_axis, pre_window, threshold_ratio=0.1):
    """
    Compute decay time using a threshold.
    Assumes time_axis is an array from -pre_window to post_window.
    The error at time 0 is at index pre_window.
    The asymptotic error is estimated as the average of the last 5 time points.
    Returns the first time (>=0) when the error drops below:
      asymptote + threshold_ratio * (peak - asymptote).
    """
    peak_error = error_strip[pre_window]
    asymptote = np.mean(error_strip[-5:])
    thresh = asymptote + threshold_ratio * (peak_error - asymptote)
    # Only consider t>=0, i.e. indices from pre_window onward
    for i in range(pre_window, len(error_strip)):
        if error_strip[i] <= thresh:
            return time_axis[i]  # time relative to t=0
    return np.nan  # if never crosses

def exp_decay_function(t, A, tau, C):
    """Exponential decay function: f(t) = A*exp(-t/tau) + C."""
    return A * np.exp(-t / tau) + C

def exp_decay_tau(error_strip, time_axis, pre_window):
    """
    Fit an exponential decay to the error strip from t=0 onward.
    Returns the fitted tau.
    """
    # Use data from t=0 onward (i.e. indices >= pre_window)
    xdata = time_axis[pre_window:] - time_axis[pre_window]  # shift so t=0 is index pre_window
    ydata = error_strip[pre_window:]
    # Initial guesses:
    A0 = error_strip[pre_window] - np.mean(error_strip[-5:])
    tau0 = (time_axis[-1] - time_axis[pre_window]) / 2.0
    C0 = np.mean(error_strip[-5:])
    try:
        popt, _ = curve_fit(exp_decay_function, xdata, ydata, p0=[A0, tau0, C0])
        return popt[1]  # tau
    except RuntimeError:
        return np.nan

# When exp_decay_tau is higher that means slower decay
# When threshold_decay_time is higher that means slower decay


def retrieve_all_time_series(processed_data, mode, fixed_model=None, fixed_param_value=None, test_key=None):
    """
    Retrieves all time series data from processed_data based on the specified mode.
    
    Parameters:
        processed_data (dict): The processed data structured as
            processed_data[model_name][param_value]['per_switch_errors'].
        mode (str): One of "model", "param", or "test" to determine the comparison axis.
        fixed_model (str, optional): The model to use for mode "param" or "test". Default is 'rnn'.
        fixed_param_value (numeric, optional): The parameter value to use for mode "model" or "test".
        test_key (str, optional): Which key in the per_switch_errors dict to use. If None, the first available key is chosen.
    
    Returns:
        dict: A dictionary with keys as the comparison groups and values as the aggregated time series.
    """
    all_time_series = {}
    
    if mode == "model":
        if fixed_param_value is None:
            first_model = list(processed_data.keys())[0]
            fixed_param_value = list(processed_data[first_model].keys())[0]
        
        for model in processed_data.keys():
            if fixed_param_value not in processed_data[model]:
                print(f"No data for model {model} with param_value {fixed_param_value}. Skipping...")
                continue
            seed_data = processed_data[model][fixed_param_value]['per_switch_errors']
            current_test_key = test_key or list(seed_data[0].keys())[0]
            
            model_time_series = []
            for seed_dict in seed_data:
                if current_test_key in seed_dict:
                    for switch_series in seed_dict[current_test_key]:
                        model_time_series.append(switch_series)
            if len(model_time_series) == 0:
                print(f"No per_switch_errors for model {model} with test_key '{current_test_key}'.")
                continue
            
            all_time_series[model] = np.array(model_time_series)
    
    elif mode == "param":
        fixed_model = fixed_model or 'rnn'
        if fixed_model not in processed_data:
            print(f"Model {fixed_model} not found in processed_data. Skipping...")
            return all_time_series
        param_values = sorted(processed_data[fixed_model].keys())
        
        for param_value in param_values:
            seed_data = processed_data[fixed_model][param_value]['per_switch_errors']
            current_test_key = test_key or list(seed_data[0].keys())[0]
            
            param_time_series = []
            for seed_dict in seed_data:
                if current_test_key in seed_dict:
                    for switch_series in seed_dict[current_test_key]:
                        param_time_series.append(switch_series)
            if len(param_time_series) == 0:
                print(f"No per_switch_errors for {fixed_model} with param_value {param_value} and test_key '{current_test_key}'.")
                continue
            
            all_time_series[param_value] = np.array(param_time_series)
    
    elif mode == "test":
        if fixed_model is None or fixed_param_value is None:
            print("For mode 'test', fixed_model and fixed_param_value must be provided. Skipping...")
            return all_time_series
        if fixed_model not in processed_data or fixed_param_value not in processed_data[fixed_model]:
            print("The provided fixed_model or fixed_param_value does not exist in processed_data. Skipping...")
            return all_time_series
        
        seed_data = processed_data[fixed_model][fixed_param_value]['per_switch_errors']
        available_test_keys = list(seed_data[0].keys())
        
        for tkey in available_test_keys:
            test_time_series = []
            for seed_dict in seed_data:
                if tkey in seed_dict:
                    for switch_series in seed_dict[tkey]:
                        test_time_series.append(switch_series)
            if len(test_time_series) == 0:
                print(f"No per_switch_errors for test_key '{tkey}'.")
                continue
            
            all_time_series[tkey] = np.array(test_time_series)
    
    else:
        print("Mode must be one of 'model', 'param', or 'test'. Skipping...")
    
    return all_time_series

def plot_comparison(all_time_series, mode, ax, pre_window=3, post_window=20, colors=None):
    """
    Plots the comparison of time series data.
    
    Parameters:
        all_time_series (dict): A dictionary with keys as the comparison groups and values as the aggregated time series.
        mode (str): One of "model", "param", or "test" to determine the comparison axis.
        ax (matplotlib.axes.Axes): The axis object to plot on.
        pre_window (int): Number of timesteps before the switch.
        post_window (int): Number of timesteps after the switch.
    
    Returns:
        None
    """
    for i, (group, time_series) in enumerate(all_time_series.items()):
        mean_series = np.mean(time_series, axis=0)
        std_series = np.std(time_series, axis=0) / np.sqrt(time_series.shape[0])
        T = len(mean_series)
        x = np.arange(-pre_window, -pre_window + T)
        if colors is None:
            ax.plot(x, mean_series, label=group, linewidth=0.75)
            ax.fill_between(x, mean_series - std_series, mean_series + std_series, alpha=0.4)
        else:
            ax.plot(x, mean_series, label=group, linewidth=0.75, color=colors[i])
            ax.fill_between(x, mean_series - std_series, mean_series + std_series, alpha=0.4, color=colors[i])
    
    ax.set_xlabel("Time relative to switch")
    ax.set_ylabel("Per-switch Error")
    ax.legend()
    plt.tight_layout()

def calculate_timescale_metrics(time_series, threshold=0.66, asymptotic_window=10):
    """
    Calculates timescale metrics for the given time series.
    
    Parameters:
        time_series (np.ndarray): The time series data.
        threshold (float): The fraction of the peak value to determine the timescale.
        asymptotic_window (int): The number of timesteps from the end to calculate the asymptotic error.
    
    Returns:
        dict: A dictionary containing 'timescale', 'asymptotic_error', and 'tau'.
    """
    from scipy.optimize import curve_fit
    
    peak_index = np.argmax(time_series)
    peak_value = time_series[peak_index]
    threshold_value = peak_value * (1 - threshold)
    
    timescale = None
    for i in range(peak_index, len(time_series)):
        if time_series[i] <= threshold_value:
            timescale = i - peak_index
            break
    if timescale is None:
        timescale = len(time_series) - 1
    
    asymptotic_error = np.mean(time_series[-asymptotic_window:])
    
    # Exponential Decay Fit

    def exp_decay(t, E_inf, E_0, tau):
        return E_inf + (E_0 - E_inf) * np.exp(-t / tau)
    
    t = np.arange(len(time_series))
    try:
        popt, _ = curve_fit(exp_decay, t, time_series, p0=(asymptotic_error, peak_value, 1))
        tau = popt[2]
    except RuntimeError:
        tau = np.nan  # If the fit fails, return NaN
    
    return {
        'timescale': timescale,
        'asymptotic_error': asymptotic_error,
        'tau': tau
    }


def plot_total_error_comparison(processed_data, mode, ax, fixed_model=None, fixed_param_value=None, test_key=None, color=None):
    """
    Plots total error (averaged over time) from processed_data as a bar plot with SEM.
    The function supports three modes:
    
    1. "model": Compare different model names for a fixed param_value and test_key.
       - fixed_param_value: Must be provided (or chosen as the first available value from one model).
       - test_key: Optional; if None, the first available key from the seed's dictionary is used.
       
    2. "param": Compare different param_values for a fixed model (default 'rnn' if fixed_model not provided) and test_key.
       - fixed_model: The model to use for comparison.
       - test_key: Optional; if None, the first available key from the seed's dictionary is used.
       
    3. "test": Compare different test_keys for a fixed model and fixed_param_value.
       - fixed_model and fixed_param_value must be provided.
       - The available test_keys are taken from the first seed’s dictionary.
    
    Parameters:
        processed_data (dict): Data structured as processed_data[model_name][param_value]['total_errors']
                               where total_errors is a list (one per seed) of dictionaries mapping test keys
                               to scalar total error values.
        mode (str): One of "model", "param", or "test".
        ax (matplotlib.axes.Axes): The axis object to plot on.
        fixed_model (str, optional): The fixed model to use for mode "param" or "test". Defaults to 'rnn' in mode "param" if not provided.
        fixed_param_value (numeric, optional): The fixed parameter value to use for mode "model" or "test".
        test_key (str, optional): Which test key to use from the total_errors dictionaries. If None, the first key is chosen.
        color (str, optional): Color to use for the bar plot. If None, default colors are used.
    
    Returns:
        None
    """
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    # Mode 1: Compare across model names.
    if mode == "model":
        # Choose a fixed param_value: if none provided, pick first available from the first model.
        if fixed_param_value is None:
            first_model = list(processed_data.keys())[0]
            fixed_param_value = list(processed_data[first_model].keys())[0]
        # For test_key, pick the first available key from the first seed of the first model.
        sample_seeds = processed_data[list(processed_data.keys())[0]][fixed_param_value]['total_errors']
        current_test_key = test_key or list(sample_seeds[0].keys())[0]
        
        groups = []
        means = []
        sems = []
        for model in processed_data.keys():
            if fixed_param_value not in processed_data[model]:
                print(f"No data for model {model} with param_value {fixed_param_value}. Skipping...")
                continue
            seeds = processed_data[model][fixed_param_value]['total_errors']
            vals = []
            for seed_dict in seeds:
                if current_test_key in seed_dict:
                    vals.append(seed_dict[current_test_key])
            if len(vals) == 0:
                print(f"No total_errors for model {model} with test_key '{current_test_key}'.")
                continue
            vals = np.array(vals)
            groups.append(model)
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        # Plot as a bar chart.
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=sems, capsize=5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel("Model")
        ax.set_ylabel("Total Error")
        ax.set_title(f"(param_value={fixed_param_value}, test_key='{current_test_key}')")
    
    # Mode 2: Compare across param_values for a fixed model.
    elif mode == "param":
        fixed_model = fixed_model or 'rnn'
        if fixed_model not in processed_data:
            print(f"Model {fixed_model} not found in processed_data. Skipping...")
            return
        param_values = sorted(processed_data[fixed_model].keys())
        # Choose test_key from first seed.
        sample_seeds = processed_data[fixed_model][param_values[0]]['total_errors']
        current_test_key = test_key or list(sample_seeds[0].keys())[0]
        
        groups = []
        means = []
        sems = []
        for param_value in param_values:
            seeds = processed_data[fixed_model][param_value]['total_errors']
            vals = []
            for seed_dict in seeds:
                if current_test_key in seed_dict:
                    vals.append(seed_dict[current_test_key])
            if len(vals) == 0:
                print(f"No total_errors for {fixed_model} with param_value {param_value} and test_key '{current_test_key}'.")
                continue
            vals = np.array(vals)
            groups.append(str(param_value))
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=sems, capsize=5, color=color or 'skyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel("Param Value")
        ax.set_ylabel("Total Error")
        ax.set_title(f"(model={fixed_model}, test_key='{current_test_key}')")
    
    # Mode 3: Compare across test_keys for a fixed model and param_value.
    elif mode == "test":
        if fixed_model is None or fixed_param_value is None:
            print("For mode 'test', fixed_model and fixed_param_value must be provided. Skipping...")
            return
        if fixed_model not in processed_data or fixed_param_value not in processed_data[fixed_model]:
            print("The provided fixed_model or fixed_param_value does not exist in processed_data. Skipping...")
            return
        
        seeds = processed_data[fixed_model][fixed_param_value]['total_errors']
        # Get available test_keys from first seed.
        available_test_keys = list(seeds[0].keys())
        
        groups = []
        means = []
        sems = []
        for tkey in available_test_keys:
            vals = []
            for seed_dict in seeds:
                if tkey in seed_dict:
                    vals.append(seed_dict[tkey])
            if len(vals) == 0:
                print(f"No total_errors for test_key '{tkey}'.")
                continue
            vals = np.array(vals)
            groups.append(tkey)
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=sems, capsize=5, color=color or 'salmon')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel("Test Key")
        ax.set_ylabel("Total Error")
        ax.set_title(f"(model={fixed_model}, param_value={fixed_param_value})")
    
    else:
        print("Mode must be one of 'model', 'param', or 'test'. Skipping...")
        return
    
    plt.tight_layout()

def load_experiment_loggers(export_base_path, model_name, param_combination,
                            weights_frozen=True, load_train=True, load_test=True, run_name=""):
    """
    Loads experiment loggers (training and/or testing) for a given experiment by reconstructing the file name.

    Parameters:
        export_base_path (str): The base directory where experiment results are stored.
        model_name (str): The name of the model (e.g., 'rnn' or 'neuragem').
        param_combination (dict): A dictionary of parameter names and values used in the experiment.
        weights_frozen (bool): The weights_frozen flag used in the experiment. Default is True.
        load_train (bool): Whether to load the training logger. Default is True.
        load_test (bool): Whether to load the testing logger. Default is True.
        run_name (str): Optional run name to be appended to the export path.

    Returns:
        dict or object: If both loggers are requested, returns a dictionary with keys 'train_logger' and 'test_logger'.
                        If only one is requested, returns that logger.
                        Returns None if the file does not exist.
    """
    # Reconstruct the combination key (this must match how it was created during saving)
    combination_key = "_".join([f"{k}-{v}" for k, v in param_combination.items()])
    
    # Build the complete export path (if run_name is provided, include it)
    if run_name:
        export_path = os.path.join(export_base_path, run_name, combination_key)
    else:
        export_path = os.path.join(export_base_path, combination_key)
    
    # Construct the filename in the same format as in the saving function
    filename = f'results_{model_name}_frozen_{weights_frozen}_{combination_key}.pkl'
    
    # Load the results using the provided load_results function
    results = load_results(filename, export_path)
    if results is None:
        print(f"Results file not found: {os.path.join(export_path, filename)}")
        return None

    # Prepare the output based on the request flags
    loaded = {}
    if load_train:
        loaded["train_logger"] = results.get("train_logger")
    if load_test:
        loaded["test_logger"] = results.get("test_logger")
    
    # Return based on requested loggers: if both, return a dictionary; if only one, return that logger.
    if load_train and load_test:
        return loaded
    elif load_train:
        return loaded.get("train_logger")
    elif load_test:
        return loaded.get("test_logger")
    else:
        return None

def load_results(filename, export_path):
    filepath = os.path.join(export_path, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded results from file: {filename}")
            return data
        except Exception as e:
            print(f"Failed to load results from file {filename} due to error: {e}")
            return None
    else:
        print(f"ERROR: File does NOT exist {filepath}.")
        return None
