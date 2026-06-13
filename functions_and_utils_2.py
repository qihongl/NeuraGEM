# more recent analysis and plots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

import numpy as np

def calculate_error(logger, error_type='abs_from_mean', pre_window=3, post_window=10, phases_to_include=None, enforce_block_size_min_post_window=False):
    """
    Calculate the error (MSE, absolute, or mean predictions from mean) between true inputs and model predictions.
    Identifies block-switch time steps using changes in lower-level latent (ll)
    or higher-level latent (hh). Returns:
        1) error_array: A 1D array of error values over all time steps.
        2) error_strips: A list of 1D slices of the error_array
                         around each block switch point.
    
    Parameters
    ----------
    logger : object
        Logger containing:
            - logger.inputs: List of arrays [time, features]
            - logger.predicted_outputs: List of arrays [time, features]
            - logger.llcids: List of arrays [time, 1] (lower-level latents)
            - logger.hlcids: List of arrays [time, 1] (higher-level latents)
    error_type : str
        Either 'mse' (mean squared error), 'abs' (absolute error), or 'mean_predictions_from_mean'. Default='mse'.
    pre_window : int
        Number of time steps before a block switch to include. Default=3.
    post_window : int
        Number of time steps after a block switch to include. Default=10.
    phases_to_include : list of str
        List of phase names to include in the analysis. Default=None (all phases).
        Example: [('Learning and inference', 994), ('No inference nor learning', 1212)]
    enforce_block_size_min_post_window : bool
        If True, only include a block if there are at least post_window time steps after the switch.
    
    Returns
    -------
    error_array : np.ndarray
        A 1D array of length T (number of total time steps).
    error_strips : list of np.ndarray
        Each entry is a 1D slice of the error_array around a switch point.
    """

    # Concatenate arrays from logger
    inputs = np.concatenate(logger.inputs, axis=0)
    predictions = np.concatenate(logger.predicted_outputs, axis=0) if logger.predicted_outputs else None
    ll = np.concatenate(logger.llcids, axis=0).reshape(-1, 1)  # Ensure shape [T, 1]
    hh = np.concatenate(logger.hlcids, axis=0).reshape(-1, 1)  # Ensure shape [T, 1]

    # Validate inputs and predictions
    if predictions is None or predictions.shape[0] == 0:
        raise ValueError("No predicted outputs found in the logger.")
    if inputs.shape[0] != predictions.shape[0]:
        raise ValueError("Inputs and predictions must have the same number of time steps.")

    # Initialize error array
    T = inputs.shape[0]
    error_array = np.zeros(T)

    # Compute error at each time step
    for t in range(T):
        if error_type == 'mse':
            error_array[t] = np.mean((predictions[t] - inputs[t])**2)
        elif error_type == 'abs':
            error_array[t] = np.mean(np.abs(predictions[t] - inputs[t]))
        elif error_type == 'abs_from_mean':
            error_array[t] = np.mean(np.abs(predictions[t] - ll[t]))
        elif error_type == 'dist_from_mean':
            error_array[t] = np.mean((predictions[t] - ll[t]))
        elif error_type == 'mean_predictions_from_mean':
            error_array[t] = (predictions[t] - ll[t])
        else:
            raise ValueError("Invalid error_type. Choose either 'mse', 'abs', 'dist_from_mean' or 'abs_from_mean'.")

    # Detect block switches
    switch_indices = [
        t for t in range(1, T)
        if (ll[t] != ll[t - 1]).any() or (hh[t] != hh[t - 1]).any()
    ]
    if phases_to_include is not None:
        # Create a dictionary with phase names as keys and (start_idx, end_idx) as values
        phase_dict = {}
        for i, (phase_name, phase_idx) in enumerate(logger.phases):
            if i < len(logger.phases) - 1:
                next_phase_idx = logger.phases[i + 1][1]
            else:
                next_phase_idx = T
            phase_dict[phase_name] = (phase_idx, next_phase_idx)

        # Filter switch_indices to include only those within the specified phases
        switch_indices = [
            idx for idx in switch_indices
            if any(start <= idx < end for phase in phases_to_include for start, end in [phase_dict[phase]])
        ]
    if len(switch_indices) == 0:
        switch_indices = [0,0] # Fallback to first index if no switches found
        pre_window = 0 # also add 2 0 switches because of the :-1 below
        print('No switches found in the logger. Using first index as switch.')
    if len(switch_indices) == 1: # this is because ood_means exp has one block with the new mean.
        switch_indices.append(T)
    error_strips = []
    for si, switch_time in enumerate(switch_indices[:-1]):
        # Check if there are enough preceding time steps
        if switch_time < pre_window:
            continue
        # If enforcing minimum post_window, skip if the following block is too short
        if enforce_block_size_min_post_window and (switch_indices[si+1] - switch_time) < post_window:
            # print(f"Block {si} is too short ({switch_indices[si+1] - switch_time}). Skipping.")
            continue

        start_idx = switch_time - pre_window
        # end_idx = min(T, switch_time + post_window + 1)  # Include endpoint
        # if end_idx > T or error_array[start_idx:end_idx].size < pre_window + post_window + 1:
        end_idx = min(T, switch_time + post_window )  # not sure why this endpoint above was ever included
        if end_idx > T or error_array[start_idx:end_idx].size < pre_window + post_window:
            print(f"Error strip {si} is too short ({error_array[start_idx:end_idx].size}). Skipping.")
            continue
        error_strips.append(error_array[start_idx:end_idx])

    return error_array, error_strips


def calculate_adaptation_times_and_errors(error_strips, pre_window=3, post_window=10, error_threshold=0.05, error_type='abs_from_mean'):   
    """
    Calculates the adaptation times and total post-window errors for each block switch.

    Parameters
    ----------
    error_strips : list of np.ndarray
        List of error strips (1D arrays) around block switches.
    pre_window : int
        Number of time steps before the switch in each strip. Default=3.
    post_window : int
        Number of time steps after the switch in each strip. Default=10.
    error_threshold : float
        Error threshold to determine adaptation speed. Default=0.05.

    Returns
    -------
    adaptation_times : list of int
        List of time steps to reach the error threshold for each block switch.
    total_post_window_errors : list of float
        List of total errors in the post-window phase for each block switch.
    """
    adaptation_times = []
    total_post_window_errors = []

    for strip in error_strips:
        post_switch_errors = strip[pre_window:]  # Only consider time steps after the switch
        adaptation_time = np.argmax(post_switch_errors <= error_threshold) if np.any(post_switch_errors <= error_threshold) else len(post_switch_errors)
        adaptation_times.append(adaptation_time)

        if error_type == 'mean_predictions_from_mean':
            total_post_window_error = np.mean(np.abs(post_switch_errors))
        else:
            total_post_window_error = np.mean(post_switch_errors)
        total_post_window_errors.append(total_post_window_error)

    return adaptation_times, total_post_window_errors


def plot_error_strips_and_adaptation(error_strips, adaptation_times, pre_window=3, post_window=10, 
                                     end_blocks_to_exclude=0, error_threshold=0.05, title="Error Strips and Adaptation Analysis", cmap_name='viridis'):
    """
    Plots error strips with a colormap and a legend, aligning the x-axis
    such that 0 is the block switch time step. Plots the time to reach an error threshold
    for each block switch in a secondary subplot.

    Parameters
    ----------
    error_strips : list of np.ndarray
        List of error strips (1D arrays) around block switches.
    adaptation_times : list of int
        List of time steps to reach the error threshold for each block switch.
    pre_window : int
        Number of time steps before the switch in each strip. Default=3.
    post_window : int
        Number of time steps after the switch in each strip. Default=10.
    end_blocks_to_exclude : int
        Number of error strips to exclude from the end of the list. Default=0.
    error_threshold : float
        Error threshold to determine adaptation speed. Default=0.05.
    title : str
        Title of the plot. Default="Error Strips and Adaptation Analysis".
    cmap_name : str
        Colormap to use for coloring the strips. Default='viridis'.
    """
    if not error_strips:
        raise ValueError("Error strips list is empty. Nothing to plot.")

    # Exclude the specified number of strips from the end
    if end_blocks_to_exclude > 0:
        error_strips = error_strips[:-end_blocks_to_exclude]

    if not error_strips:
        raise ValueError("All error strips were excluded. Nothing to plot.")

    # Initialize colormap
    cmap = get_cmap(cmap_name)
    n_strips = len(error_strips)
    colors = cmap(np.linspace(0, 1, n_strips))  # Distribute colors evenly

    # Generate x-axis ticks relative to the block switch
    x_ticks = np.arange(-pre_window, post_window + 1)

    # Create the figure with double vertical size
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True)

    # First subplot: Error strips
    ax1 = axes[0]

    # Plot each strip
    for i, strip in enumerate(error_strips):
        if len(strip) != len(x_ticks):
            print(f"Error strip {i+1} length does not match pre_window + post_window + 1. Skipping.")
            continue
        ax1.plot(x_ticks, strip, color=colors[i], 
                 label=f"Strip {i+1}" if i in [0, n_strips // 3, 2 * n_strips // 3, n_strips - 1] else None)

    # Add mean line
    mean_error = np.mean(np.vstack(error_strips), axis=0)
    ax1.plot(x_ticks, mean_error, color='black', linewidth=2, linestyle='-', label='Mean Error')

    # Add a dashed vertical line at the block switch (time step 0)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.8, label="Block Switch")

    # Generate legend entries for 4 representative strips
    legend_handles = [
        Line2D([0], [0], color=colors[idx], lw=2, label=f"Strip {idx+1}")
        for idx in [0, n_strips // 3, 2 * n_strips // 3, n_strips - 1]
    ]
    legend_handles.append(Line2D([0], [0], color='black', lw=2, label='Mean Error'))
    ax1.legend(handles=legend_handles, title="Sample Strips", loc="upper right")

    # Finalize first subplot
    ax1.set_title("Error Strips")
    ax1.set_xlabel("Time Steps Relative to Block Switch")
    ax1.set_ylabel("Error")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Second subplot: Adaptation times
    ax2 = axes[1]
    ax2.plot(range(len(adaptation_times)), adaptation_times, marker='o', linestyle='-', color='blue', label='Adaptation Time')

    # Finalize second subplot
    ax2.set_title("Adaptation Time to Error Threshold")
    ax2.set_xlabel("Block Switch Index")
    ax2.set_xticks(range(len(adaptation_times)))
    ax2.set_ylabel("Timesteps to Reach Error Threshold")
    ax2.axhline(error_threshold, color='red', linestyle='--', alpha=0.7, label=f"Threshold = {error_threshold}")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim([0, post_window+2])

    # Set the overall title
    fig.suptitle(title)

    return fig, axes
#%
def plot_adaptation_times(adaptation_times_seeds, ax, use_first=True, label='Model', color = 'tab:blue'):
    """
    Plot adaptation times with means on the x-axis and either the first value or 
    the average of values after the first to the end as the y-axis, with SEM.

    Parameters:
    - adaptation_times_seeds: List of dictionaries containing adaptation times per seed.
    - ax: Matplotlib axis object to plot on.
    - use_first: Boolean, whether to use the first value (True) or the average of values 
      after the first to the end (False).

    Returns:
    - None
    """
    means = sorted(adaptation_times_seeds[0].keys())  # Get the sorted list of means
    mean_adaptation = []
    sem_adaptation = []

    for mean in means:
        if use_first:
            values = [seed_dict[mean][0] for seed_dict in adaptation_times_seeds]
        else:
            values = [
                np.mean(seed_dict[mean][1:]) for seed_dict in adaptation_times_seeds
            ]
        
        mean_adaptation.append(np.mean(values))
        sem_adaptation.append(np.std(values) / np.sqrt(len(values)))

    # Plot the results
    mean_adaptation = np.array(mean_adaptation)
    sem_adaptation = np.array(sem_adaptation)

    ax.plot(means, mean_adaptation, label=label, marker='o',color=color,)
    ax.fill_between(
        means,
        mean_adaptation - sem_adaptation,
        mean_adaptation + sem_adaptation,
        color=color,
        alpha=0.2,
        # label="SEM"
    )

    # ax.set_title("Adaptation Time vs Means")
    ax.set_xlabel("Latent states")
    ax.set_ylabel("Adaptation Time (timesteps)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

#%
def plot_post_window_errors(post_errors_seeds, ax, use_first=True, label='Model', color = 'tab:blue'):  
    """
    Plot post window errors with means on the x-axis and either the first value or 
    the average of values as the y-axis, with SEM.

    Parameters:
    - post_errors_seeds: List of dictionaries containing post window errors per seed.
    - ax: Matplotlib axis object to plot on.
    - use_first: Boolean, whether to use the first value (True) or the average of values 
      (False).

    Returns:
    - None
    """
    means = sorted(post_errors_seeds[0].keys())  # Get the sorted list of means
    mean_post_errors = []
    sem_post_errors = []

    for mean in means:
        if use_first:
            values = [seed_dict[mean][0] for seed_dict in post_errors_seeds]
        else:
            values = [np.mean(seed_dict[mean][1:]) for seed_dict in post_errors_seeds]
        
        mean_post_errors.append(np.mean(values))
        sem_post_errors.append(np.std(values) / np.sqrt(len(values)))

    # Plot the results
    mean_post_errors = np.array(mean_post_errors)
    sem_post_errors = np.array(sem_post_errors)

    ax.plot(means, mean_post_errors, label=label, linewidth=0.5, markersize= 2, marker='o', color = color)
    ax.fill_between(
        means,
        mean_post_errors - sem_post_errors,
        mean_post_errors + sem_post_errors,
        color=color,
        alpha=0.2,
        # label="SEM"
    )

    # ax.set_title("Post Window Error vs Means")
    ax.set_xlabel("Latent states")
    ax.set_ylabel("MSE (time averaged)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    return mean_post_errors

