import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import axes, cm
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from functions_and_utils import explore_data_container, plot_logger_panels, get_corrects_and_trial_starts
from functions_and_utils_2 import calculate_error
from functions_adaptation_dynamics_analysis import *

''' Functions for the statistical analysis of the CSW task results '''


def get_latent_states_and_features(logger, ):

    # Plot a simple average of the latent for each of the transitions
    inputs = np.vstack(logger.inputs)
    # inputs has shape (no of batches, batch_size, seq_len, features)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    states = np.argmax(inputs, axis=-1)
    states = states.squeeze()
    # states.shape is (2145,)
    latent = np.vstack(logger.latent_values)
    latent = latent.reshape(-1, latent.shape[-1])
    latent = latent[:, -1] # take only one value since they are symmetrical

    # Create a predictor for each transition and see which one mostly correltes with latent using linear regression
    from sklearn.linear_model import LinearRegression

    T0_predictor = np.zeros(len(latent))
    T0_predictor[states == 0] = 1

    T1_2_predictor = np.zeros(len(latent))
    T1_2_predictor[states == 1] = 1
    T1_2_predictor[states == 2] = -1

    T3_4_predictor = np.zeros(len(latent))
    T3_4_predictor[states == 3] = 1
    T3_4_predictor[states == 4] = -1

    T5_6_predictor = np.zeros(len(latent))
    T5_6_predictor[states == 5] = 1
    T5_6_predictor[states == 6] = -1

    T7_8_predictor = np.zeros(len(latent))
    T7_8_predictor[states == 7] = 1
    T7_8_predictor[states == 8] = -1

    T9_predictor = np.zeros(len(latent))
    T9_predictor[states == 9] = 1
    
    features = np.hstack([T0_predictor.reshape(-1, 1), T1_2_predictor.reshape(-1, 1), T3_4_predictor.reshape(-1, 1), T5_6_predictor.reshape(-1, 1), T7_8_predictor.reshape(-1, 1), T9_predictor.reshape(-1, 1)])
    target = latent
    return features, target


# Function to compute regression betas
def compute_regression_betas(ts_begin, ts_end, features, targets):
    _features = features[ts_begin:ts_end]
    _targets = targets[ts_begin:ts_end]
    regression_model = LinearRegression()
    regression_model.fit(_features, _targets)
    betas = regression_model.coef_
    return np.abs(betas)

# Function to plot regression betas
def plot_betas(betas_dict, predictor_names):
    fig, axes = plt.subplots(1, len(betas_dict), figsize=(4, 1), sharey=True)
    for i, (phase, betas) in enumerate(betas_dict.items()):
        axes[i].bar(predictor_names, betas)
        axes[i].set_xlabel('Predictor')
        axes[i].set_ylabel('Beta')
        axes[i].set_title(f'{phase} phase')
    plt.tight_layout()
    plt.show()

# Function to gather and plot the evolution of unpredictable transitions betas and end accuracy
def plot_unpredictable_transition_evolution(interleaved_unpredictable, blocked_unpredictable, end_accuracies):
    fig, ax = plt.subplots(figsize=(3, 2))
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(min(end_accuracies), max(end_accuracies))

    for i in range(len(interleaved_unpredictable)):
        color = cmap(norm(end_accuracies[i]))
        ax.plot(['Interleaved', 'Blocked'], [interleaved_unpredictable[i], blocked_unpredictable[i]], '-o', color=color, label=f'Model {i+1}')
    ax.set_xlabel('Training Phase')
    ax.set_ylabel('Unpredictable Transition Beta (Difference)')
    ax.set_title('Evolution of Unpredictable Transition Betas Across Phases')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='End Accuracy')
    plt.show()


def compute_phase_betas(loggers: list, phase: str) -> tuple [np.ndarray, np.ndarray]:
    """
    Compute CID and RND regression betas for a given phase across loggers.
    phase: 'interleaved' or 'blocked'
    Returns: (cid_betas, rnd_betas) as numpy arrays.
    """
    cid_betas, rnd_betas = [], []
    for logger in loggers:
        features, target = get_latent_states_and_features(logger)
        
        # find the index of the requested phase
        phases = logger.phases  # list of (name, t)
        idx = next((i for i, (name, _) in enumerate(phases)
                    if name.lower().startswith(phase)), None)
        if idx is None or idx == len(phases) - 1:
            raise ValueError(f"Cannot locate phase '{phase}' or no subsequent phase")
        
        ts_start = phases[idx][1]
        ts_end   = phases[idx + 1][1]
        
        betas = compute_regression_betas(ts_start, ts_end, features, target)
        cid_betas.append(betas[1])
        rnd_betas.append(betas[2] )#+ betas[4])
    return np.array(cid_betas), np.array(rnd_betas)

def plot_cid_rnd_single_phase(cid_betas, rnd_betas, phase_name, cs=None, ax : mpl.axes._axes.Axes =None, orient : str ='h'):
    """
    Plot CID vs RND betas for a single phase.
    """
    import plot_style
    cs = cs or plot_style.Color_scheme()
    df = pd.DataFrame({'T_cid': cid_betas, 'T_rnd': rnd_betas})
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(cs.panel_small_size[0], cs.panel_small_size[1]))
    sns.boxplot(data=df, ax=ax, palette=['grey', 'grey'],
                width=0.6, showfliers=False, orient=orient, linewidth=0.7)
    sns.stripplot(data=df, ax=ax, color='black',
                  size=cs.marker_size, jitter=True, alpha=0.5, orient=orient)
    if orient == 'h':
        ax.set_ylabel('Predictor')
        ax.set_xlabel('Z encoding (Reg β)')
        ax.set_yticklabels(['$T_{cid}$', '$T_{rnd}$'])
    elif orient == 'v':    
        ax.set_xlabel('Predictor')
        ax.set_ylabel('Z encoding (Reg β)')
        ax.set_xticklabels(['$T_{cid}$', '$T_{rnd}$'])


def betas_scatter(t_rnd_betas, t_cid_betas, ax : mpl.axes._axes.Axes =None):
    """
    Scatter plot of T_rnd vs T_cid betas with regression line.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    reg_color = 'tab:red'
    tx1, tx2 = 0.3, 1.01
    # Scatter plot between blocked unpredictable betas and end accuracy
    ax.scatter(t_rnd_betas, t_cid_betas, s=20, c='grey', marker='x')
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_rnd_betas, t_cid_betas)
    ax.plot(t_rnd_betas, intercept + slope * np.array(t_rnd_betas), reg_color)
    ax.set_xlabel('T$_{rnd}$ Beta')
    ax.set_ylabel('T$_{cid}$ Beta')


from scipy import stats
def plot_statistical_relationships(interleaved_unpredictable, blocked_unpredictable, blocked_task_identifying, end_accuracies):
    # fig, axes = plt.subplots(1, 4, figsize=(7, 2))
    fig, axes = plt.subplots(2, 2, figsize=(3., 3.))
    p_color = 'tab:red'
    reg_color = 'tab:red'
    tx1, tx2 = 0.3, 1.01
    # Scatter plot between blocked unpredictable betas and end accuracy
    ax = axes.flatten()[0]
    ax.scatter(blocked_unpredictable, end_accuracies, s=20, c='grey', marker='x')
    slope, intercept, r_value, p_value, std_err = stats.linregress(blocked_unpredictable, end_accuracies)
    ax.plot(blocked_unpredictable, intercept + slope * np.array(blocked_unpredictable), reg_color)
    ax.set_xlabel('Blocked T$_{rnd}$ Beta')
    ax.set_ylabel('End Accuracy')
    # ax.text(tx1,tx2, f'p = {p_value:.3f}', transform=ax.transAxes, color=p_color)

    # Scatter plot between blocked task identifying beta and end accuracy
    ax = axes.flatten()[1]
    ax.scatter(blocked_task_identifying, end_accuracies, s=20, c='grey', marker='x')
    slope, intercept, r_value, p_value, std_err = stats.linregress(blocked_task_identifying, end_accuracies)
    ax.plot(blocked_task_identifying, intercept + slope * np.array(blocked_task_identifying), reg_color)
    ax.set_xlabel('Blocked T$_{cid}$ Beta')
    ax.set_ylabel('End Accuracy')
    # ax.text(tx1,tx2, f'p = {p_value:.3f}', transform=ax.transAxes, color=p_color)

    # Scatter plot between interleaved and blocked unpredictable betas
    ax = axes.flatten()[2]
    ax.scatter(interleaved_unpredictable, blocked_unpredictable, s=20, c='grey', marker='x')
    slope, intercept, r_value, p_value, std_err = stats.linregress(interleaved_unpredictable, blocked_unpredictable)
    ax.plot(interleaved_unpredictable, intercept + slope * np.array(interleaved_unpredictable), reg_color)
    ax.set_xlabel('Interleaved T$_{rnd}$ Beta')
    ax.set_ylabel('Blocked T$_{rnd}$ Beta')
    # ax.text(tx1,tx2, f'p = {p_value:.3f}', transform=ax.transAxes, color=p_color)

    # Scatter plot between interleaved unpredictable beta and blocked task identifying beta
    ax = axes.flatten()[3]
    ax.scatter(interleaved_unpredictable, blocked_task_identifying,  s=20, c='grey', marker='x')
    slope, intercept, r_value, p_value, std_err = stats.linregress(interleaved_unpredictable, blocked_task_identifying)
    ax.plot(interleaved_unpredictable, intercept + slope * np.array(interleaved_unpredictable), reg_color)
    ax.set_xlabel('Interleaved T$_{rnd}$ Beta')
    ax.set_ylabel('Blocked T$_{cid}$ Beta')
    # ax.text(tx1,tx2, f'p = {p_value:.3f}', transform=ax.transAxes, color=p_color)

    plt.tight_layout()
    return (fig)

def load_and_split_loggers(
    run_name: str,
    model_name: str,
    params: dict,
    seeds=range(20),
    export_root: str = './exports/csw/experiments'
):
    """
    Loads loggers for all seeds, computes testing scores,
    returns combination_key, mean score, full dict of loggers,
    list of above‐avg seeds, list of below‐avg seeds.
    """
    # rebuild key exactly as when saved
    key = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    loggers, scores = {}, []
    for seed in seeds:
        path = os.path.join(export_root, run_name, key)
        fname = f"results_{model_name}_{key}_seed-{seed}.pkl"
        lg = load_results(fname, path)
        if not lg:
            continue
        sc = compute_testing_score(lg)
        loggers[seed] = {'logger': lg, 'score': sc}
        scores.append(sc)
    if not scores:
        return key, None, {}, [], []
    avg = np.mean(scores)
    above = [s for s,v in loggers.items() if v['score'] > avg]
    below = [s for s,v in loggers.items() if v['score'] <= avg]
    return key, avg, loggers, above, below

# helper function to compute the testing score for one logger
def compute_testing_score(logger, alpha=0.5, transitions_to_use=['T1/2']):
    corrects, states, transitions, A_starts, B_starts, both_starts = \
        get_corrects_and_trial_starts(logger)
    both_starts = np.array(both_starts)
    focused = np.full_like(corrects, 0.5, dtype=float)
    for tr in transitions_to_use:
        idx = both_starts + transitions[tr]
        focused[idx] = corrects[idx]
    # exponential moving average
    run_avg = []
    prev = 0.0
    for v in focused:
        if v == 0.5:
            run_avg.append(prev)
        else:
            prev = (1 - alpha) * prev + alpha * v
            run_avg.append(prev)
    # find testing phase start and compute mean score
    testing_start = next(t for name, t in logger.phases if name.startswith('Testing'))
    return np.mean(run_avg[testing_start:])

def plot_violin_for_param(
    results_df, 
    curriculum, 
    param_name, 
    param_value, 
    x='latent_updates', 
    y='score', 
    ax=None, 
    color_palette='Set2', 
    title=None
):
    """
    Plots a violin plot (with stripplot overlay) for a given curriculum and parameter value.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results.
        curriculum (str): Curriculum to filter.
        param_name (str): Name of parameter to filter (e.g., 'phase_length').
        param_value (any): Value of the parameter to filter.
        x (str): Column for x-axis (default: 'latent_updates').
        y (str): Column for y-axis (default: 'score').
        ax (matplotlib.axes.Axes): Axis to plot to (optional).
        color_palette (str): Seaborn color palette.
        title (str): Plot title (optional).
    """
    plot_df = results_df[
        (results_df['curriculum'] == curriculum) &
        (results_df[param_name] == param_value)
    ]
    if ax is None:
        fig, ax = plt.subplots(figsize=(cs.panel_small_size))
    sns.violinplot(
        data=plot_df,
        x=x,
        y=y,
        palette=color_palette,
        ax=ax,
        scale='width',
    )
    sns.stripplot(
        data=plot_df,
        x=x,
        y=y,
        color='black',
        alpha=0.5,
        jitter=True,
        size=3,
        ax=ax
    )
    ax.axhline(y=0.5, color='red', linestyle='--', label='Chance Level (0.5)')
    ax.set_xlabel('Latent Updates')
    ax.set_ylabel('Testing accuracy')
    if title is None:
        title = f'({param_name}={param_value}, curriculum={curriculum})'
    ax.set_title(title, fontsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['False', 'True'])
    ax.set_ylim(0.4, 1.3)
    return ax

def draw_curriculum_schematic(ax, curriculum, Tcid_mode=None, Trnd_mode=None):
    """
    Draws a schematic of the curriculum phases on the given axis,
    with phase names labeled above.
    """
    ax.clear()
    ax.axis('off')
    fontsize = 6
    y_base = 0.13  # Base y position for the rectangles
    from plot_style import Color_scheme
    cs = Color_scheme()

    color1 = cs.contextA
    color2 = cs.contextB
    if curriculum == 'interleaved':
        n_boxes = 20
        box_width = 0.15
        box_height = 0.33
        spacing = 0.03
        if Tcid_mode == 'shuffle':
            rng = np.random.default_rng(2)
        for i in range(n_boxes):
            x = i * (box_width + spacing)
            if Tcid_mode == 'shuffle':
                color = color1 if rng.integers(0, 2) == 0 else color2
            else:
                color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), box_width, box_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        ax.set_xlim(-0.1, n_boxes * (box_width + spacing))
        ax.set_ylim(0, 1)
        # Phase label
        label = "Interleaved"
        if Tcid_mode is not None:
            # label = "$T_{cid}$ Shuffled\n$T_{rnd}$ Shuffled" if Tcid_mode == 'shuffle' else "$T_{cid}$ Interleaved"
            label = f"$T_{{cid}}$ {'Shuffled' if Tcid_mode == 'shuffle' else 'Interleaved'}\n$T_{{rnd}}$ {'Shuffled' if Trnd_mode == 'shuffle' else 'Interleaved'}"
        ax.text(n_boxes * (box_width + spacing) / 2, y_base + box_height + 0.08, label,
                ha='center', va='bottom', fontsize=fontsize)
    elif curriculum == 'blocked':
        n_blocks = 4
        block_width = 0.7
        block_height = 0.33
        spacing = 0.08
        for i in range(n_blocks):
            x = i * (block_width + spacing)
            color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), block_width, block_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        ax.set_xlim(-0.1, n_blocks * (block_width + spacing))
        ax.set_ylim(0, 1)
        # Phase labels
        phase = "Blocked"
        ax.text(
            n_blocks * (block_width + spacing) / 2,
            y_base + block_height + 0.08,
            phase,
            ha='center',  va='bottom', fontsize=fontsize,
    
        )
    elif curriculum == 'interleaved_blocked':
        n_interleaved = 10
        n_blocked = 2
        box_width = 0.15
        box_height = 0.33
        spacing = 0.03
        block_width = (n_interleaved * (box_width + spacing)) / n_blocked
        # Interleaved part
        for i in range(n_interleaved):
            x = i * (box_width + spacing)
            color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), box_width, box_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        # Blocked part
        for i in range(n_blocked):
            x = n_interleaved * (box_width + spacing) + i * (block_width + spacing)
            color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), block_width, box_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        total_width = n_interleaved * (box_width + spacing) + n_blocked * (block_width + spacing)
        ax.set_xlim(-0.1, total_width)
        ax.set_ylim(0, 1)
        # Phase labels
        interleaved_center = (n_interleaved * (box_width + spacing)) / 2
        blocked_center = n_interleaved * (box_width + spacing) + block_width
        ax.text(interleaved_center, y_base + box_height + 0.08, "Interleaved", 
            ha='center', va='bottom', fontsize=fontsize, rotation=30)
        ax.text(blocked_center, y_base + box_height + 0.08, "Blocked", 
            ha='center', va='bottom', fontsize=fontsize, rotation=30)
    elif curriculum == 'blocked_interleaved':
        n_blocked = 2
        n_interleaved = 10
        box_width = 0.15
        box_height = 0.33
        spacing = 0.03
        block_width = (n_interleaved * (box_width + spacing)) / n_blocked
        # Blocked part
        for i in range(n_blocked):
            x = i * (block_width + spacing)
            color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), block_width, box_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        # Interleaved part
        for i in range(n_interleaved):
            x = n_blocked * (block_width + spacing) + i * (box_width + spacing)
            color = color1 if i % 2 == 0 else color2
            rect = plt.Rectangle((x, y_base), box_width, box_height, color=color, ec=None, lw=0, alpha=0.7)
            ax.add_patch(rect)
        total_width = n_blocked * (block_width + spacing) + n_interleaved * (box_width + spacing)
        ax.set_xlim(-0.1, total_width)
        ax.set_ylim(0, 1)
        # Phase labels
        blocked_center = (n_blocked * (block_width + spacing)) / 2
        interleaved_center = n_blocked * (block_width + spacing) + (n_interleaved * (box_width + spacing)) / 2
        ax.text(blocked_center, y_base + box_height + 0.08, "Blocked", ha='center', va='bottom', fontsize=fontsize, rotation=30)
        ax.text(interleaved_center, y_base + box_height + 0.08, "Interleaved", ha='center', va='bottom', fontsize=fontsize, rotation=30)
    else:
        ax.text(0.5, 0.5, curriculum, ha='center', va='center', fontsize=fontsize)
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
