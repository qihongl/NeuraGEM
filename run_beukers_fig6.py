"""
Replicate Figure 6 from the NeuraGEM paper:
Run NeuraGEM on the Beukers et al. (2024) sequence learning task.

Curricula:
  - blocked: context constant for many stories
  - interleaved: context alternates every story
  - interleaved_blocked (mixed): interleaved first, then blocked

Testing: random curriculum (matching human experiment in Beukers et al.)
"""
import os, sys, pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import SeqLearnConfig, seq_learnConfig
from models import RNN_with_latent
from datasets import create_datasets_and_loaders
from train_and_infer_functions import predictive_learning
from functions_and_utils import Logger, plot_logger_panels
from seq_learn_config import apply_neuragem_overrides
import plot_style
plot_style.set_plot_style()


def run_one_curriculum(model_name, curriculum, seed, config_overrides=None):
    """
    Run a single NeuraGEM experiment for one curriculum.
    Mirrors run_single_experiment from seq_learn_run.py.
    """
    # 1) Config
    config = SeqLearnConfig(experiment_to_run='few_long_blocks')

    # CRITICAL: Match paper's seq_len = 6 (the default is 18)
    config.seq_len = 6

    # Neuragem-specific overrides (LU_lr, l2_loss, etc.)
    param_combo = {'seq_len': 6}
    if model_name == 'neuragem':
        apply_neuragem_overrides(config, param_combo)

    torch.manual_seed(seed)
    np.random.seed(seed)
    config.env_seed = seed

    # Global overrides
    config_overrides = config_overrides or {}
    for k, v in config_overrides.items():
        setattr(config, k, v)

    # Curriculum settings
    if curriculum == 'interleaved':
        config.add_interleaved_phase = True
        config.add_blocked_phase = False
        config.add_passive_learning_phase = False
    elif curriculum == 'blocked':
        config.add_interleaved_phase = False
        config.add_blocked_phase = True
        config.add_passive_learning_phase = False
    elif curriculum in ['interleaved_blocked', 'blocked_interleaved']:
        config.add_interleaved_phase = True
        config.add_blocked_phase = True
        config.add_passive_learning_phase = False

    # Phase lengths from seq_learn_config
    ng_long_phase = 2500
    ng_short_phase = 1250
    if curriculum == 'interleaved':
        config.interleaved_phase_length = ng_long_phase
    elif curriculum == 'blocked':
        config.blocked_phase_length = ng_long_phase
    elif curriculum in ['interleaved_blocked', 'blocked_interleaved']:
        config.interleaved_phase_length = 700
        config.blocked_phase_length = ng_long_phase

    config.latent_updates_during_shuffle = True

    # Create model
    model = RNN_with_latent(config).to(config.device)
    stored_block_size = config.block_size
    logger = Logger()

    # --- Interleaved phase ---
    if config.add_interleaved_phase and curriculum != 'blocked_interleaved':
        print(f'  Running interleaved phase ({config.interleaved_phase_length} stories)...')
        logger.log_phase('Interleaved\ntraining')
        config.block_size = config.task_length  # one story at a time
        config.no_of_blocks = config.interleaved_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        config._allow_latent_updates = config.latent_updates_during_shuffle
        predictive_learning(logger, config, dataloader, model)

    # --- Blocked phase ---
    if curriculum in ['blocked_interleaved', 'interleaved_blocked']:
        config._allow_latent_updates = True
    else:
        config._allow_latent_updates = config.latent_updates_during_shuffle

    if config.add_blocked_phase:
        print(f'  Running blocked phase ({config.blocked_phase_length} stories)...')
        config.block_size = stored_block_size
        logger.log_phase('Blocked\ntraining')
        config.no_of_blocks = config.blocked_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        predictive_learning(logger, config, dataloader, model)

    # Second interleaved phase (for blocked_interleaved)
    if curriculum == 'blocked_interleaved':
        print('  Running second interleaved phase...')
        logger.log_phase('Interleaved\ntraining\n(2nd)')
        config.block_size = config.task_length
        config.no_of_blocks = config.interleaved_phase_length // config.block_size
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        config._allow_latent_updates = config.latent_updates_during_shuffle
        predictive_learning(logger, config, dataloader, model)

    # --- Testing phase (random curriculum) ---
    print('  Running testing phase (random curriculum)...')
    model.config = config
    model.LU_optimizer = model.get_LU_optimizer()

    logger.log_phase('Testing\n(W frozen)')
    config.env_seed = seed + 1
    config.no_of_steps_in_weight_space = 0

    # Testing adjustments (matching paper Methods 3.5)
    config.shuffle_or_interleave = 'random'
    config.block_size = config.task_length

    # NeuraGEM-specific testing overrides
    config.LU_lr = 0.4
    config.no_of_steps_in_latent_space = 5
    config.l2_loss = 0.0004

    testing_phase_length = 40 * config.task_length  # 240 time steps
    config.no_of_blocks = int(testing_phase_length / config.block_size)
    _, _, dataloader, _ = create_datasets_and_loaders(config)
    predictive_learning(logger, config, dataloader, model)

    config.block_size = stored_block_size
    logger.config = config

    return logger


def compute_accuracy(logger):
    """Compute prediction accuracy on Tcid transitions (state 3->5/6)."""
    # The Tcid transition is from state 3 to either 5 (context A) or 6 (context B)
    # In the one-hot encoding, state 3 is at index 3
    # We need to find timesteps where the input is state 3 and check if the model
    # predicts the correct next state.

    inputs = np.concatenate(logger.inputs, axis=0)   # (T, batch, dims)
    preds = np.concatenate(logger.predicted_outputs, axis=0)  # (T, batch, dims)

    # Squeeze batch dim
    inputs = inputs.squeeze(1)    # (T, 10)
    preds = preds.squeeze(1)      # (T, 10)

    # Find timesteps where input is state 3 (index 3)
    state3_mask = inputs[:, 3] > 0.5  # one-hot encoding

    # The correct next state: if context A → state 5 (index 5), if context B → state 6 (index 6)
    # We have the high-level latent in logger.hlcids
    hlcids = np.concatenate(logger.hlcids, axis=0).squeeze(1)  # (T, 1)
    hlcids = hlcids.squeeze(-1)  # (T,)

    # For state3_mask, the next timestep is the target
    tcid_indices = np.where(state3_mask)[0]
    tcid_indices = tcid_indices[tcid_indices < len(hlcids) - 1]

    if len(tcid_indices) == 0:
        return 0.0, 0

    # Get predictions at Tcid+1
    correct = 0
    total = 0
    for idx in tcid_indices:
        if idx + 1 >= len(preds):
            continue
        # With predict_first_frame=True:
        # preds[idx] predicts inputs[idx] (same timestep)
        # preds[idx+1] predicts inputs[idx+1] (next timestep after state 3)
        # The context of the predicted state is hlcids[idx+1]
        true_context = hlcids[idx + 1]
        if true_context == 0:
            correct_state = 5
        else:
            correct_state = 6
        pred_state = np.argmax(preds[idx + 1])
        if pred_state == correct_state:
            correct += 1
        total += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


def compute_z_encoding(logger):
    """
    Compute linear regression betas quantifying how much Z encodes
    Tcid (context-identifying) vs Trnd (random) transitions.
    Uses numpy least-squares (no external deps).
    """
    inputs = np.concatenate(logger.inputs, axis=0).squeeze(1)   # (T, 10)
    z_vals = np.concatenate(logger.latent_values, axis=0).squeeze(1)  # (T, Z_dim)

    # Only test phase
    phases = logger.phases  # list of (name, start_idx) tuples
    test_phases = [p for p in phases if 'Testing' in p[0]]
    if test_phases:
        test_start = test_phases[0][1]
        z_vals = z_vals[test_start:]
        inputs = inputs[test_start:]

    # Tcid: state 3 → context-dependent transition (states 5/6)
    tcid_mask = np.zeros(len(inputs))
    for i in range(1, len(inputs)):
        if inputs[i-1, 3] > 0.5:  # previous state is state 3
            tcid_mask[i] = 1.0

    # Trnd: random transition (state 4 → ...)
    trnd_mask = np.zeros(len(inputs))
    for i in range(1, len(inputs)):
        if inputs[i-1, 4] > 0.5:  # previous state is state 4
            trnd_mask[i] = 1.0

    # OLS via numpy: Z = [1, Tcid, Trnd] @ betas
    X = np.column_stack([np.ones(len(tcid_mask)), tcid_mask, trnd_mask])
    beta_tcid = 0
    beta_trnd = 0
    try:
        for zdim in range(z_vals.shape[1]):
            y = z_vals[:, zdim]
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            beta_tcid += abs(betas[1])
            beta_trnd += abs(betas[2])
    except Exception:
        pass

    return beta_tcid, beta_trnd


def plot_fig6_panels(loggers, export_dir):
    """Create Figure-6 style comparison panels."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    panel_labels = ['Accuracy', 'Z timeseries', 'Z encoding', 'Training loss']

    for row_idx, curriculum in enumerate(curricula):
        if curriculum not in loggers:
            continue
        logger = loggers[curriculum]

        # Compute metrics
        acc, n = compute_accuracy(logger)
        beta_tcid, beta_trnd = compute_z_encoding(logger)

        # Panel 1: Accuracy bar
        ax = axes[row_idx, 0]
        ax.bar(['NeuraGEM'], [acc], color='#2196F3')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Prediction accuracy\n(Tcid transition)')
        ax.set_title(f'{curriculum}\nAccuracy = {acc:.3f}')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='chance')
        ax.legend(fontsize=8)

        # Panel 2: Z timeseries (testing phase)
        ax = axes[row_idx, 1]
        z_vals = np.concatenate(logger.latent_values, axis=0).squeeze(1)
        phases = logger.phases
        test_phases = [p for p in phases if 'Testing' in p[0]]
        if test_phases:
            test_start = test_phases[0][1]
            z_test = z_vals[test_start:]
            ax.plot(z_test[:, 0], label='Z dim 0', alpha=0.8, linewidth=0.8)
            if z_test.shape[1] > 1:
                ax.plot(z_test[:, 1], label='Z dim 1', alpha=0.8, linewidth=0.8)
        ax.set_ylabel('Z activity')
        ax.set_xlabel('Test time steps')
        ax.legend(fontsize=7)
        ax.set_title(f'{curriculum}: Z dynamics')

        # Panel 3: Z encoding bar
        ax = axes[row_idx, 2]
        ax.bar(['Tcid', 'Trnd'], [beta_tcid, beta_trnd],
               color=['#4CAF50', '#FF9800'])
        ax.set_ylabel('Z encoding (|β|)')
        ax.set_title(f'{curriculum}: Z→transition')

        # Panel 4: Training loss
        ax = axes[row_idx, 3]
        losses = np.concatenate(logger.training_losses, axis=0).squeeze()
        if len(losses) > 200:
            # Smooth
            window = len(losses) // 100
            losses_sm = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(losses_sm, linewidth=0.8)
        else:
            ax.plot(losses, linewidth=0.8)
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Batch')
        ax.set_title(f'{curriculum}: Training loss')

    plt.suptitle('NeuraGEM on Beukers et al. Task — Figure 6 Replication (1 seed)', fontsize=14)
    plt.tight_layout()
    os.makedirs(export_dir, exist_ok=True)
    fig_path = os.path.join(export_dir, 'fig6_replication.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {fig_path}')
    return fig_path


def main():
    seed = 42
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']

    config_overrides = {
        'start_always_on_the_same_block': False,
        'add_passive_learning_phase': False,
    }

    export_dir = './exports/seq_learn/fig6_replication'
    os.makedirs(export_dir, exist_ok=True)

    loggers = {}
    for curriculum in curricula:
        print(f'\n{"="*60}')
        print(f'Running: neuragem | curriculum={curriculum} | seed={seed}')
        print(f'{"="*60}')

        logger = run_one_curriculum('neuragem', curriculum, seed, config_overrides)

        # Save logger
        pkl_path = os.path.join(export_dir, f'logger_{curriculum}_seed{seed}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(logger, f)
        print(f'Saved logger to {pkl_path}')

        loggers[curriculum] = logger

    # Print summary
    print(f'\n{"="*60}')
    print('RESULTS SUMMARY')
    print(f'{"="*60}')
    for curriculum in curricula:
        if curriculum in loggers:
            acc, n = compute_accuracy(loggers[curriculum])
            beta_tcid, beta_trnd = compute_z_encoding(loggers[curriculum])
            print(f'  {curriculum:25s} | Accuracy: {acc:.3f} (n={n}) | Tcid β: {beta_tcid:.4f} | Trnd β: {beta_trnd:.4f}')

    # Generate figure
    plot_fig6_panels(loggers, export_dir)

    print('\nDone!')


if __name__ == '__main__':
    main()
