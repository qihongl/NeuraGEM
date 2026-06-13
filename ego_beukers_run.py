"""
ego_beukers_run.py

Runs the EGO model (Giallanza et al., 2024, Study 2) on the Beukers et al. (2024)
sequence learning task, using the same data pipeline (seq_learnDataset) and Logger
format as the NeuraGEM experiments.

Curricula: blocked, interleaved, interleaved_blocked
Models: EGO (this file); compare with NeuraGEM results from seq_learn_run.py

Usage:
    python ego_beukers_run.py

    # Or with SLURM array for parallel seeds:
    SLURM_ARRAY_TASK_ID=0 python ego_beukers_run.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from datasets import seq_learnDataset
from ego_beukers_model import EGOBeukersModel
from functions_and_utils import Logger

# Import NeuraGEM config for parameter matching
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ego_beukers_model import EMModule, RecurrentContextModule
import torch.nn as nn


class EGOBeukersModelFlex(EGOBeukersModel):
    """
    EGOBeukersModel with generalised weight initialisation.
    Handles arbitrary hidden_d (the base class requires hidden_d == state_d).
    """
    def _init_weights(self, persistence):
        """Mirrors prep_recurrent_network from EGO experiment3.py, generalised."""
        with torch.no_grad():
            cm = self.context_module
            sd = self.state_d
            hd = cm.n_hidden_units
            cd = self.context_d

            # state_to_hidden: partial identity
            cm.state_to_hidden.weight.zero_()
            n = min(sd, hd)
            cm.state_to_hidden.weight[:n, :n] = torch.eye(n, dtype=torch.float)
            cm.state_to_hidden.bias.zero_()

            cm.hidden_to_hidden.weight.zero_()
            cm.hidden_to_hidden.bias.zero_()

            cm.state_to_hidden_wt.weight.zero_()
            cm.state_to_hidden_wt.bias.copy_(
                torch.ones(hd, dtype=torch.float) * persistence
            )

            cm.hidden_to_hidden_wt.weight.zero_()
            cm.hidden_to_hidden_wt.bias.zero_()

            # hidden_to_context: identity if dims match, random otherwise
            if cd == hd:
                cm.hidden_to_context.weight.copy_(torch.eye(hd, dtype=torch.float))
            cm.hidden_to_context.bias.zero_()

        # Freeze all recurrent weights except hidden_to_context
        for p in cm.parameters():
            p.requires_grad = False
        cm.hidden_to_context.weight.requires_grad = True


def _init_ego_weights(model, persistence):
    """
    Re-initialize EGO model weights to handle arbitrary hidden_d.
    (The built-in _init_weights requires hidden_d == state_d.)
    This mirrors prep_recurrent_network from EmergentIntelligentControl/experiment3.py
    but generalises to hidden_d ≠ state_d.
    """
    with torch.no_grad():
        cm = model.context_module
        sd = model.state_d
        hd = cm.n_hidden_units
        cd = model.context_d

        # state_to_hidden: partial identity (sd columns, hd rows)
        cm.state_to_hidden.weight.zero_()
        n = min(sd, hd)
        cm.state_to_hidden.weight[:n, :n] = torch.eye(n, dtype=torch.float)
        cm.state_to_hidden.bias.zero_()

        cm.hidden_to_hidden.weight.zero_()
        cm.hidden_to_hidden.bias.zero_()

        cm.state_to_hidden_wt.weight.zero_()
        cm.state_to_hidden_wt.bias.copy_(
            torch.ones(len(cm.state_to_hidden_wt.bias), dtype=torch.float) * persistence
        )

        cm.hidden_to_hidden_wt.weight.zero_()
        cm.hidden_to_hidden_wt.bias.zero_()

        # hidden_to_context: identity if dimensions match, random otherwise
        if cd == hd:
            cm.hidden_to_context.weight.copy_(torch.eye(hd, dtype=torch.float))
        cm.hidden_to_context.bias.zero_()

    # Freeze all recurrent weights except hidden_to_context
    for p in cm.parameters():
        p.requires_grad = False
    cm.hidden_to_context.weight.requires_grad = True


# ---------------------------------------------------------------------------
# Configuration — matched to NeuraGEM's seq_learn_config.py
# ---------------------------------------------------------------------------

# Phase lengths (timesteps) — equated to NeuraGEM's seq_learn_config.py
BLOCKED_PHASE_LENGTH = 2500      # for pure blocked (matches NeuraGEM ng_long_phase)
INTERLEAVED_PHASE_LENGTH = 2500  # for pure interleaved (matches NeuraGEM ng_long_phase)
IB_INTERLEAVED_LENGTH = 700      # for interleaved→blocked (matches NeuraGEM)
IB_BLOCKED_LENGTH = 2500         # for interleaved→blocked (matches NeuraGEM ng_long_phase)

TASK_LENGTH = 6           # timesteps per story
BLOCK_SIZE = 120           # 20 stories for blocked training
INTERLEAVED_BLOCK_SIZE = 6 # 1 story for interleaved
TEST_LENGTH = 240          # 40 stories testing

# Model params — matched to EGO paper
STATE_D = 10
HIDDEN_D = 10
CONTEXT_D = 4
TEMPERATURE = 0.2          # EM softmax temperature (lower = sharper retrieval)
PERSISTENCE = 1.0
EPISODIC_LR = 1.0          # SGD learning rate for context module + state_weight

BATCH_SIZE = 1
N_SEEDS = 20

# Output
EXPORT_BASE = './exports/ego_beukers/'


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def get_curriculum_config(curriculum, phase):
    """
    Returns (no_of_blocks, block_size, add_interleaved, add_blocked, phase_order)
    matching NeuraGEM's seq_learn_run.py logic.
    """
    if curriculum == 'blocked':
        if phase == 'train':
            n_blocks = BLOCKED_PHASE_LENGTH // BLOCK_SIZE
            bs = BLOCK_SIZE
            interleaved = False
            blocked = True
            order = ['blocked']
        else:  # test
            n_blocks = TEST_LENGTH // BLOCK_SIZE
            bs = BLOCK_SIZE
            interleaved = False
            blocked = True
            order = ['test']
    
    elif curriculum == 'interleaved':
        if phase == 'train':
            n_blocks = INTERLEAVED_PHASE_LENGTH // INTERLEAVED_BLOCK_SIZE
            bs = INTERLEAVED_BLOCK_SIZE
            interleaved = True
            blocked = False
            order = ['interleaved']
        else:
            n_blocks = TEST_LENGTH // INTERLEAVED_BLOCK_SIZE
            bs = INTERLEAVED_BLOCK_SIZE
            interleaved = True
            blocked = False
            order = ['test']
    
    elif curriculum == 'interleaved_blocked':
        if phase == 'interleaved':
            n_blocks = IB_INTERLEAVED_LENGTH // INTERLEAVED_BLOCK_SIZE
            bs = INTERLEAVED_BLOCK_SIZE
            interleaved = True
            blocked = False
            order = ['interleaved']
        elif phase == 'blocked':
            n_blocks = IB_BLOCKED_LENGTH // BLOCK_SIZE
            bs = BLOCK_SIZE
            interleaved = False
            blocked = True
            order = ['blocked']
        else:  # test
            n_blocks = TEST_LENGTH // BLOCK_SIZE
            bs = BLOCK_SIZE
            interleaved = False
            blocked = True
            order = ['test']
    
    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    
    return n_blocks, bs, interleaved, blocked, order


# ---------------------------------------------------------------------------
# Simple config object matching the interface seq_learnDataset expects
# ---------------------------------------------------------------------------

class EGOConfig:
    """Minimal config providing the attributes seq_learnDataset needs."""
    def __init__(self, no_of_blocks, block_size):
        self.no_of_blocks = no_of_blocks
        self.block_size = block_size
        self.task_length = TASK_LENGTH
        self.space_size = STATE_D
        self.observation_scale = 1.0
        self.seq_len = 18        # used only for Dataset.__len__
        self.stride = 1          # used only for Dataset.__len__
        self.env_seed = 0
        self.shuffle_or_interleave = 'interleave'
        self.random_transition_shuffle_or_interleave = 'shuffle'  # equate to NeuraGEM (random)
        self.seq_learn_use_deterministic_transition_2 = False
        # Not used by EGO but required by Logger compat
        self.device = torch.device('cpu')


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_ego_experiment(curriculum, seed, episodc_lr=EPISODIC_LR,
                        hidden_d=None, context_d=None, temperature=None):
    """
    Run a single EGO experiment for one (curriculum, seed) pair.
    
    Returns:
        logger: NeuraGEM-compatible Logger with inputs, predicted_outputs, 
                llcids, hlcids, latent_values (EGO context), and phases.
        model: The trained EGOBeukersModel (for external accuracy computation).
    """
    hd = hidden_d if hidden_d is not None else HIDDEN_D
    cd = context_d if context_d is not None else CONTEXT_D
    temp = temperature if temperature is not None else TEMPERATURE

    # CRITICAL: seed PyTorch RNG for reproducible weight init
    torch.manual_seed(seed)
    
    # Use subclass with generalised init when hidden_d ≠ state_d
    if hd != STATE_D:
        model = EGOBeukersModelFlex(
            state_d=STATE_D, hidden_d=hd, context_d=cd,
            temperature=temp, persistence=PERSISTENCE,
        )
    else:
        model = EGOBeukersModel(
            state_d=STATE_D, hidden_d=hd, context_d=cd,
            temperature=temp, persistence=PERSISTENCE,
        )
    
    # Only trainable params: context_module.hidden_to_context.weight + em_module.state_weight
    all_params = list(model.trainable_parameters()) + [model.em_module.state_weight]
    optimizer = torch.optim.SGD(all_params, lr=episodc_lr)
    logger = Logger()
    
    # Determine phase order
    if curriculum == 'interleaved_blocked':
        phases = ['interleaved', 'blocked']
    elif curriculum in ['blocked', 'interleaved']:
        phases = ['train']  # single training phase
    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    
    # ---- Training phases ----
    for phase in phases:
        n_blocks, block_size, _, _, _ = get_curriculum_config(curriculum, phase)
        config = EGOConfig(no_of_blocks=n_blocks, block_size=block_size)
        config.env_seed = seed
        np.random.seed(seed)
        
        dataset = seq_learnDataset(config)
        states = dataset.states  # (total_timesteps, 10)
        low_lats = dataset.low_level_latents   # list of ints
        high_lats = dataset.high_level_latents  # list of ints
        
        # Phase label for logger
        if curriculum == 'interleaved_blocked':
            phase_label = 'Interleaved\ntraining' if phase == 'interleaved' else 'Blocked\ntraining'
        else:
            phase_label = 'Blocked\ntraining' if curriculum == 'blocked' else 'Interleaved\ntraining'
        
        logger.log_phase(phase_label)
        
        # Iterate timestep by timestep
        for t in range(len(states) - 1):
            state_t = torch.from_numpy(states[t]).float().unsqueeze(0)      # (1, 10)
            next_t = torch.from_numpy(states[t + 1]).float().unsqueeze(0)   # (1, 10)
            
            # Forward step
            pred, loss, context = model.forward_step(state_t, next_t, train=True)
            
            # Backward pass through context module (skip when EM is empty)
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Write to episodic memory (detached)
            model.write_to_memory(state_t, next_t, context)
            
            # Log — match NeuraGEM Logger format: (batch, stride, dim)
            # NeuraGEM logs the TARGET observation as "input" and the prediction as "predicted_output"
            # For EGO: state_t is the input to context module, next_t is the target observation
            logger.log_input(next_t.detach().numpy().reshape(1, 1, -1))
            logger.log_predicted_output(pred.detach().numpy().reshape(1, 1, -1))
            logger.log_latent_value(context.numpy().reshape(1, 1, -1))  # context → latent_values
            logger.llcids.append(np.array([[[low_lats[t]]]], dtype=np.float32))
            logger.hlcids.append(np.array([[[high_lats[t]]]], dtype=np.float32))
    
    # Record training end
    logger.others['timestep_learning_ended'] = len(logger.inputs)
    
    # ---- Testing phase ----
    logger.log_phase('Testing\n(W frozen)')
    
    n_blocks, block_size, _, _, _ = get_curriculum_config(curriculum, 'test')
    config = EGOConfig(no_of_blocks=n_blocks, block_size=block_size)
    config.env_seed = seed + 1  # different seed for test data
    np.random.seed(seed + 1)
    
    dataset = seq_learnDataset(config)
    states = dataset.states
    low_lats = dataset.low_level_latents
    high_lats = dataset.high_level_latents
    
    # Direct Tcid computation from raw states (no logger indirection)
    tcid_correct, tcid_total = 0, 0
    for t in range(len(states) - 1):
        state_t = torch.from_numpy(states[t]).float().unsqueeze(0)
        next_t = torch.from_numpy(states[t + 1]).float().unsqueeze(0)
        
        # Test: predict only, no weight updates
        with torch.no_grad():
            pred = model.forward_step(state_t, next_t, train=False)
        
        # Tcid: when current state = 3, predict next state (5/6)
        if np.argmax(states[t]) == 3:
            ctx_val = high_lats[t]
            correct_state = 5 if ctx_val == 0 else 6
            if np.argmax(pred.numpy()) == correct_state:
                tcid_correct += 1
            tcid_total += 1
        
        # Extract context for logging (stored internally in context_module)
        context = model.context_module.hidden_to_context(
            model.context_module.hidden_state.to(state_t.device).unsqueeze(0)
        ).detach().numpy()
        
        # Log
        logger.log_input(next_t.numpy().reshape(1, 1, -1))
        logger.log_predicted_output(pred.numpy().reshape(1, 1, -1))
        logger.log_latent_value(context.reshape(1, 1, -1))
        logger.llcids.append(np.array([[[low_lats[t]]]], dtype=np.float32))
        logger.hlcids.append(np.array([[[high_lats[t]]]], dtype=np.float32))
    
    tcid_acc = tcid_correct / tcid_total if tcid_total else 0.0
    logger.others['ego_tcid_accuracy'] = tcid_acc
    logger.others['ego_tcid_counts'] = (tcid_correct, tcid_total)
    
    return logger, model


def save_results(filename, data, export_path):
    os.makedirs(export_path, exist_ok=True)
    filepath = os.path.join(export_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return filepath


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    
    # Targeted sweep: temperature × learning_rate × hidden_d
    temperatures = [0.05, 0.1, 0.2, 0.5]
    learning_rates = [0.5, 1.0, 2.0, 5.0]
    hidden_dims = [16, 64]
    context_d = 4  # canonical from EGO paper
    n_seeds_sweep = 5
    
    # Build experiment list
    experiments = []
    for curriculum in curricula:
        for seed in range(n_seeds_sweep):
            for hd in hidden_dims:
                for lr in learning_rates:
                    for temp in temperatures:
                        experiments.append((curriculum, seed, hd, lr, temp))
    
    total = len(experiments)
    print(f'Total experiments: {total} '
          f'({len(curricula)} curricula × {n_seeds_sweep} seeds × '
          f'{len(hidden_dims)} HD × {len(learning_rates)} LR × {len(temperatures)} T)')
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if task_id >= 0:
        if task_id >= total:
            raise ValueError(f'Task {task_id} >= {total} experiments')
        to_run = [experiments[task_id]]
    else:
        to_run = experiments
    
    for curriculum, seed, hd, lr, temp in to_run:
        label = f'EGO|{curriculum[:3]}|s={seed}|h={hd}|lr={lr:.1f}|T={temp:.2f}'
        print(f'Running {label}')
        
        logger, model = run_ego_experiment(curriculum, seed, hidden_d=hd, context_d=context_d,
                                     episodc_lr=lr, temperature=temp)
        
        export_path = os.path.join(EXPORT_BASE, f'h{hd}_c{context_d}_T{temp}_LR{lr}', curriculum)
        filename = f'ego_{curriculum}_seed-{seed}.pkl'
        filepath = save_results(filename, logger, export_path)
        
        inputs = np.vstack(logger.inputs).reshape(-1, STATE_D)
        outputs = np.vstack(logger.predicted_outputs).reshape(-1, STATE_D)
        pred_states = np.argmax(outputs, axis=-1)
        true_states = np.argmax(inputs, axis=-1)
        
        train_end = logger.others.get('timestep_learning_ended', len(true_states))
        train_acc = np.mean(pred_states[:train_end] == true_states[:train_end])
        test_acc = np.mean(pred_states[train_end:] == true_states[train_end:])
        
        print(f'  → Train: {train_acc:.3f} | Test: {test_acc:.3f} | → {filepath}')
    
    if task_id < 0:
        print(f'\nDone. {len(to_run)} experiments completed.')
