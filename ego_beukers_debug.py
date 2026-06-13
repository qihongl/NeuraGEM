"""
Debugging script: test hypotheses for why EGO fails on Beukers task.
H1: Reset MGRU hidden state at story boundaries
H2: Use only blueprint stories (no variants)
H3: H1 + H2 combined  
H4: Reset EM between training and testing
Baseline: current implementation (no changes)
"""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import seq_learnDataset
from ego_beukers_model import EGOBeukersModel
from ego_beukers_run import EGOBeukersModelFlex, EGOConfig, get_curriculum_config

STATE_D = 10
TASK_LENGTH = 6

def compute_tcid(logger):
    inputs = np.concatenate(logger.inputs, axis=0).squeeze(1)
    preds = np.concatenate(logger.predicted_outputs, axis=0).squeeze(1)
    hlcids = np.concatenate(logger.hlcids, axis=0).squeeze(1).squeeze(-1)
    test_start = logger.phases[-1][1]
    state3 = np.where(inputs[:, 3] > 0.5)[0]
    test_tcid = [i for i in state3 if i >= test_start and i < len(hlcids) - 1]
    if not test_tcid:
        return 0.0, 0
    correct = sum(1 for i in test_tcid
                  if np.argmax(preds[i + 1]) == (5 if hlcids[i + 1] == 0 else 6))
    return correct / len(test_tcid), len(test_tcid)


def run_one(curriculum, seed, name, reset_hidden_between_stories=False,
            blueprint_only=False, reset_em_before_test=False,
            unfreeze_gate=False, unfreeze_all_mgru=False):
    """Run one EGO experiment with specific hypothesis modifications."""
    from ego_beukers_run import (
        BLOCKED_PHASE_LENGTH, INTERLEAVED_PHASE_LENGTH, BLOCK_SIZE,
        INTERLEAVED_BLOCK_SIZE, IB_INTERLEAVED_LENGTH, IB_BLOCKED_LENGTH,
        TEST_LENGTH, STATE_D, EGOBeukersModel, PERSISTENCE
    )
    from functions_and_utils import Logger

    hd, cd, temp, lr = 16, 4, 0.2, 1.0

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

    # H5/H6: unfreeze MGRU weights
    if unfreeze_all_mgru:
        for p in model.context_module.parameters():
            p.requires_grad = True
    elif unfreeze_gate:
        model.context_module.state_to_hidden_wt.bias.requires_grad = True
        model.context_module.state_to_hidden_wt.weight.requires_grad = True

    all_params = list(model.trainable_parameters()) + [model.em_module.state_weight]
    optimizer = torch.optim.SGD(all_params, lr=lr)
    logger = Logger()

    # Determine phases
    if curriculum == 'interleaved_blocked':
        train_phases = ['interleaved', 'blocked']
    else:
        train_phases = ['train']

    for phase in train_phases:
        n_blocks, block_size, _, _, _ = get_curriculum_config(curriculum, phase)
        cfg = EGOConfig(no_of_blocks=n_blocks, block_size=block_size)
        cfg.env_seed = seed
        if blueprint_only:
            cfg.seq_learn_use_deterministic_transition_2 = True
        np.random.seed(seed)

        ds = seq_learnDataset(cfg)
        states = ds.states
        low_lats = ds.low_level_latents
        high_lats = ds.high_level_latents

        if curriculum == 'interleaved_blocked':
            phase_label = 'Interleaved\ntraining' if phase == 'interleaved' else 'Blocked\ntraining'
        else:
            phase_label = 'Blocked\ntraining' if curriculum == 'blocked' else 'Interleaved\ntraining'
        logger.log_phase(phase_label)

        # Detect story boundaries: state 9 → state 0 transition
        # State 9 is the last state of each story; next state 0 starts new story
        for t in range(len(states) - 1):
            # Story boundary detection: state 9 → state 0
            is_boundary = bool(
                np.argmax(states[t]) == 9 and np.argmax(states[t + 1]) == 0
            )
            if reset_hidden_between_stories and is_boundary:
                model.context_module.reset_hidden()

            state_t = torch.from_numpy(states[t]).float().unsqueeze(0)
            next_t = torch.from_numpy(states[t + 1]).float().unsqueeze(0)

            pred, loss, context = model.forward_step(state_t, next_t, train=True)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.write_to_memory(state_t, next_t, context)

            logger.log_input(next_t.detach().numpy().reshape(1, 1, -1))
            logger.log_predicted_output(pred.detach().numpy().reshape(1, 1, -1))
            logger.log_latent_value(context.numpy().reshape(1, 1, -1))
            logger.llcids.append(np.array([[[low_lats[t]]]], dtype=np.float32))
            logger.hlcids.append(np.array([[[high_lats[t]]]], dtype=np.float32))

    logger.others['timestep_learning_ended'] = len(logger.inputs)

    # ---- Testing phase ----
    logger.log_phase('Testing\n(W frozen)')

    if reset_em_before_test:
        model.em_module.reset()

    n_blocks, block_size, _, _, _ = get_curriculum_config(curriculum, 'test')
    cfg = EGOConfig(no_of_blocks=n_blocks, block_size=block_size)
    cfg.env_seed = seed + 1
    if blueprint_only:
        cfg.seq_learn_use_deterministic_transition_2 = True
    np.random.seed(seed + 1)

    ds = seq_learnDataset(cfg)
    states = ds.states
    low_lats = ds.low_level_latents
    high_lats = ds.high_level_latents

    for t in range(len(states) - 1):
        if reset_hidden_between_stories:
            is_boundary = bool(
                np.argmax(states[t]) == 9 and np.argmax(states[t + 1]) == 0
            )
            if is_boundary:
                model.context_module.reset_hidden()

        state_t = torch.from_numpy(states[t]).float().unsqueeze(0)
        next_t = torch.from_numpy(states[t + 1]).float().unsqueeze(0)

        with torch.no_grad():
            pred = model.forward_step(state_t, next_t, train=False)

        context = model.context_module.hidden_to_context(
            model.context_module.hidden_state.to(state_t.device).unsqueeze(0)
        ).detach().numpy()

        logger.log_input(next_t.numpy().reshape(1, 1, -1))
        logger.log_predicted_output(pred.numpy().reshape(1, 1, -1))
        logger.log_latent_value(context.reshape(1, 1, -1))
        logger.llcids.append(np.array([[[low_lats[t]]]], dtype=np.float32))
        logger.hlcids.append(np.array([[[high_lats[t]]]], dtype=np.float32))

    return logger


if __name__ == '__main__':
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    seeds = list(range(5))
    configs = [
        ('baseline', False, False, False, False, False),
        ('H5_unfreezeGate', False, False, False, True, False),
        ('H6_unfreezeAll', False, False, False, False, True),
    ]

    results = {}
    export_dir = './exports/ego_beukers/debug'
    os.makedirs(export_dir, exist_ok=True)

    for name, reset_hidden, blueprint_only, reset_em, unfreeze_gate, unfreeze_all in configs:
        print(f'\n{"=" * 60}')
        print(f'CONFIG: {name}')
        desc_parts = []
        if reset_hidden: desc_parts.append('reset_hidden')
        if blueprint_only: desc_parts.append('blueprint_only')
        if reset_em: desc_parts.append('reset_em')
        if unfreeze_gate: desc_parts.append('unfreeze_gate')
        if unfreeze_all: desc_parts.append('unfreeze_all_mgru')
        print(f'  {", ".join(desc_parts) if desc_parts else "none"}')
        print(f'{"=" * 60}')
        res = {c: [] for c in curricula}
        for curriculum in curricula:
            for seed in seeds:
                try:
                    logger = run_one(curriculum, seed, name,
                                     reset_hidden_between_stories=reset_hidden,
                                     blueprint_only=blueprint_only,
                                     reset_em_before_test=reset_em,
                                     unfreeze_gate=unfreeze_gate,
                                     unfreeze_all_mgru=unfreeze_all)
                    acc, n = compute_tcid(logger)
                    res[curriculum].append(acc)
                except Exception as e:
                    print(f'  ERROR: {curriculum} s={seed}: {e}')
                    res[curriculum].append(None)

            accs = [a for a in res[curriculum] if a is not None]
            if accs:
                mean = np.mean(accs)
                sem = np.std(accs) / np.sqrt(len(accs))
                print(f'  {curriculum:20s}: Tcid = {mean:.3f} ± {sem:.3f} (n={len(accs)})')
            else:
                print(f'  {curriculum:20s}: NO DATA')

        results[name] = res

        # Save
        with open(f'{export_dir}/{name}.pkl', 'wb') as f:
            pickle.dump(res, f)

    # Summary table
    print(f'\n{"=" * 80}')
    print(f'{"Config":25s} | {"blocked":>12s} | {"interleaved":>12s} | {"IB":>12s} | {"Δ b-i":>8s}')
    print(f'{"-" * 80}')
    for name, _, _, _, _, _ in configs:
        row = f'{name:25s}'
        row_data = []
        for c in curricula:
            accs = [a for a in results[name][c] if a is not None]
            if accs:
                m = np.mean(accs)
                row_data.append(m)
                row += f' | {m:6.3f}±{np.std(accs)/np.sqrt(len(accs)):.2f}'
            else:
                row_data.append(None)
                row += f' | {"N/A":>12s}'
        if row_data[0] is not None and row_data[1] is not None:
            row += f' | {row_data[0] - row_data[1]:+8.3f}'
        print(row)

    print(f'\nResults saved to {export_dir}/')
