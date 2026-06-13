"""
Quick test: three promising EGO modifications for Beukers task.
H7: Multiple optimization steps per timestep
H8: Adam optimizer 
H9: Lower persistence
"""
import sys, numpy as np, torch
sys.path.insert(0, '.')

from datasets import seq_learnDataset
from ego_beukers_model import EGOBeukersModel
from ego_beukers_run import (EGOBeukersModelFlex, EGOConfig, get_curriculum_config,
                              BLOCKED_PHASE_LENGTH, INTERLEAVED_PHASE_LENGTH,
                              BLOCK_SIZE, INTERLEAVED_BLOCK_SIZE,
                              IB_INTERLEAVED_LENGTH, IB_BLOCKED_LENGTH, TEST_LENGTH)


def run_test(curriculum, seed, n_opt_steps=1, use_adam=False, persistence=1.0):
    hd, cd, temp, lr = 16, 4, 0.2, 1.0

    torch.manual_seed(seed)
    model = EGOBeukersModelFlex(state_d=10, hidden_d=hd, context_d=cd,
                                 temperature=temp, persistence=persistence)
    params = list(model.trainable_parameters()) + [model.em_module.state_weight]
    opt = torch.optim.Adam(params, lr=lr) if use_adam else torch.optim.SGD(params, lr=lr)

    if curriculum == 'interleaved_blocked':
        train_phases = ['interleaved', 'blocked']
    else:
        train_phases = ['train']

    for phase in train_phases:
        n_blocks, block_size, _, _, _ = get_curriculum_config(curriculum, phase)
        cfg = EGOConfig(no_of_blocks=n_blocks, block_size=block_size)
        cfg.env_seed = seed
        np.random.seed(seed)
        ds = seq_learnDataset(cfg)
        states = ds.states

        for t in range(len(states) - 1):
            s = torch.from_numpy(states[t]).float().unsqueeze(0)
            ns = torch.from_numpy(states[t + 1]).float().unsqueeze(0)

            for _ in range(n_opt_steps):
                pred, loss, ctx = model.forward_step(s, ns, train=True)
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Write to EM once per timestep (use context from last opt step)
            model.write_to_memory(s, ns, ctx)

    # Test
    n_blocks_t, bs_t, _, _, _ = get_curriculum_config(curriculum, 'test')
    cfg_t = EGOConfig(no_of_blocks=n_blocks_t, block_size=bs_t)
    cfg_t.env_seed = seed + 1
    np.random.seed(seed + 1)
    tds = seq_learnDataset(cfg_t)
    ts = tds.states

    c_tcid, n_tcid = 0, 0
    for t in range(len(ts) - 1):
        s = torch.from_numpy(ts[t]).float().unsqueeze(0)
        ns = torch.from_numpy(ts[t + 1]).float().unsqueeze(0)
        with torch.no_grad():
            pred = model.forward_step(s, ns, train=False)
        if np.argmax(s.numpy()) == 3:
            ctx_val = tds.high_level_latents[t]
            correct_state = 5 if ctx_val == 0 else 6
            if np.argmax(pred.numpy()) == correct_state:
                c_tcid += 1
            n_tcid += 1

    return c_tcid / n_tcid if n_tcid else 0.0


if __name__ == '__main__':
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    seeds = list(range(5))

    configs = [
        ('baseline',          1, False, 1.0),
        ('H7_nOpt=3',         3, False, 1.0),
        ('H7_nOpt=5',         5, False, 1.0),
        ('H7_nOpt=10',       10, False, 1.0),
        ('H8_Adam',           1, True,  1.0),
        ('H9_persist=0.5',    1, False, 0.5),
        ('H9_persist=0.0',    1, False, 0.0),
        ('H9_persist=-0.6',   1, False, -0.6),
    ]

    print(f'{"Config":20s} | {"blocked":>10s} | {"interleaved":>12s} | {"IB":>10s} | {"B-I":>6s}')
    print('-' * 70)

    for name, n_opt, adam, persist in configs:
        results = {c: [] for c in curricula}
        for curriculum in curricula:
            for seed in seeds:
                acc = run_test(curriculum, seed, n_opt_steps=n_opt,
                               use_adam=adam, persistence=persist)
                results[curriculum].append(acc)

        b_m = np.mean(results['blocked'])
        i_m = np.mean(results['interleaved'])
        ib_m = np.mean(results['interleaved_blocked'])
        b_i_delta = b_m - i_m

        marker = ''
        if b_m > 0.6 or i_m > 0.6 or ib_m > 0.6:
            marker = ' ←'
        elif b_m > 0.55:
            marker = ' ↑'

        b_sem = np.std(results['blocked']) / np.sqrt(5)
        i_sem = np.std(results['interleaved']) / np.sqrt(5)
        ib_sem = np.std(results['interleaved_blocked']) / np.sqrt(5)

        print(f'{name:20s} | {b_m:6.3f}+-{b_sem:.2f} | {i_m:6.3f}+-{i_sem:.2f} | {ib_m:6.3f}+-{ib_sem:.2f} | {b_i_delta:+6.3f}{marker}')
