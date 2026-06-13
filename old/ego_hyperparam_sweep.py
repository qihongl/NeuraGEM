"""
Hyperparameter sweep for EGO model on Beukers task.
Searches for parameters that produce blocked >> interleaved advantage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from datasets import seq_learnDataset
from ego_beukers_model import EGOBeukersModel
import json
from itertools import product

TASK_LENGTH = 6
STATE_D = 10
TEST_STEPS = 240

class MiniConfig:
    def __init__(self, no_of_blocks, block_size, seed=0):
        self.no_of_blocks = no_of_blocks
        self.block_size = block_size
        self.task_length = TASK_LENGTH
        self.space_size = STATE_D
        self.observation_scale = 1.0
        self.seq_len = 18
        self.stride = 1
        self.env_seed = seed
        self.shuffle_or_interleave = 'interleave'
        self.random_transition_shuffle_or_interleave = 'interleave'
        self.seq_learn_use_deterministic_transition_2 = False


def run_one(config_dict, seed=0):
    """Run one config on both blocked and interleaved, return test accuracies."""
    torch.manual_seed(seed)
    
    # Parse config
    temp = config_dict['temperature']
    persistence = config_dict['persistence']
    lr = config_dict['episodic_lr']
    hidden_d = config_dict.get('hidden_d', 10)
    context_d = config_dict.get('context_d', 4)
    wr = config_dict.get('weighted_retrieval', False)
    fix_state_weight = config_dict.get('fix_state_weight', None)
    n_opt_steps = config_dict.get('n_opt_steps', 1)
    
    results = {}
    
    for curriculum in ['blocked', 'interleaved']:
        model = EGOBeukersModel(
            state_d=STATE_D, hidden_d=hidden_d, context_d=context_d,
            temperature=temp, persistence=persistence,
            weighted_retrieval=wr,
        )
        
        # Fix state_weight if specified
        if fix_state_weight is not None:
            with torch.no_grad():
                model.em_module.state_weight.copy_(torch.tensor([fix_state_weight]))
            model.em_module.state_weight.requires_grad = False
        
        # Build optimizer
        all_params = list(model.trainable_parameters())
        if wr and fix_state_weight is None:
            all_params.append(model.em_module.state_weight)
        opt = torch.optim.SGD(all_params, lr=lr)
        
        # Training data
        if curriculum == 'blocked':
            n_blocks = 15; bs = 120  # 1800 timesteps
        else:
            n_blocks = 300; bs = 6  # 1800 timesteps (interleaved)
        
        cfg = MiniConfig(n_blocks, bs, seed)
        ds = seq_learnDataset(cfg)
        states = ds.states
        
        # Train
        for t in range(len(states) - 1):
            s = torch.from_numpy(states[t]).float().unsqueeze(0)
            ns = torch.from_numpy(states[t + 1]).float().unsqueeze(0)
            
            # Multiple optimization steps per timestep
            for _ in range(n_opt_steps):
                pred, loss, ctx = model.forward_step(s, ns, train=True)
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            # Write to EM once per timestep (use context from last opt step)
            model.write_to_memory(s, ns, ctx)
        
        # Test
        test_cfg = MiniConfig(2, 120, seed + 1)
        test_ds = seq_learnDataset(test_cfg)
        test_states = test_ds.states
        
        correct = 0
        total = 0
        for t in range(min(TEST_STEPS, len(test_states) - 1)):
            s = torch.from_numpy(test_states[t]).float().unsqueeze(0)
            ns = torch.from_numpy(test_states[t + 1]).float().unsqueeze(0)
            with torch.no_grad():
                pred = model.forward_step(s, ns, train=False)
            correct += int(pred[0].argmax().item() == ns[0].argmax().item())
            total += 1
        
        results[curriculum] = correct / total if total > 0 else 0.0
    
    return results['blocked'], results['interleaved']


# -------------------------------------------------------------------
# Sweep definitions
# -------------------------------------------------------------------

# Grid 1: Fix state_weight to force context-dominant retrieval
grid1 = [
    {'temperature': t, 'persistence': p, 'episodic_lr': 1.0, 
     'weighted_retrieval': False, 'fix_state_weight': sw,
     'n_opt_steps': 1, 'hidden_d': 10, 'context_d': 4}
    for t in [0.05, 0.1, 0.2]
    for p in [0.5, 1.0, 2.0]
    for sw in [-1.0, -2.0, -5.0]  # sigmoid(-1)=0.27 state, sigmoid(-5)=0.007 state
]

# Grid 2: Multiple optimization steps (analogous to NeuraGEM's latent steps)
grid2 = [
    {'temperature': t, 'persistence': p, 'episodic_lr': 2.0,
     'weighted_retrieval': False, 'fix_state_weight': -2.0,
     'n_opt_steps': n, 'hidden_d': 10, 'context_d': 4}
    for t in [0.1, 0.2]
    for p in [0.5, 1.0, 2.0]
    for n in [3, 5]
]

# Grid 3: Larger context + hidden dims with context-dominant retrieval
grid3 = [
    {'temperature': t, 'persistence': p, 'episodic_lr': 1.0,
     'weighted_retrieval': False, 'fix_state_weight': -2.0,
     'n_opt_steps': n, 'hidden_d': h, 'context_d': c}
    for t in [0.1, 0.2]
    for p in [1.0, 2.0]
    for n in [1, 3]
    for h, c in [(20, 8), (32, 8)]
]

# Grid 4: Standard approach but with weighted_retrieval (learnable state_weight)
# but with strong initialization toward context
grid4 = [
    {'temperature': t, 'persistence': p, 'episodic_lr': 1.0,
     'weighted_retrieval': True, 'fix_state_weight': sw,
     'n_opt_steps': 1, 'hidden_d': 10, 'context_d': 4}
    for t in [0.1, 0.2]
    for p in [0.5, 1.0, 2.0]
    for sw in [-2.0]  # start state_weight at -2 (sigmoid(-2)=0.12 → ~12% state, 88% context)
]

all_grids = grid1 + grid2 + grid3 + grid4
print(f"Total configs: {len(all_grids)}")

# Pick via SLURM or run first N
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
start_idx = task_id * 50
end_idx = min(start_idx + 50, len(all_grids))

results_list = []
for idx in range(start_idx, end_idx):
    cfg = all_grids[idx]
    try:
        b_acc, i_acc = run_one(cfg, seed=0)
        advantage = b_acc - i_acc
        cfg['blocked_acc'] = b_acc
        cfg['interleaved_acc'] = i_acc
        cfg['advantage'] = advantage
        results_list.append(cfg)
        
        marker = "***" if advantage > 0.15 else ("**" if advantage > 0.05 else ("*" if advantage > 0 else " "))
        print(f"[{idx:3d}] {marker} blocked={b_acc:.3f} interleaved={i_acc:.3f} adv={advantage:+.3f} | "
              f"T={cfg['temperature']:.2f} P={cfg['persistence']:.1f} LR={cfg['episodic_lr']:.1f} "
              f"sw={cfg['fix_state_weight']} n_opt={cfg['n_opt_steps']} "
              f"h={cfg['hidden_d']} c={cfg['context_d']} wr={cfg['weighted_retrieval']}")
    except Exception as e:
        print(f"[{idx:3d}] ERROR: {e}")

# Sort and report best
if results_list:
    results_list.sort(key=lambda x: x['advantage'], reverse=True)
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGS:")
    print(f"{'='*80}")
    for i, r in enumerate(results_list[:10]):
        print(f"  {i+1}. adv={r['advantage']:+.3f} blocked={r['blocked_acc']:.3f} interleaved={r['interleaved_acc']:.3f}"
              f" | T={r['temperature']} P={r['persistence']} LR={r['episodic_lr']}"
              f" sw={r['fix_state_weight']} n_opt={r['n_opt_steps']}"
              f" h={r['hidden_d']} c={r['context_d']} wr={r['weighted_retrieval']}")
    
    os.makedirs('figs', exist_ok=True)
    with open('figs/ego_sweep_results.json', 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"\nSaved {len(results_list)} results to figs/ego_sweep_results.json")
