"""
ego_diagnostics.py

Runs EGO model on all three curricula with detailed diagnostics,
produces figures to understand model behavior.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from datasets import seq_learnDataset
from ego_beukers_model import EGOBeukersModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_D = 10
HIDDEN_D = 10
CONTEXT_D = 4
TEMPERATURE = 0.2
PERSISTENCE = 1.0
EPISODIC_LR = 1.0
TASK_LENGTH = 6
BLOCK_SIZE_BLOCKED = 120
BLOCK_SIZE_INTERLEAVED = 6
N_BLOCKS_BLOCKED = 15     # 1800 timesteps
N_BLOCKS_INTERLEAVED = 300  # 1800 timesteps
IB_INTERLEAVED_BLOCKS = 84  # ~500 timesteps
IB_BLOCKED_BLOCKS = 9       # ~1080 timesteps
SEED = 0


class MiniConfig:
    """Minimal config for seq_learnDataset."""
    def __init__(self, no_of_blocks, block_size):
        self.no_of_blocks = no_of_blocks
        self.block_size = block_size
        self.task_length = TASK_LENGTH
        self.space_size = STATE_D
        self.observation_scale = 1.0
        self.seq_len = 18
        self.stride = 1
        self.env_seed = SEED
        self.shuffle_or_interleave = 'interleave'
        self.random_transition_shuffle_or_interleave = 'interleave'
        self.seq_learn_use_deterministic_transition_2 = False


# ---------------------------------------------------------------------------
# Run one experiment with diagnostics
# ---------------------------------------------------------------------------

def run_diagnostic(curriculum, temp=TEMPERATURE, weighted_retrieval=True):
    """
    Run EGO model and collect:
    - Per-timestep accuracy
    - Context vectors over time
    - State weight over time
    - Softmax weight distributions at switch points
    """
    model = EGOBeukersModel(
        state_d=STATE_D, hidden_d=HIDDEN_D, context_d=CONTEXT_D,
        temperature=temp, persistence=PERSISTENCE,
        weighted_retrieval=weighted_retrieval,
    )
    all_params = list(model.trainable_parameters())
    if weighted_retrieval:
        all_params.append(model.em_module.state_weight)
    opt = torch.optim.SGD(all_params, lr=EPISODIC_LR)
    
    # Build dataset phases
    if curriculum == 'blocked':
        phase_configs = [
            ('train', MiniConfig(N_BLOCKS_BLOCKED, BLOCK_SIZE_BLOCKED)),
            ('test', MiniConfig(2, BLOCK_SIZE_BLOCKED)),
        ]
    elif curriculum == 'interleaved':
        phase_configs = [
            ('train', MiniConfig(N_BLOCKS_INTERLEAVED, BLOCK_SIZE_INTERLEAVED)),
            ('test', MiniConfig(40, BLOCK_SIZE_INTERLEAVED)),
        ]
    elif curriculum == 'interleaved_blocked':
        phase_configs = [
            ('interleaved', MiniConfig(IB_INTERLEAVED_BLOCKS, BLOCK_SIZE_INTERLEAVED)),
            ('blocked', MiniConfig(IB_BLOCKED_BLOCKS, BLOCK_SIZE_BLOCKED)),
            ('test', MiniConfig(2, BLOCK_SIZE_BLOCKED)),
        ]
    
    # Diagnostic storage
    diag = {
        'correct': [],         # 1 if prediction correct, 0 otherwise
        'loss': [],            # per-timestep loss
        'context_norms': [],   # ||context|| per timestep
        'state_weight': [],    # sigmoid(state_weight) over time
        'high_level_latent': [],  # which graph (0=A, 1=B)
        'phase_boundaries': [],   # timestep indices where phases change
        'softmax_max': [],      # max softmax weight per query
        'softmax_entropy': [],  # entropy of softmax weights per query
        'context_change': [],   # ||context_t - context_{t-1}||
        'cumulative_timestep': 0,
    }
    
    cumulative_t = 0
    test_phase = False
    
    for phase_label, config in phase_configs:
        diag['phase_boundaries'].append(cumulative_t)
        
        ds = seq_learnDataset(config)
        states = ds.states
        hls = ds.high_level_latents
        
        for t in range(len(states) - 1):
            s = torch.from_numpy(states[t]).float().unsqueeze(0)
            ns = torch.from_numpy(states[t + 1]).float().unsqueeze(0)
            
            # Record context before step
            ctx_before = model.context_module.hidden_state.clone()
            
            # Get EM match weights before softmax (for diagnostics)
            if model.em_module.values is not None and len(model.em_module.values) > 1:
                with torch.no_grad():
                    # Get context for diagnostic
                    cm = model.context_module
                    h_prev_diag = cm.hidden_state.clone()
                    x_diag = s
                    h_up_diag = torch.tanh(
                        cm.state_to_hidden(x_diag) + cm.hidden_to_hidden(h_prev_diag))
                    h_wt_diag = torch.sigmoid(
                        cm.state_to_hidden_wt(x_diag) + cm.hidden_to_hidden_wt(h_prev_diag))
                    h_new_diag = h_wt_diag * h_prev_diag + (1 - h_wt_diag) * h_up_diag
                    ctx_diag = cm.hidden_to_context(h_new_diag)
                    
                    mw = model.em_module.get_match_weights(s, ctx_diag)
                    sw = torch.softmax(mw, dim=-1)
                    diag['softmax_max'].append(sw.max().item())
                    diag['softmax_entropy'].append(
                        -(sw * torch.log(sw + 1e-10)).sum().item())
            else:
                diag['softmax_max'].append(np.nan)
                diag['softmax_entropy'].append(np.nan)
            
            # Forward step
            if not test_phase:
                pred, loss, ctx = model.forward_step(s, ns, train=True)
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                diag['state_weight'].append(
                    torch.sigmoid(model.em_module.state_weight).item())
            else:
                with torch.no_grad():
                    pred = model.forward_step(s, ns, train=False)
            
            # Write to memory (always during training)
            if not test_phase:
                model.write_to_memory(s, ns, ctx)
            
            # Record metrics
            correct = int(pred[0].argmax().item() == ns[0].argmax().item())
            diag['correct'].append(correct)
            diag['loss'].append(loss.item() if not test_phase else 
                float(torch.nn.functional.mse_loss(pred, ns).item()))
            diag['high_level_latent'].append(hls[t])
            
            # Context metrics
            ctx_after = model.context_module.hidden_state.clone()
            diag['context_norms'].append(ctx_after.norm().item())
            diag['context_change'].append(
                (ctx_after - ctx_before).norm().item())
            
            cumulative_t += 1
            diag['cumulative_timestep'] = cumulative_t
        
        if 'test' in phase_label:
            test_phase = True
    
    return diag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(y, window=20):
    """Exponential moving average."""
    alpha = 2.0 / (window + 1)
    result = np.zeros_like(y)
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha * y[i] + (1 - alpha) * result[i - 1]
    return result


def make_figures():
    """Generate diagnostic figures for all curricula."""
    
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    all_diags = {}
    
    print("Running diagnostics...")
    for curriculum in curricula:
        print(f"  {curriculum}...", end=' ', flush=True)
        diag = run_diagnostic(curriculum)
        all_diags[curriculum] = diag
        final_acc = np.mean(diag['correct'][-200:])
        print(f"final accuracy: {final_acc:.3f}")
    
    # -----------------------------------------------------------------------
    # Figure 1: Training accuracy over time + context dynamics
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    
    for idx, (curriculum, diag) in enumerate(all_diags.items()):
        correct = np.array(diag['correct'], dtype=float)
        n = len(correct)
        t = np.arange(n)
        
        # ---- Panel A: Accuracy ----
        ax1 = plt.subplot(3, 3, idx * 3 + 1)
        ax1.plot(t, smooth(correct, 50), color='#2c3e50', linewidth=1.0, alpha=0.8)
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        
        # Mark phase boundaries
        colors_bg = ['#e8f4f8', '#fef9e7', '#eaf7ee']
        for pi, pb in enumerate(diag['phase_boundaries']):
            if pi + 1 < len(diag['phase_boundaries']):
                ax1.axvspan(pb, diag['phase_boundaries'][pi + 1],
                           color=colors_bg[pi % 3], alpha=0.4)
            ax1.axvline(x=pb, color='#e74c3c', linewidth=1.5, alpha=0.5)
        
        ax1.set_ylabel('Accuracy (EMA)', fontsize=9)
        ax1.set_title(f'{curriculum}', fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, n)
        ax1.grid(alpha=0.2)
        
        # Highlight graph identity
        hl = np.array(diag['high_level_latent'])
        for i in range(0, len(hl), 60):
            if i < n:
                color = '#3498db' if hl[i] == 0 else '#e74c3c'
                ax1.axvspan(i, min(i + 60, n), color=color, alpha=0.03)
        
        # ---- Panel B: Context norm ----
        ax2 = plt.subplot(3, 3, idx * 3 + 2)
        cn = np.array(diag['context_norms'])
        ax2.plot(t, cn, color='#8e44ad', linewidth=0.5, alpha=0.7)
        ax2.plot(t, smooth(cn, 50), color='#6c3483', linewidth=1.0)
        ax2.set_ylabel('||hidden state||', fontsize=9)
        ax2.set_title(f'{curriculum} — context norm', fontsize=9)
        ax2.grid(alpha=0.2)
        
        # ---- Panel C: Softmax entropy ----
        ax3 = plt.subplot(3, 3, idx * 3 + 3)
        se = np.array(diag['softmax_entropy'])
        valid = ~np.isnan(se)
        if valid.sum() > 0:
            ax3.plot(t[valid], se[valid], color='#16a085', linewidth=0.5, alpha=0.7)
            ax3.plot(t[valid], smooth(se[valid], 50), color='#0e6655', linewidth=1.0)
        max_ent = np.log(100)  # approximate max entropy
        ax3.axhline(y=max_ent, color='gray', linestyle=':', alpha=0.3, label=f'log(N)≈{max_ent:.1f}')
        ax3.set_ylabel('Softmax entropy', fontsize=9)
        ax3.set_title(f'{curriculum} — retrieval entropy', fontsize=9)
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.2)
    
    plt.tight_layout(pad=2.0)
    fig.savefig('figs/ego_training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("→ Saved figs/ego_training_dynamics.png")
    
    # -----------------------------------------------------------------------
    # Figure 2: Context representation analysis (blocked only, most revealing)
    # -----------------------------------------------------------------------
    # Re-run blocked with context logging
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    model = EGOBeukersModel(
        state_d=STATE_D, hidden_d=HIDDEN_D, context_d=CONTEXT_D,
        temperature=TEMPERATURE, persistence=PERSISTENCE,
        weighted_retrieval=True,
    )
    all_params = list(model.trainable_parameters()) + [model.em_module.state_weight]
    opt = torch.optim.SGD(all_params, lr=EPISODIC_LR)
    
    config = MiniConfig(15, 120)  # blocked training
    ds = seq_learnDataset(config)
    states = ds.states
    hls = ds.high_level_latents
    
    # Collect context vectors at specific times
    contexts_by_graph = {0: [], 1: []}  # context vectors per graph
    contexts_by_state = {i: [] for i in range(10)}  # context per state
    timesteps_by_graph = {0: [], 1: []}
    accuracy_by_transition = []  # accuracy per timestep
    preds_all = []
    
    for t in range(len(states) - 1):
        s = torch.from_numpy(states[t]).float().unsqueeze(0)
        ns = torch.from_numpy(states[t + 1]).float().unsqueeze(0)
        
        pred, loss, ctx = model.forward_step(s, ns, train=True)
        if loss.requires_grad:
            opt.zero_grad(); loss.backward(); opt.step()
        model.write_to_memory(s, ns, ctx)
        
        # Store context
        g = int(hls[t])
        contexts_by_graph[g].append(ctx.numpy().flatten())
        timesteps_by_graph[g].append(t)
        
        state_id = int(states[t].argmax())
        next_id = int(states[t + 1].argmax())
        contexts_by_state[state_id].append(ctx.numpy().flatten())
        
        correct = int(pred[0].argmax().item() == next_id)
        accuracy_by_transition.append((t, state_id, next_id, correct))
        preds_all.append((t, state_id, next_id, pred[0].argmax().item()))
    
    # Panel 1: Context PCA
    ax = axes[0, 0]
    all_ctx = []
    all_colors = []
    for g in [0, 1]:
        ctx = np.array(contexts_by_graph[g])
        if len(ctx) > 0:
            all_ctx.append(ctx)
            all_colors.extend([g] * len(ctx))
    if sum(len(c) for c in all_ctx) > 2:
        all_ctx = np.vstack(all_ctx)
        all_colors = np.array(all_colors)
        # Simple projection: first 2 dims of context
        ax.scatter(all_ctx[:, 0], all_ctx[:, 1],
                   c=all_colors, cmap='coolwarm', alpha=0.3, s=2)
        ax.set_xlabel('Context dim 0')
        ax.set_ylabel('Context dim 1')
        ax.set_title('Context vectors (color = graph)')
    
    # Panel 2: Context similarity across blocks
    ax = axes[0, 1]
    # Take context samples every 10 timesteps
    sample_ctx = []
    sample_graph = []
    for g in [0, 1]:
        ctx_g = np.array(contexts_by_graph[g])
        for i in range(0, len(ctx_g), 10):
            sample_ctx.append(ctx_g[i])
            sample_graph.append(g)
    if len(sample_ctx) > 1:
        sample_ctx = np.array(sample_ctx)
        sim = sample_ctx @ sample_ctx.T
        # Normalize
        norms = np.linalg.norm(sample_ctx, axis=1, keepdims=True)
        norms[norms == 0] = 1
        sim = sim / (norms @ norms.T)
        im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Context cosine similarity\n(red = similar, blue = different)')
        plt.colorbar(im, ax=ax)
    
    # Panel 3: State weight evolution
    ax = axes[0, 2]
    sw = np.array(all_diags['blocked']['state_weight'])
    ax.plot(sw, color='#e67e22', linewidth=1.0)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('sigmoid(state_weight)')
    ax.set_title('State vs context weighting\n(>0.5 = state dominates)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(alpha=0.2)
    
    # Panel 4: Accuracy by transition type (blocked)
    ax = axes[1, 0]
    t_cid_correct = []  # transition 0->1 or 0->2 (context-identifying)
    t_rnd_correct = []  # other transitions
    t_all = []
    window = 10
    for i in range(len(accuracy_by_transition) - window):
        chunk = accuracy_by_transition[i:i + window]
        cid = [(t, s, n, c) for t, s, n, c in chunk if s == 0]
        rnd = [(t, s, n, c) for t, s, n, c in chunk if s != 0]
        if cid:
            t_cid_correct.append((chunk[-1][0], np.mean([c for _, _, _, c in cid])))
        if rnd:
            t_rnd_correct.append((chunk[-1][0], np.mean([c for _, _, _, c in rnd])))
        t_all.append((chunk[-1][0], np.mean([c for _, _, _, c in chunk])))
    
    if t_cid_correct:
        tx, ty = zip(*t_cid_correct)
        ax.plot(tx, smooth(np.array(ty), 50), color='#e74c3c', label='T_cid (0→1/2)', linewidth=1.0)
    if t_rnd_correct:
        tx, ty = zip(*t_rnd_correct)
        ax.plot(tx, smooth(np.array(ty), 50), color='#3498db', label='T_rnd (other)', linewidth=1.0)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Accuracy (EMA)')
    ax.set_title('Accuracy by transition type')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    
    # Panel 5: Amplification of match weights
    ax = axes[1, 1]
    se_blocked = np.array(all_diags['blocked']['softmax_entropy'])
    se_interleaved = np.array(all_diags['interleaved']['softmax_entropy'])
    v_blocked = ~np.isnan(se_blocked)
    v_interleaved = ~np.isnan(se_interleaved)
    
    if v_blocked.sum() > 0:
        ax.plot(np.arange(len(se_blocked))[v_blocked], se_blocked[v_blocked],
                color='#2c3e50', linewidth=0.5, alpha=0.7, label='blocked')
    if v_interleaved.sum() > 0:
        ax.plot(np.arange(len(se_interleaved))[v_interleaved], se_interleaved[v_interleaved],
                color='#e74c3c', linewidth=0.5, alpha=0.7, label='interleaved')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Softmax entropy')
    ax.set_title('Retrieval focus (lower = more focused)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    
    # Panel 6: Test-phase accuracy comparison
    ax = axes[1, 2]
    curricula_names = ['blocked', 'interleaved', 'IB']
    test_accs = []
    for curriculum in curricula:
        diag = all_diags[curriculum]
        correct = np.array(diag['correct'])
        # Last 200 timesteps = test phase
        test_start = max(0, len(correct) - 200)
        test_acc = np.mean(correct[test_start:])
        test_accs.append(test_acc)
    
    colors = ['#2c3e50', '#e74c3c', '#16a085']
    bars = ax.bar(curricula_names, test_accs, color=colors, alpha=0.8)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Final test accuracy comparison')
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')
    
    plt.tight_layout(pad=2.0)
    fig.savefig('figs/ego_context_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("→ Saved figs/ego_context_analysis.png")
    
    # -----------------------------------------------------------------------
    # Figure 3: What the model actually retrieves
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for cidx, (curriculum, diag) in enumerate(all_diags.items()):
        ax = axes[cidx]
        correct = np.array(diag['correct'])
        n = len(correct)
        
        # Plot accuracy
        ax.plot(smooth(correct[:n], 50), color='#2c3e50', linewidth=1.0)
        
        # Mark phases
        for pi, pb in enumerate(diag['phase_boundaries']):
            ax.axvline(x=pb, color='#e74c3c', linewidth=1.5, alpha=0.5)
        
        ax.set_title(f'{curriculum}', fontsize=11)
        ax.set_ylabel('Accuracy (EMA)', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        
        # Add summary stats
        train_end = diag['phase_boundaries'][-1] if 'phase_boundaries' in diag else n
        train_acc = np.mean(correct[:train_end]) if train_end > 0 else 0
        test_acc = np.mean(correct[train_end:]) if train_end < n else 0
        ax.text(0.02, 0.98, f'Train: {train_acc:.3f}\nTest: {test_acc:.3f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig('figs/ego_accuracy_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("→ Saved figs/ego_accuracy_overview.png")
    
    print("\nDone. All figures saved to figs/")


if __name__ == '__main__':
    make_figures()
