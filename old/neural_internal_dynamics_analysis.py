# %%
"""
Closed-loop hidden trajectories by initial input

This script trains the model and produces a single figure titled
'Closed-loop hidden trajectories by initial input'. It builds a PCA basis from
hidden states observed under constant input blocks (0.2, 0.8), then for a range
of initial inputs x0 in [0,1], it simulates closed-loop dynamics x_{t+1}=F(x_t),
projects hidden trajectories into the PCA plane, and plots all trajectories with
start/end markers and attractor stars.
"""

# ----------------------------- Imports --------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Literal

import plot_style
from configs import *  # noqa: F401,F403
from train_and_infer_functions import *  # noqa: F401,F403

plot_style.set_plot_style()
cs = plot_style.Color_scheme()

# ----------------------------- Train Model ----------------------------------
config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
config.use_mul_gating = True
config.use_add_gating = not config.use_mul_gating
if config.use_add_gating:
    config.input_size = config.input_size + np.prod(config.latent_dims)
config.rnn_type = 'lstm'
config.default_std = 0.3
config.log_hidden_states = True


# Baseline RNN (no latent learning)
baseline_rnn = True
if baseline_rnn:
    config.LU_lr = 0.0
    config.env_seed = 2
    # config.seq_len = 50
    if config.seq_len == 50: config.blocked_phase_length = 650
else:
    config.WU_lr = 1e-4

print('Running model with seed:', config.env_seed)
logger, model, config, _ = train_model(config, seed=config.env_seed,
                                       save_models=False, load_models=False, run_test_phase = False)

#%%
# ----------------------- Closed-loop helpers (PCA) ---------------------------
@torch.no_grad()
def single_step_hidden_dynamics(model, config, x_scalar, hidden_state, cell_state, step_idx):
    """One recurrent step, returns (y, h, c). Handles gating modes."""
    device = config.device
    x_tensor = torch.tensor([[x_scalar]], device=device, dtype=torch.float32)  # (1,1)
    if config.use_add_gating:
        latent_dim = int(np.prod(config.latent_dims))
        base_input_dim = config.input_size - latent_dim
        x_in = x_tensor.view(1, 1, 1).repeat(1, 1, base_input_dim)
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        latent_step = model.latent[:, seq_idx:seq_idx + 1, :]
        combined = torch.cat([x_in, model.latent_activation_function(latent_step)], dim=-1)
        x_proj = model.input_layer(combined)
    else:
        base_input_dim = config.input_size
        x_in = x_tensor.view(1, 1, 1).repeat(1, 1, base_input_dim)
        x_proj = model.input_layer(x_in)
    if config.use_mul_gating and config.pre_gating:
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        hidden_state, cell_state = model.apply_mul_gating([hidden_state, cell_state], seq_step=seq_idx, what_latent='self', taskID=None)
    if model.rnn_type == 'lstm':
        hidden_state, cell_state = model.rnn_cell(x_proj[:, 0, :], (hidden_state, cell_state))
    else:
        hidden_state = model.rnn_cell(x_proj[:, 0, :], hidden_state)
    if config.use_mul_gating and config.post_gating:
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        hidden_state, cell_state = model.apply_mul_gating([hidden_state, cell_state], seq_step=seq_idx, what_latent='self', taskID=None)
    y = model.output_layer(hidden_state)
    return y[0, 0].item(), hidden_state, cell_state

# New: differentiable one-step used for latent (Z) update

def single_step_hidden_dynamics_with_grad(model, config, x_scalar, hidden_state, cell_state, step_idx):
    """One recurrent step (with gradients), returns (y_scalar_tensor, h, c)."""
    device = config.device
    x_tensor = torch.tensor([[x_scalar]], device=device, dtype=torch.float32)
    if config.use_add_gating:
        latent_dim = int(np.prod(config.latent_dims))
        base_input_dim = config.input_size - latent_dim
        x_in = x_tensor.view(1, 1, 1).repeat(1, 1, base_input_dim)
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        latent_step = model.latent[:, seq_idx:seq_idx + 1, :]
        combined = torch.cat([x_in, model.latent_activation_function(latent_step)], dim=-1)
        x_proj = model.input_layer(combined)
    else:
        base_input_dim = config.input_size
        x_in = x_tensor.view(1, 1, 1).repeat(1, 1, base_input_dim)
        x_proj = model.input_layer(x_in)
    if config.use_mul_gating and config.pre_gating:
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        hidden_state, cell_state = model.apply_mul_gating([hidden_state, cell_state], seq_step=seq_idx, what_latent='self', taskID=None)
    if model.rnn_type == 'lstm':
        hidden_state, cell_state = model.rnn_cell(x_proj[:, 0, :], (hidden_state, cell_state))
    else:
        hidden_state = model.rnn_cell(x_proj[:, 0, :], hidden_state)
    if config.use_mul_gating and config.post_gating:
        seq_idx = min(step_idx, model.latent.shape[1] - 1)
        hidden_state, cell_state = model.apply_mul_gating([hidden_state, cell_state], seq_step=seq_idx, what_latent='self', taskID=None)
    y = model.output_layer(hidden_state)  # shape [1, out_dim]
    return y[0, 0], hidden_state, cell_state

# New: loss over a short block with constant input to mirror training

def _latent_block_loss(model, config, x_scalar, h0, c0, start_idx, T, l2_coeff=0.0):
    """Run T steps with constant input x_scalar starting from (h0,c0) and
    accumulate mean squared error to x_scalar. Returns (loss, hT, cT)."""
    h, c = h0, c0
    losses = []
    for t in range(int(T)):
        y_pred, h, c = single_step_hidden_dynamics_with_grad(model, config, x_scalar, h, c, start_idx + t)
        losses.append((y_pred - torch.tensor(x_scalar, device=y_pred.device, dtype=y_pred.dtype)) ** 2)
    loss = torch.stack(losses).mean()
    if l2_coeff and hasattr(model, 'latent'):
        loss = loss + l2_coeff * (model.latent.pow(2).mean())
    return loss, h, c

@torch.no_grad()
def collect_hidden_states_blocks(model, config, block_inputs=(0.2, 0.8), steps_per_block=80, repeats=3):
    """Collect hidden states under constant inputs for PCA basis. Returns (M,H)."""
    H_list = []
    for _ in range(repeats):
        for val in block_inputs:
            model.init_hidden(batch_size=1)
            h = model.hidden_state
            c = model.cell_state if model.rnn_type == 'lstm' else None
            for k in range(steps_per_block):
                _, h, c = single_step_hidden_dynamics(model, config, val, h, c, k)
                H_list.append(h.clone().cpu().numpy().squeeze(0))
    return np.vstack(H_list)

# Helper: safely write latent (broadcast constant across seq_len)

def _write_latent_broadcast(model, config, z_vec, requires_grad=False):
    z_tensor = z_vec.view(1, 1, -1).repeat(1, config.seq_len, 1)
    if hasattr(model, 'set_latent'):
        # set_latent may wrap as Parameter internally
        model.set_latent(z_tensor)
    else:
        if hasattr(model, 'latent') and isinstance(model.latent, torch.nn.Parameter):
            if model.latent.data.shape == z_tensor.shape:
                model.latent.data.copy_(z_tensor)
            else:
                model.latent = torch.nn.Parameter(z_tensor.detach().clone())
        else:
            model.latent = torch.nn.Parameter(z_tensor.detach().clone())
    model.latent.requires_grad_(requires_grad)

# New: closed-loop simulator with per-step Z updates

def simulate_closed_loop_from_x0_adaptive(model, config, x0, steps, update_lr=None, update_steps=1, adapt_mode: str = 'first_only', adapt_T: int | None = None):
    """Closed-loop rollout with online latent adaptation.
    At each step t:
      - If adapt_mode == 'every_step' or (adapt_mode == 'first_only' and t == 0):
          Update Z via gradient descent to fit a block of constant input of length adapt_T (defaults to config.seq_len).
      - Advance one recurrent step to produce y_t = F(x_t; Z) and set x_{t+1} = y_t.
    Returns dict with x_traj, h_traj, z_traj.
    """
    if update_lr is None:
        update_lr = float(getattr(config, 'LU_lr', 1e-2))
    l2_coeff = float(getattr(config, 'l2_loss', 0.0) or 0.0)
    T_block = int(adapt_T if adapt_T is not None else getattr(config, 'seq_len', 1))

    # Initialize hidden state
    model.init_hidden(batch_size=1)
    h = model.hidden_state.detach()
    c = model.cell_state.detach() if model.rnn_type == 'lstm' else None

    # Initialize latent vector from current model.latent or zeros
    latent_dim = int(np.prod(config.latent_dims)) if hasattr(config, 'latent_dims') else 0
    if latent_dim > 0 and hasattr(model, 'latent') and model.latent is not None:
        with torch.no_grad():
            L = latent_dim
            try:
                z0 = model.latent.detach().view(-1, L).mean(dim=0).to(config.device).float()
            except Exception:
                z0 = torch.zeros(latent_dim, device=config.device, dtype=torch.float32)
    else:
        z0 = torch.zeros(latent_dim, device=config.device, dtype=torch.float32)

    # Write initial latent (constant across time)
    _write_latent_broadcast(model, config, z0, requires_grad=False)

    xs, hs = [x0], []
    z_hist = []
    x_curr = float(x0)

    for k in range(steps):
        do_adapt = (adapt_mode == 'every_step') or (adapt_mode == 'first_only' and k == 0)
        if do_adapt:
            for _ in range(int(update_steps)):
                # Ensure model.latent accumulates grads
                model.latent.requires_grad_(True)
                if model.latent.grad is not None:
                    model.latent.grad.zero_()
                if hasattr(model, 'zero_grad'):
                    model.zero_grad(set_to_none=True)
                # Snapshot hidden state for adaptation block
                h0 = h.detach()
                c0 = c.detach() if c is not None else None
                # Compute loss over a block of constant inputs to mirror training
                loss, _, _ = _latent_block_loss(model, config, x_curr, h0, c0, start_idx=k, T=T_block, l2_coeff=l2_coeff)
                loss.backward()
                with torch.no_grad():
                    # Aggregate gradients only over the time slices touched in this block
                    idxs = [min(k + t, model.latent.shape[1] - 1) for t in range(T_block)]
                    grads = []
                    vals = []
                    for idx in idxs:
                        grads.append(model.latent.grad[:, idx, :].mean(dim=0))
                        vals.append(model.latent.detach()[:, idx, :].mean(dim=0))
                    g = torch.stack(grads, dim=0).mean(dim=0)
                    z_slice_mean = torch.stack(vals, dim=0).mean(dim=0)
                    z_new = z_slice_mean - update_lr * g
                    _write_latent_broadcast(model, config, z_new, requires_grad=False)
        # Log current Z (mean across time)
        with torch.no_grad():
            L = model.latent.shape[-1]
            z_mean_now = model.latent.view(-1, L).mean(dim=0).detach().cpu().numpy()
            z_hist.append(z_mean_now)

        # 2) Advance one step (no grad), update hidden and next input
        with torch.no_grad():
            y_val, h, c = single_step_hidden_dynamics(model, config, x_curr, h, c, k)
            xs.append(y_val)
            hs.append(h.clone().cpu().numpy().squeeze(0))
            x_curr = float(y_val)

    return {'x_traj': np.array(xs), 'h_traj': np.array(hs), 'z_traj': np.stack(z_hist, axis=0)}

def compute_pca_basis(hidden_states, k=2):
    mean = hidden_states.mean(0)
    X = hidden_states - mean
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        X = X + 1e-6 * np.random.randn(*X.shape)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:k]
    return mean, comps

def project_hidden(h, mean, comps):
    return (h - mean) @ comps.T

@torch.no_grad()
def simulate_closed_loop_from_x0(model, config, x0, steps):
    """Closed-loop rollout from initial scalar x0. Returns dict with x_traj, h_traj, p_traj."""
    model.init_hidden(batch_size=1)
    h = model.hidden_state
    c = model.cell_state if model.rnn_type == 'lstm' else None
    x_curr = x0
    xs, hs = [x_curr], []
    for k in range(steps):
        y, h, c = single_step_hidden_dynamics(model, config, x_curr, h, c, k)
        xs.append(y)
        hs.append(h.clone().cpu().numpy().squeeze(0))
        x_curr = y
    return {'x_traj': np.array(xs), 'h_traj': np.array(hs)}

# ----------------------- PCA input scan (single latent) ----------------------

def run_pca_input_scan(model, config, input_values, gen_steps=50, block_inputs=(0.2, 0.8), attractor_tol=2e-3):
    """PCA from block data, closed-loop trajs for each x0, cluster endpoints."""
    hidden_states_blocks = collect_hidden_states_blocks(model, config, block_inputs=block_inputs)
    mean_h, comps = compute_pca_basis(hidden_states_blocks, k=2)
    trajectories = []
    for x0 in input_values:
        td = simulate_closed_loop_from_x0(model, config, x0, steps=gen_steps)
        td['p_traj'] = project_hidden(td['h_traj'], mean_h, comps)
        trajectories.append(td)
    endpoints = np.vstack([td['p_traj'][-1] for td in trajectories])
    rounded = np.round(endpoints / attractor_tol).astype(int)
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
    n_clusters = len(uniq)
    cluster_ids = inv
    cluster_centers = np.vstack([endpoints[cluster_ids == k].mean(0) for k in range(n_clusters)])
    order = np.argsort(cluster_centers[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    cluster_ids = np.array([remap[c] for c in cluster_ids])
    cluster_centers = cluster_centers[order]
    return {
        'mean_h': mean_h,
        'comps': comps,
        'input_values': np.array(input_values),
        'trajectories': trajectories,
        'endpoints': endpoints,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
        'n_clusters': n_clusters,
        'attractor_tol': attractor_tol,
    }

# New: PCA input scan with adaptive latent updates

def run_pca_input_scan_adaptive(model, config, input_values, gen_steps=5, block_inputs=(0.2, 0.8), attractor_tol=2e-3, update_lr=None, update_steps=1, adapt_mode: str = 'first_only'):
    """Same as run_pca_input_scan but uses adaptive latent updates (optionally only at first step)."""
    hidden_states_blocks = collect_hidden_states_blocks(model, config, block_inputs=block_inputs)
    mean_h, comps = compute_pca_basis(hidden_states_blocks, k=2)
    trajectories = []
    for x0 in input_values:
        td = simulate_closed_loop_from_x0_adaptive(
            model, config, x0, steps=gen_steps, update_lr=update_lr, update_steps=update_steps, adapt_mode=adapt_mode
        )
        td['p_traj'] = project_hidden(td['h_traj'], mean_h, comps)
        trajectories.append(td)
    endpoints = np.vstack([td['p_traj'][-1] for td in trajectories])
    rounded = np.round(endpoints / attractor_tol).astype(int)
    uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
    n_clusters = len(uniq)
    cluster_ids = inv
    cluster_centers = np.vstack([endpoints[cluster_ids == k].mean(0) for k in range(n_clusters)])
    order = np.argsort(cluster_centers[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    cluster_ids = np.array([remap[c] for c in cluster_ids])
    cluster_centers = cluster_centers[order]
    return {
        'mean_h': mean_h,
        'comps': comps,
        'input_values': np.array(input_values),
        'trajectories': trajectories,
        'endpoints': endpoints,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
        'n_clusters': n_clusters,
        'attractor_tol': attractor_tol,
    }

# New: PCA trajectories plotting for multiple result sets (reintroduced)

def plot_closed_loop_pca_multi(results_dict, save_prefix, cmap_name='viridis'):
    """Plot PCA-plane trajectories for one or more results dicts.
    results_dict: {label: res}, where res is output of run_pca_input_scan(_adaptive).
    """
    if not results_dict:
        return None
    # Assume all results share the same PCA basis; use the first for axis labels.
    first_key = next(iter(results_dict))
    res0 = results_dict[first_key]
    input_values = res0['input_values']

    fig, ax = plt.subplots(1, 1, figsize=(cs.panel_small_size[0]*1.6, cs.panel_small_size[1]*1.2))

    for label, res in results_dict.items():
        cm = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=res['input_values'].min(), vmax=res['input_values'].max())
        # Plot trajectories colored by initial input
        for x0, td in zip(res['input_values'], res['trajectories']):
            p = td['p_traj']  # [T, 2]
            ax.plot(p[:, 0], p[:, 1], color=cm(norm(x0)), lw=1.2, alpha=0.95)
            # start/end markers (no labels to avoid duplicate legend entries)
            ax.plot(p[0, 0], p[0, 1], marker='o', ms=3, color=cm(norm(x0)), mec='none', alpha=0.9)
            ax.plot(p[-1, 0], p[-1, 1], marker='s', ms=3, color=cm(norm(x0)), mec='k', mew=0.5, alpha=0.9)
        # End-state centers (formerly "attractors"): color each star by the mean x0 of trajectories in that cluster
        if 'cluster_centers' in res and res['cluster_centers'] is not None:
            cc = res['cluster_centers']
            cluster_ids = res.get('cluster_ids', None)
            xvals = res.get('input_values', None)
            if cluster_ids is not None and xvals is not None:
                for k in range(cc.shape[0]):
                    mask = (cluster_ids == k)
                    mean_x0 = float(xvals[mask].mean()) if np.any(mask) else float(xvals.mean())
                    ax.plot(cc[k, 0], cc[k, 1], '*', color=cm(norm(mean_x0)), ms=10, mec='w', mew=0.8)
            else:
                ax.plot(cc[:, 0], cc[:, 1], '*', color='k', ms=10, mec='w', mew=0.8)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    # ax.grid(alpha=0.2, linestyle=':')

    ###
    # Legend proxies for start (initial input) and attractor (end-point star)
    ax.plot([], [], linestyle='None', marker='o', ms=4, color='k', label='Initial state')
    ax.plot([], [], linestyle='None', marker='*', ms=10, markerfacecolor='k', markeredgecolor='w', mew=0.8,
        color='k', label='End state')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

    # Enforce equal scaling on both axes so PC variances are comparable visually
    ax.set_aspect('equal', adjustable='datalim')

    # Shared colorbar based on first result's input range
    cm = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=input_values.min(), vmax=input_values.max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Initial input x0')

    model_label = f"{'neuragem' if not baseline_rnn else 'mrnn' if config.seq_len == 50 else 'rnn'}"
    outfile = f"{save_prefix}_closed_loop_pca_multi_{model_label}.pdf"
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=150, transparent=True)
    print(f'Figure saved to: {outfile}')
    return fig

# New: Plot Z evolution per initial input (same color mapping)

def plot_z_evolution_for_results(res, save_prefix, cmap_name='viridis'):
    trajectories = res.get('trajectories', [])
    if not trajectories or 'z_traj' not in trajectories[0]:
        return None
    input_values = res['input_values']
    cm = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=input_values.min(), vmax=input_values.max())

    z_dim = trajectories[0]['z_traj'].shape[1]
    n_rows = z_dim
    fig, axes = plt.subplots(n_rows, 1, figsize=(cs.panel_small_size[0]*1.8, max(1.0, 0.8*z_dim)), sharex=True)
    if n_rows == 1:
        axes = [axes]
    t = None
    for x0, td in zip(input_values, trajectories):
        zt = td['z_traj']  # [steps, z_dim]
        if t is None:
            t = np.arange(1, zt.shape[0]+1)
        for d in range(z_dim):
            axes[d].plot(t, zt[:, d], color=cm(norm(x0)), lw=1.0, alpha=0.9)
    for d in range(z_dim):
        axes[d].set_ylabel(f'Z[{d}]')
        axes[d].grid(alpha=0.2, linestyle=':')
    axes[-1].set_xlabel('Time step')
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Initial input x0')

    model_label = f"{'neuragem' if not baseline_rnn else 'mrnn' if config.seq_len == 50 else 'rnn'}"
    outfile = f"{save_prefix}_z_evolution_{model_label}.pdf"
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=150, transparent=True)
    print(f'Figure saved to: {outfile}')
    return fig

# ------------------------------ Run & Save -----------------------------------
try:
    INPUT_SCAN = np.linspace(0.0, 1.0, 11)
    GEN_STEPS = 20
    update_steps = 20
    adapt_mode: Literal['every_step', 'first_only'] = 'every_step'

    
    latent_dim = int(np.prod(config.latent_dims))
    if not baseline_rnn:
        # Disable L2 regularization on Z for adaptive latent updates
        config.l2_loss = 0.0
        model.config = config
        model.LU_optimizer =  model.get_LU_optimizer()

        # Adaptive latent updates across initial inputs; adapt Z only on the first step to avoid collapse
        res = run_pca_input_scan_adaptive(
            model, config, input_values=INPUT_SCAN, gen_steps=GEN_STEPS,
            block_inputs=(0.2, 0.8), attractor_tol=2e-3,
            update_lr=3*float(getattr(config, 'LU_lr', 1e-2)), update_steps=update_steps, adapt_mode=adapt_mode
        )
        _ = plot_closed_loop_pca_multi({'online-z': res}, save_prefix=f"{config.export_path}{config.dataset_name}")
        _ = plot_z_evolution_for_results(res, save_prefix=f"{config.export_path}{config.dataset_name}")
    else:
        res = run_pca_input_scan(model, config, input_values=INPUT_SCAN, gen_steps=GEN_STEPS,
                                 block_inputs=(0.2, 0.8), attractor_tol=2e-3)
        _ = plot_closed_loop_pca_multi({'[0,0]': res}, save_prefix=f"{config.export_path}{config.dataset_name}")
except Exception as e:
    print(f"[WARN] Closed-loop PCA multi-latent analysis failed: {e}")

#%%