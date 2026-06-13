"""Neural Dissection figure, panels A through C. 

Paper-ready script for three conditions:
- NeuraGEM
- RNN^{short}  (LU_lr = 0)
- RNN^{long}   (LU_lr = 0, seq_len = 50)
"""

import logging

logging.getLogger("matplotlib.font_manager").disabled = True

import numpy as np
import torch
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()
cs = plot_style.Color_scheme()

from configs import ContextualSwitchingTaskConfig
from train_and_infer_functions import train_model


# ----------------------------------------------------------------------------
# Config / condition helpers

ADAPTATION_STEPS_TO_PLOT = 3
ADAPTATION_DISPLAY_MODE = "arrows"  # 'arrows' or 'lines'
SCAN_X_RANGE = (-0.2, 1.2)
SCAN_N_POINTS = 181
PROBE_X = np.linspace(0.0, 1.0, 15)


def slugify(name):
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return "_".join(part for part in slug.split("_") if part)


def format_model_title(name):
    if name == "RNN^{short}":
        return r"RNN$^{\mathrm{short}}$"
    if name == "RNN^{long}":
        return r"RNN$^{\mathrm{long}}$"
    return name


def build_base_config():
    config = ContextualSwitchingTaskConfig(experiment_to_run="figure")

    config.rnn_type = "lstm"
    config.default_std = 0.3

    config.save_model = False
    config.load_saved_model = False
    return config


def apply_condition_profile(config, model_family, overrides):
    if model_family == "neuragem":
        # NeuraGEM-specific additive gating setup from the original script.
        config.use_mul_gating = False
        config.use_add_gating = True
        if config.use_add_gating:
            config.input_size = config.input_size + int(np.prod(config.latent_dims))
        config.WU_lr = 1e-4
        config.blocked_phase_length = 840

    for key, value in overrides.items():
        setattr(config, key, value)

    if model_family == "baseline_rnn":
        # Baseline-specific settings from the original script.
        config.LU_lr = 0.0
        if "env_seed" not in overrides:
            config.env_seed = 2
        if "blocked_phase_length" not in overrides:
            config.blocked_phase_length = 610 if int(config.seq_len) == 50 else 960

    return config


def is_baseline_condition(config):
    return float(getattr(config, "LU_lr", 1.0)) == 0.0


def get_z_settings(config):
    latent_dim = int(np.prod(config.latent_dims))
    if is_baseline_condition(config):
        return {"[0, 0]": [0.0] * latent_dim}
    return {
        "[2.0, -2.0]": [2.0, -2.0][:latent_dim],
        "[0.5, -0.5]": [0.5, -0.5][:latent_dim],
        "[0.0, 0.0]": [0.0] * latent_dim,
        "[-0.5, 0.5]": [-0.5, 0.5][:latent_dim],
        "[-2.0, 2.0]": [-2.0, 2.0][:latent_dim],
    }


def get_behavior_input_dim(config):
    if getattr(config, "use_add_gating", False):
        return int(config.input_size - np.prod(config.latent_dims))
    return int(config.input_size)


# ----------------------------------------------------------------------------
# Drift / adaptation analysis helpers (dependency chain for plot_drift_only)

@torch.no_grad()
def _one_step_F(model, config, x_scalar: float, input_dim: int, reset_hidden: bool = True):
    if reset_hidden and hasattr(model, "init_hidden"):
        try:
            model.init_hidden(batch_size=1)
        except TypeError:
            model.init_hidden()

    x_tensor = torch.full((1, config.seq_len, input_dim), float(x_scalar), device=config.device)
    if config.use_add_gating:
        combined = model.combine_input_with_latent(
            x_tensor, what_latent=config.what_latent_to_use, taskID=None
        )
        outs, _ = model(combined, taskID=None, what_latent=config.what_latent_to_use)
    else:
        outs, _ = model(x_tensor, taskID=None, what_latent=config.what_latent_to_use)
    outs = torch.stack(outs, dim=1)
    return float(outs[0, -1, 0].item())


@torch.no_grad()
def scan_fixed_points(model, config, z_dict, x_min=0.0, x_max=1.0, n_points=201):
    input_dim = get_behavior_input_dim(config)
    x_grid = np.linspace(x_min, x_max, n_points)
    results = {}

    current_latent = model.latent.clone() if hasattr(model, "latent") and model.latent is not None else None
    latent_dim = int(np.prod(config.latent_dims))

    for name, z_val in z_dict.items():
        z_arr = np.array(z_val, dtype=float)
        if z_arr.size != latent_dim:
            raise ValueError(f"Provided z for '{name}' has size {z_arr.size}, expected {latent_dim}")

        z_tensor = (
            torch.tensor(z_arr, device=config.device, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(1, config.seq_len, 1)
        )
        model.set_latent(z_tensor)

        Fx = np.array([_one_step_F(model, config, x, input_dim, reset_hidden=True) for x in x_grid])
        delta = Fx - x_grid
        abs_err = np.abs(delta)
        dFdx = np.gradient(Fx, x_grid)
        dDelta_dx = dFdx - 1.0
        dAbs_dx = np.gradient(abs_err, x_grid)

        sign = np.sign(delta)
        zero_idx = np.where(np.diff(sign) != 0)[0]
        fps = []
        for idx in zero_idx:
            x0, x1 = x_grid[idx], x_grid[idx + 1]
            y0, y1 = delta[idx], delta[idx + 1]
            x_fp = x0 if (y1 - y0) == 0 else x0 - y0 * (x1 - x0) / (y1 - y0)
            d_idx = idx if abs(x_fp - x0) < abs(x_fp - x1) else idx + 1
            slope = dFdx[d_idx]
            fps.append({"x": x_fp, "dFdx": slope, "stable": abs(slope) < 1.0})

        results[name] = {
            "x": x_grid,
            "Fx": Fx,
            "delta": delta,
            "abs_err": abs_err,
            "dFdx": dFdx,
            "dDelta_dx": dDelta_dx,
            "dAbs_dx": dAbs_dx,
            "fixed_points": fps,
        }

    if current_latent is not None:
        model.set_latent(current_latent)
    return results


def _predict_single(model, config, x_scalar, input_dim, reset_hidden: bool = True):
    if reset_hidden and hasattr(model, "init_hidden"):
        try:
            model.init_hidden(batch_size=1)
        except TypeError:
            model.init_hidden()

    x_tensor = torch.full((1, config.seq_len, input_dim), float(x_scalar), device=config.device)
    if config.use_add_gating:
        combined = model.combine_input_with_latent(
            x_tensor, what_latent=config.what_latent_to_use, taskID=None
        )
        outs, _ = model(combined, taskID=None, what_latent=config.what_latent_to_use)
    else:
        outs, _ = model(x_tensor, taskID=None, what_latent=config.what_latent_to_use)
    outs = torch.stack(outs, dim=1)
    return outs[0, -1, 0]


def compute_adaptation_trajectories(model, config, x_values, steps=3, target_mode="self", lr_scale=1.0):
    if not hasattr(model, "latent") or model.latent is None:
        return None

    latent_dim = int(np.prod(config.latent_dims)) if hasattr(config, "latent_dims") else 0
    if latent_dim == 0:
        return None

    input_dim = get_behavior_input_dim(config)
    device = config.device
    original_latent_value = model.latent.detach().clone()
    LU_lr_backup = getattr(config, "LU_lr", None)
    x_values = np.array(x_values)

    if LU_lr_backup is not None and lr_scale != 1.0:
        config.LU_lr = LU_lr_backup * lr_scale
        if hasattr(model, "get_LU_optimizer"):
            model.get_LU_optimizer()

    deltas = np.zeros((steps + 1, len(x_values)))
    preds = np.zeros((steps + 1, len(x_values)))

    for xi, x in enumerate(x_values):
        if hasattr(model, "reset_latent"):
            model.reset_latent(batch_size=1, seq_len=config.seq_len)
        if model.latent is not None:
            model.latent.requires_grad_(True)

        with torch.no_grad():
            p0 = float(_predict_single(model, config, x, input_dim, reset_hidden=True).item())
        preds[0, xi] = p0
        deltas[0, xi] = p0 - x

        for k in range(steps):
            if hasattr(model, "LU_optimizer"):
                model.LU_optimizer.param_groups[0]["lr"] = config.LU_lr * (np.power(lr_scale, k))
                model.LU_optimizer.zero_grad()

            pred = _predict_single(model, config, x, input_dim, reset_hidden=True)
            target_val = x if target_mode == "self" else float(target_mode)
            target = torch.tensor(target_val, device=device, dtype=pred.dtype)
            loss = (pred - target) ** 2
            loss.backward()

            if hasattr(model, "LU_optimizer"):
                model.LU_optimizer.step()
            elif model.latent.grad is not None:
                model.latent.data -= (getattr(config, "LU_lr", 0.01)) * model.latent.grad.data
                model.latent.grad.zero_()

            with torch.no_grad():
                pk = float(_predict_single(model, config, x, input_dim, reset_hidden=True).item())
            preds[k + 1, xi] = pk
            deltas[k + 1, xi] = pk - x

    if model.latent is not None:
        if model.latent.data.shape == original_latent_value.shape:
            model.latent.data.copy_(original_latent_value)
        else:
            model.latent = torch.nn.Parameter(original_latent_value)
            if hasattr(model, "get_LU_optimizer"):
                model.get_LU_optimizer()

    if LU_lr_backup is not None and lr_scale != 1.0:
        config.LU_lr = LU_lr_backup
        if hasattr(model, "get_LU_optimizer"):
            model.get_LU_optimizer()

    return {"x": x_values, "deltas": deltas, "preds": preds}


def compute_repetition_trajectories_baseline(model, config, x_values, steps=3):
    """Pseudo-adaptation for LU_lr=0 baselines by repeated probe presentation."""
    x_values = np.array(x_values)
    n = len(x_values)
    deltas = np.zeros((steps + 1, n))
    preds = np.zeros((steps + 1, n))
    input_dim = get_behavior_input_dim(config)
    device = config.device

    for xi, x in enumerate(x_values):
        for k in range(steps + 1):
            repeats = k + 1
            seq = torch.full((1, repeats, input_dim), float(x), device=device)
            model.init_hidden(batch_size=1)

            if not config.use_input_attention:
                proc_seq = model.input_layer(seq)

            hidden_state = model.hidden_state
            cell_state = model.cell_state if model.rnn_type == "lstm" else None
            outputs = []
            for t in range(repeats):
                if config.use_mul_gating and config.pre_gating:
                    hidden_state, cell_state = model.apply_mul_gating(
                        hidden_state, cell_state, seq_step=t, what_latent="self", taskID=None
                    )
                if config.use_input_attention:
                    input_ts = model.apply_input_gating(seq[:, t, ...])
                    input_ts = model.input_layer(input_ts)
                else:
                    input_ts = proc_seq[:, t, ...]

                if model.rnn_type == "lstm":
                    hidden_state, cell_state = model.rnn_cell(input_ts, (hidden_state, cell_state))
                else:
                    hidden_state = model.rnn_cell(input_ts, hidden_state)

                if config.use_mul_gating and config.post_gating:
                    hidden_state, cell_state = model.apply_mul_gating(
                        [hidden_state, cell_state], seq_step=t, what_latent="self", taskID=None
                    )
                outputs.append(model.output_layer(hidden_state))

            pred = outputs[-1][0, 0].item()
            preds[k, xi] = pred
            deltas[k, xi] = pred - x

    return {"x": x_values, "deltas": deltas, "preds": preds, "mode": "baseline_repeats"}


def plot_drift_only(
    results,
    save_path=None,
    placeholder_title="Adaptation Trajectories",
    adaptation=None,
    traj=None,
    traj_mode="arrows",
):
    """Drift plot with adaptation trajectories overlaid in drift space."""
    fig, ax = plt.subplots(1, 1, figsize=cs.panel_large_size)
    drift_ax = ax
    adapt_ax = ax

    def _open_v_arrow(
        axis,
        x,
        y0,
        y1,
        color="#888888",
        alpha=0.65,
        lw=1.0,
        head_frac=0.06,
        head_width_frac=0.026,
        zorder=1,
    ):
        dy = y1 - y0
        if dy == 0:
            return
        axis.plot([x, x], [y0, y1], color=color, alpha=alpha, lw=lw, zorder=zorder)
        yspan = axis.get_ylim()[1] - axis.get_ylim()[0]
        xspan = axis.get_xlim()[1] - axis.get_xlim()[0]
        head_len = min(head_frac * yspan, abs(dy) * 0.6)
        head_w = head_width_frac * xspan
        tip_y = y1
        base_y = y1 - head_len if dy > 0 else y1 + head_len
        axis.plot([x, x - head_w], [tip_y, base_y], color=color, alpha=alpha, lw=lw, zorder=zorder)
        axis.plot([x, x + head_w], [tip_y, base_y], color=color, alpha=alpha, lw=lw, zorder=zorder)

    latent_names = list(results.keys())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(3, len(latent_names))))

    for li, (name, d) in enumerate(results.items()):
        col = colors[li]
        drift_ax.plot(d["x"], d["delta"], label=name, lw=1.2, color=col)
        for fp in d["fixed_points"]:
            marker = "o" if fp["stable"] else "s"
            drift_ax.scatter(fp["x"], 0, c="k", s=22, marker=marker, zorder=6)
            drift_ax.axvline(fp["x"], color=col, ls=":", alpha=0.25, lw=0.9)

    drift_ax.axhline(0, color="k", lw=0.8)
    drift_ax.set_xlabel("Inputs (x).")
    drift_ax.set_ylabel("Asymptotic bias Δ = F(x)-x.")
    legend = drift_ax.legend(
        fontsize=6,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(0.53, 0.64),
        title="Fixed Z values",
        title_fontsize=6,
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    arrow_x = 0.05
    arrow_y_start = -0.35
    arrow_y_end = -0.45
    _open_v_arrow(drift_ax, arrow_x, arrow_y_start, arrow_y_end, color="#888888", alpha=0.75, lw=1.0, zorder=5)
    drift_ax.text(
        arrow_x + 0.05,
        (arrow_y_start + arrow_y_end) / 2,
        "Bias for inputs\n$X_t=1$ $X_t=2$ $X_t=3$",
        fontsize=6,
        color="#444444",
        va="center",
        ha="left",
    )

    adapt_ax.set_xlabel("Inputs (x).")
    adapt_ax.set_ylabel("Asymptotic bias Δ = F(x)-x.")
    adapt_ax.axhline(0, color="k", lw=0.5, alpha=0.6)

    if traj is not None:
        xs = traj["x"]
        deltas = traj["deltas"]
        steps = deltas.shape[0] - 1

        if traj_mode == "lines":
            for i, x in enumerate(xs):
                y_series = deltas[:, i]
                adapt_ax.plot([x] * len(y_series), y_series, "-o", ms=3, lw=0.9, color="#555555", alpha=0.9, zorder=2)
        else:
            adapt_ax.set_xlim([xs.min() - 0.05, xs.max() + 0.05])
            y_all = deltas.flatten()
            y_pad = (y_all.max() - y_all.min()) * 0.08 if y_all.max() > y_all.min() else 0.1
            adapt_ax.set_ylim([y_all.min() - y_pad, y_all.max() + y_pad])
            for i, x in enumerate(xs):
                y_series = deltas[:, i]
                for k in range(steps):
                    _open_v_arrow(adapt_ax, x, y_series[k], y_series[k + 1], color="#888888", alpha=0.75, lw=1.0, zorder=1)
    else:
        adapt_ax.text(0.5, 0.5, "No trajectories", ha="center", va="center", fontsize=8, alpha=0.5)

    if traj is not None and traj_mode == "lines":
        adapt_ax.set_xlim([xs.min() - 0.05, xs.max() + 0.05])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150, transparent=True)
        print(f"Figure saved to: {save_path}")
    return fig


# ----------------------------------------------------------------------------
# Run a condition


def run_drift_condition(model_name, overrides, title_color, model_family):
    config = apply_condition_profile(build_base_config(), model_family, overrides)
    config.run_name = f"additive_drift_{slugify(model_name)}"

    print("Running:", model_name, "seed:", config.env_seed)
    logger, model, config, _ = train_model(
        config,
        seed=config.env_seed,
        save_models=False,
        load_models=False,
    )
    del logger  # training logs are not used here
    model.rnn_type = config.rnn_type  # Ensure model has rnn_type attribute for trajectory computation
    z_settings = get_z_settings(config)
    dyn_results = scan_fixed_points(
        model,
        config,
        z_settings,
        x_min=SCAN_X_RANGE[0],
        x_max=SCAN_X_RANGE[1],
        n_points=SCAN_N_POINTS,
    )

    if is_baseline_condition(config):
        traj = compute_repetition_trajectories_baseline(
            model,
            config,
            PROBE_X,
            steps=ADAPTATION_STEPS_TO_PLOT,
        )
    else:
        traj = compute_adaptation_trajectories(
            model,
            config,
            PROBE_X,
            steps=ADAPTATION_STEPS_TO_PLOT,
            target_mode="self",
            lr_scale=1.0,
        )

    out_path = f"{config.export_path}{config.dataset_name}_{slugify(model_name)}_drift_only.pdf"
    fig = plot_drift_only(dyn_results, save_path=None, traj=traj, traj_mode=ADAPTATION_DISPLAY_MODE)
    fig.suptitle(format_model_title(model_name), color=title_color, fontsize=8, y=0.995)
    fig.savefig(out_path, bbox_inches="tight", dpi=150, transparent=True)
    print(f"Figure saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    conditions = [
        {
            "model_name": "RNN^{short}",
            "overrides": {},
            "title_color": cs.short_horizon_rnn,
            "model_family": "baseline_rnn",
        },
        {
            "model_name": "RNN^{long}",
            "overrides": {"seq_len": 50},
            "title_color": cs.long_horizon_rnn,
            "model_family": "baseline_rnn",
        },
        {
            "model_name": "NeuraGEM",
            "overrides": {},
            "title_color": cs.neuragem,
            "model_family": "neuragem",
        },
    ]

    for condition in conditions:
        try:
            run_drift_condition(**condition)
        except Exception as exc:
            print(f"Drift analysis failed for {condition['model_name']}: {exc}")
