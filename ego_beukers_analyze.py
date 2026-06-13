"""
Systematic analysis of EGO sweep results on Beukers et al. task.
Aggregates across seeds, computes Tcid accuracy, and generates
head-to-head comparison with NeuraGEM.
"""
import os, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.set_plot_style()


# ---------------------------------------------------------------------------
# Tcid accuracy computation (same metric as NeuraGEM)
# ---------------------------------------------------------------------------

def compute_tcid_accuracy(logger):
    """
    Test-phase Tcid accuracy: given state 3, does model predict the
    context-appropriate next state (5 for ctx 0, 6 for ctx 1)?
    
    Uses preds[idx+1] with predict_first_frame=True semantics.
    EGO predicts next_t from state_t — direct one-step prediction.
    """
    inputs = np.concatenate(logger.inputs, axis=0).squeeze(1)
    preds = np.concatenate(logger.predicted_outputs, axis=0).squeeze(1)
    hlcids = np.concatenate(logger.hlcids, axis=0).squeeze(1).squeeze(-1)

    phases = logger.phases
    test_start = phases[-1][1] if len(phases) > 1 else 0

    # EGO predicts next_t directly from state_t.
    # preds[idx+1] is the prediction for the state AFTER the logged input.
    state3_mask = inputs[:, 3] > 0.5
    tcid_indices = np.where(state3_mask)[0]
    test_tcid = [i for i in tcid_indices if i >= test_start and i < len(hlcids) - 1]

    correct, total = 0, 0
    correct_ctx0, total_ctx0 = 0, 0
    correct_ctx1, total_ctx1 = 0, 0

    for idx in test_tcid:
        true_ctx = hlcids[idx + 1]
        correct_state = 5 if true_ctx == 0 else 6
        is_correct = (np.argmax(preds[idx + 1]) == correct_state)

        if is_correct:
            correct += 1
        total += 1

        if true_ctx == 0:
            if is_correct:
                correct_ctx0 += 1
            total_ctx0 += 1
        else:
            if is_correct:
                correct_ctx1 += 1
            total_ctx1 += 1

    acc = correct / total if total else 0.0
    acc0 = correct_ctx0 / total_ctx0 if total_ctx0 else 0.0
    acc1 = correct_ctx1 / total_ctx1 if total_ctx1 else 0.0
    return acc, total, acc0, acc1


def compute_context_encoding(logger):
    """Linear regression: how much does EGO context encode Tcid vs Trnd?"""
    inputs = np.concatenate(logger.inputs, axis=0).squeeze(1)
    ctx_vals = np.concatenate(logger.latent_values, axis=0).squeeze(1)

    phases = logger.phases
    test_start = phases[-1][1]
    ctx_test = ctx_vals[test_start:]
    inputs_test = inputs[test_start:]

    tcid_mask = np.zeros(len(inputs_test))
    for i in range(1, len(inputs_test)):
        if inputs_test[i - 1, 3] > 0.5:
            tcid_mask[i] = 1.0
    trnd_mask = np.zeros(len(inputs_test))
    for i in range(1, len(inputs_test)):
        if inputs_test[i - 1, 4] > 0.5:
            trnd_mask[i] = 1.0

    X = np.column_stack([np.ones(len(tcid_mask)), tcid_mask, trnd_mask])
    beta_tcid, beta_trnd = 0.0, 0.0
    try:
        for d in range(ctx_test.shape[1]):
            y = ctx_test[:, d]
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            beta_tcid += abs(betas[1])
            beta_trnd += abs(betas[2])
    except Exception:
        pass
    return beta_tcid, beta_trnd


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate_sweep(base_dir='./exports/ego_beukers'):
    """Aggregate Tcid accuracy across all seeds for each hyperparameter combo."""
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    hidden_dims = [16, 64]
    context_dims = [2, 4, 8]

    summary = {}
    for curriculum in curricula:
        summary[curriculum] = {}
        for hd in hidden_dims:
            for cd in context_dims:
                key = f'h{hd}_c{cd}'
                pattern = os.path.join(base_dir, key, curriculum, 'ego_*.pkl')
                files = sorted(glob.glob(pattern))

                accs, accs0, accs1 = [], [], []
                betas_tcid, betas_trnd = [], []

                for fp in files:
                    try:
                        with open(fp, 'rb') as f:
                            logger = pickle.load(f)
                        acc, n, a0, a1 = compute_tcid_accuracy(logger)
                        bt, br = compute_context_encoding(logger)
                        accs.append(acc)
                        accs0.append(a0)
                        accs1.append(a1)
                        betas_tcid.append(bt)
                        betas_trnd.append(br)
                    except Exception as e:
                        print(f'  WARN: {fp}: {e}')

                if accs:
                    summary[curriculum][key] = {
                        'acc_mean': np.mean(accs),
                        'acc_sem': np.std(accs) / np.sqrt(len(accs)),
                        'acc_ctx0': np.mean(accs0),
                        'acc_ctx1': np.mean(accs1),
                        'beta_tcid': np.mean(betas_tcid),
                        'beta_trnd': np.mean(betas_trnd),
                        'n_seeds': len(accs),
                        'accs_all': accs,  # for boxplots
                    }

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_main_summary(summary, export_dir):
    """Head-to-head comparison: EGO vs NeuraGEM Tcid accuracy."""
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    curricula_labels = ['Blocked', 'Interleaved', 'Interleaved→Blocked']
    hidden_context_keys = [(16, 2), (16, 4), (16, 8), (64, 2), (64, 4), (64, 8)]

    # --- Panel A: Sweep results heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ci, curriculum in enumerate(curricula):
        ax = axes[ci]

        # Build matrix: rows=hidden_d, cols=context_d
        data = np.zeros((2, 3))
        for hi, hd in enumerate([16, 64]):
            for cj, cd in enumerate([2, 4, 8]):
                key = f'h{hd}_c{cd}'
                if key in summary.get(curriculum, {}):
                    data[hi, cj] = summary[curriculum][key]['acc_mean']

        im = ax.imshow(data, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['c=2', 'c=4', 'c=8'])
        ax.set_yticks(range(2))
        ax.set_yticklabels(['h=16', 'h=64'])
        ax.set_title(curricula_labels[ci], fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for hi in range(2):
            for cj in range(3):
                ax.text(cj, hi, f'{data[hi, cj]:.3f}', ha='center', va='center',
                        color='white' if data[hi, cj] < 0.65 else 'black', fontsize=9)

    plt.suptitle('EGO Tcid Accuracy by Hyperparameters', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(export_dir, 'ego_sweep_heatmap.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {fig_path}')

    # --- Panel B: Best EGO config vs NeuraGEM bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Load NeuraGEM results (seed 42 single-run, for reference)
    ng_dir = './exports/seq_learn/fig6_replication'
    ng_accs = {}
    for c in curricula:
        try:
            with open(f'{ng_dir}/logger_{c}_seed42.pkl', 'rb') as f:
                logger = pickle.load(f)
            acc, n, a0, a1 = compute_tcid_accuracy(logger)
            ng_accs[c] = acc
        except Exception:
            ng_accs[c] = None

    # For EGO, pick the best config per curriculum
    best_configs = {}
    for curriculum in curricula:
        best_acc = 0
        best_key = None
        for hd, cd in hidden_context_keys:
            key = f'h{hd}_c{cd}'
            if key in summary.get(curriculum, {}):
                a = summary[curriculum][key]['acc_mean']
                if a > best_acc:
                    best_acc = a
                    best_key = key
        best_configs[curriculum] = best_key

    for ci, curriculum in enumerate(curricula):
        ax = axes[ci]
        x_pos = [0, 1]
        labels = ['NeuraGEM\n(1 seed)', 'EGO\n(20 seeds)']

        ng_acc = ng_accs.get(curriculum)
        ego_key = best_configs[curriculum]
        ego_data = summary[curriculum].get(ego_key, {})

        heights = []
        errs = []
        colors = []

        if ng_acc is not None:
            heights.append(ng_acc)
            errs.append(0)
            colors.append('#2196F3')

        if ego_data:
            heights.append(ego_data['acc_mean'])
            errs.append(ego_data['acc_sem'])
            colors.append('#FF9800')

        bars = ax.bar(labels[:len(heights)], heights, color=colors,
                       yerr=errs[:len(heights)] if errs[0] != 0 else None,
                       capsize=5 if errs[0] != 0 else 0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{curricula_labels[ci]}')
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{h:.3f}', ha='center', fontsize=10)

        if ego_key:
            ax.text(0.5, -0.2, f'EGO: {ego_key}', transform=ax.transAxes, ha='center', fontsize=7, color='gray')

    plt.suptitle('NeuraGEM vs EGO: Tcid Accuracy (Best EGO Config per Curriculum)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(export_dir, 'ego_vs_neuragem_bars.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {fig_path}')

    # --- Panel C: Context encoding comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, curriculum in enumerate(curricula):
        ax = axes[ci]
        ego_key = best_configs[curriculum]
        ego_data = summary[curriculum].get(ego_key, {})

        x = np.arange(2)
        width = 0.35

        # EGO context encoding
        if ego_data:
            ax.bar(x - width / 2, [ego_data['beta_tcid'], ego_data['beta_trnd']],
                   width, label='EGO', color=['#4CAF50', '#FF9800'], alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(['Tcid β', 'Trnd β'])
        ax.set_title(f'{curricula_labels[ci]}')
        ax.legend(fontsize=8)

    plt.suptitle('EGO Context Encoding (|β| from linear regression)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(export_dir, 'ego_context_encoding.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {fig_path}')

    # --- Panel D: Per-ctx accuracy breakdown for best config ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ci, curriculum in enumerate(curricula):
        ax = axes[ci]
        ego_key = best_configs[curriculum]
        ego_data = summary[curriculum].get(ego_key, {})

        if ego_data:
            x_labels = ['ctx 0\n(→5)', 'ctx 1\n(→6)', 'overall']
            vals = [ego_data['acc_ctx0'], ego_data['acc_ctx1'], ego_data['acc_mean']]
            bars = ax.bar(x_labels, vals, color=['#4CAF50', '#FF9800', '#2196F3'])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1.1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{v:.3f}', ha='center', fontsize=9)
        ax.set_title(f'{curricula_labels[ci]} ({ego_key})')

    plt.suptitle('EGO Per-Context Tcid Accuracy (Best Config)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(export_dir, 'ego_per_context_accuracy.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {fig_path}')

    return best_configs


def print_summary_table(summary, best_configs):
    """Print a comprehensive results table."""
    curricula = ['blocked', 'interleaved', 'interleaved_blocked']
    hidden_context_keys = [(16, 2), (16, 4), (16, 8), (64, 2), (64, 4), (64, 8)]

    print()
    print('=' * 100)
    print('EGO BEUKERS SWEEP — Tcid Accuracy Summary (mean ± SEM across 20 seeds)')
    print('=' * 100)

    # Header
    header = f'{"Curriculum":25s}'
    for hd, cd in hidden_context_keys:
        header += f' | {"h=" + str(hd) + " c=" + str(cd):>16s}'
    header += ' | {"Best":>12s}'
    print(header)
    print('-' * len(header))

    for curriculum in curricula:
        row = f'{curriculum:25s}'
        best_acc = 0
        best_label = ''
        for hd, cd in hidden_context_keys:
            key = f'h{hd}_c{cd}'
            if key in summary.get(curriculum, {}):
                d = summary[curriculum][key]
                row += f' | {d["acc_mean"]:6.3f}±{d["acc_sem"]:.3f}'
                if d['acc_mean'] > best_acc:
                    best_acc = d['acc_mean']
                    best_label = key
            else:
                row += f' | {"N/A":>16s}'
        best_str = f'{best_label} ({best_acc:.3f})' if best_label else 'N/A'
        row += f' | {best_str:>12s}'
        print(row)

    print()
    print('Best config per curriculum:')
    for c in curricula:
        print(f'  {c:25s} → {best_configs.get(c, "N/A")}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    export_dir = './exports/ego_beukers/analysis'
    os.makedirs(export_dir, exist_ok=True)

    print('Aggregating results...')
    summary = aggregate_sweep()
    print(f'Aggregated {sum(len(summary[c]) for c in summary)} configs across 3 curricula')

    best_configs = plot_main_summary(summary, export_dir)
    print_summary_table(summary, best_configs)

    # Save summary dict
    with open(os.path.join(export_dir, 'ego_sweep_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    print(f'\nSaved summary to {export_dir}/ego_sweep_summary.pkl')
    print('Done!')
