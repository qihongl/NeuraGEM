# NeuraGEM Project Memory

## Environment
- **Python**: Use system miniforge3 Python (`/Users/qlu/miniforge3/bin/python3`) for experiments. Managed Python has code-signing issues with native libs (PyTorch, charset_normalizer).
- **Key deps**: PyTorch 2.12.0, NumPy 2.4.4, Matplotlib 3.10.8 (all system-installed)

## Beukers Task Configuration
- **seq_len must be 6** (the paper's value). Default `SeqLearnConfig` uses 18, which makes the LSTM alone sufficient to distinguish contexts.
- **predict_first_frame=True** by default — predictions are for same-timestep inputs (zero-frame trick). Accuracy computation must use `preds[idx+1]` for Tcid.
- **Logger attributes**: `latent_values` (not `z_values`), `phases` (list of `(name, start_idx)` tuples, not `phase_markers`)
