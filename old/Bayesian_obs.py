#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import TaskDataset
from configs import ContextualSwitchingTaskConfig
from torch.utils.data import DataLoader
from scipy.stats import sem
from functions_adaptation_dynamics_analysis import *
import plot_style
cs = plot_style.Color_scheme()

# --- Step 1. Create dataset ---
config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
config.no_of_blocks = 400
config.default_std = 0.4
dataset = TaskDataset(no_of_blocks=config.no_of_blocks, config=config)

# Use the full data sequence for simulation
data_sequence = dataset.data_sequence
latent_sequence = dataset.latent_sequence

# --- Step 2. Define the Bayesian observer ---
latent_values = np.array(config.training_data_means)  # e.g., [0.2, 0.8]
prior = np.array([0.5, 0.5])
hazard_rate = 0.05  
# Note: sigma here is set as variance (sigma^2) as in your code.
sigma = (config.default_std) * (config.default_std) * np.sqrt(2*np.pi)  # Adjusted for variance

def normal_pdf(y, mu, sigma):
    """Compute the probability density of y given a Normal(mu, sigma^2) distribution."""
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((y - mu) / sigma) ** 2)

posterior = prior.copy()
bayesian_predictions = []  # To store Bayesian predictions (predictive mean)
posterior_history = []     # To record posterior over time

for y in data_sequence:
    # Bayesian prediction: weighted sum of latent values.
    pred = np.sum(posterior * latent_values)
    bayesian_predictions.append(pred)
    
    likelihoods = np.array([normal_pdf(y, mu, sigma) for mu in latent_values])
    updated_posterior = (1 - hazard_rate) * posterior * likelihoods + hazard_rate * prior * likelihoods
    posterior = updated_posterior / np.sum(updated_posterior)  # Normalize
    posterior_history.append(posterior.copy())

bayesian_predictions  = np.array(bayesian_predictions)
# --- Step 3. Define the Naive model ---
# Naive model: prediction at time t is the average of the two previous observations.
naive_predictions = []
# We start predictions from t=2, so the first two time points have no prediction.
for t in range(2, len(data_sequence)):
    naive_pred = (data_sequence[t - 1] + data_sequence[t - 2]) / 2
    naive_predictions.append(naive_pred)

# --- Step 4. Visualize the results ---
# Bayesian observer plot
plt.figure(figsize=(3, 2))
plt.plot(data_sequence, label='Observed Data', marker='o', markersize=3, linestyle='-', alpha=0.7)
plt.plot(bayesian_predictions, label='Bayesian Prediction', linestyle='--', linewidth=2)
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title('Bayesian Observer Predictions')
plt.xlim([0, len(data_sequence)//4])
plt.legend()
plt.show()

# Naive model plot
# Adjust x-axis to start from t=2
plt.figure(figsize=(3, 2))
plt.plot(data_sequence, label='Observed Data', marker='o', markersize=3, linestyle='-', alpha=0.7)
plt.plot(range(2, len(data_sequence)), naive_predictions, label='Naive Prediction', linestyle='--', linewidth=2)
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title('Naive Model Predictions (Average of Last Two Observations)')
plt.xlim([0, len(data_sequence)//4])
plt.legend()
plt.show()

# Optional: Plot evolution of the posterior over time for Bayesian model
posterior_history = np.array(posterior_history)  # shape: (n_steps, 2)
plt.figure(figsize=(12, 3))
plt.plot(posterior_history[:, 0], label=f'Posterior for latent {latent_values[0]}')
plt.plot(posterior_history[:, 1], label=f'Posterior for latent {latent_values[1]}')
plt.xlabel('Time step')
plt.ylabel('Posterior probability')
plt.title('Evolution of Posterior Beliefs')
plt.xlim([0, len(data_sequence)//4])
plt.legend()
plt.show()

# --- Step 5. Compute error metrics ---
# We use errors computed from observations ("observations" target) for both models.
error_target = np.array(data_sequence)

# Bayesian errors: computed from full sequence predictions.
bayesian_mse_errors = (np.array(bayesian_predictions) - error_target) ** 2
bayesian_overall_mse = np.mean(bayesian_mse_errors)
print("Bayesian Overall MSE error:", bayesian_overall_mse)

# Naive model errors: note predictions start at t=2, so compare with data_sequence[2:].
naive_mse_errors = (np.array(naive_predictions) - error_target[2:]) ** 2
naive_overall_mse = np.mean(naive_mse_errors)
print("Naive Overall MSE error:", naive_overall_mse)

# Find switch points (where the latent state changes)
latent_sequence = np.array(latent_sequence)
switch_indices = np.where(np.diff(latent_sequence) != 0)[0] + 1  # +1 to mark the start of the new block
# print("Switch point indices:", switch_indices)

# Set pre and post windows.
pre_window = 3
post_window = 23
window_length = pre_window + post_window + 1

def compute_error_strips(mse_errors, offset=0, enforce_block_size_min_post_window=False):
    """
    Computes error strips given mse_errors.
    'offset' is used if mse_errors array does not start at t=0.
    """
    strips = []
    for i, idx in enumerate(switch_indices[:-1]):
        adjusted_idx = idx - offset
        # If enforcing minimum post_window, skip if the current block is too short
        if enforce_block_size_min_post_window and (switch_indices[i+1] - idx) < post_window:
            continue
        if adjusted_idx - pre_window >= 0 and adjusted_idx + post_window < len(mse_errors):
            strip = mse_errors[adjusted_idx - pre_window: adjusted_idx + post_window + 1]
            strips.append(strip)
    return np.array(strips)

# For Bayesian model (predictions start at t=0)
bayesian_error_strips = compute_error_strips(bayesian_mse_errors, offset=0, enforce_block_size_min_post_window=True)
bayesian_mean_strip = np.mean(bayesian_error_strips, axis=0)
bayesian_sem_strip = sem(bayesian_error_strips, axis=0)

# For Naive model (predictions start at t=2)
naive_error_strips = compute_error_strips(naive_mse_errors, offset=2)
naive_mean_strip = np.mean(naive_error_strips, axis=0)
naive_sem_strip = sem(naive_error_strips, axis=0)

time_axis = np.arange(-pre_window, post_window + 1)

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.plot(time_axis, bayesian_mean_strip, label='Mean MSE Error (Bayesian)')
ax.fill_between(time_axis, bayesian_mean_strip - bayesian_sem_strip, bayesian_mean_strip + bayesian_sem_strip,
                color='blue', alpha=0.4, label='Bayesian SEM')
ax.plot(time_axis, naive_mean_strip, label='Mean MSE Error (Naive)')
ax.fill_between(time_axis, naive_mean_strip - naive_sem_strip, naive_mean_strip + naive_sem_strip,
                color='orange', alpha=0.4, label='Naive SEM')
ax.set_xlabel('Time relative to switch point')
ax.set_ylabel('MSE error')
ax.set_title('Error Strips around Switch Points')
ax.legend()

# --- Step 6. Output error strips as text arrays ---
print("time_axis =", "[" + ", ".join(map(str, time_axis)) + "]")
print("bayesian_mean_strip =", "[" + ", ".join(map(str, bayesian_mean_strip)) + "]")
print("bayesian_sem_strip =", "[" + ", ".join(map(str, bayesian_sem_strip)) + "]")
print("naive_mean_strip =", "[" + ", ".join(map(str, naive_mean_strip)) + "]")
print("naive_sem_strip =", "[" + ", ".join(map(str, naive_sem_strip)) + "]")


#%%
from sklearn.linear_model import LinearRegression

# ----------------------------
# Step 2. Prepare Regression Data
# ----------------------------

def prepare_regression_data(observations, predictions, regression_window=50,
                            target_is_learning_rate=True, predictors_source='observations'):
    """
    Prepare the design matrix X and target y for regression.
    
    Parameters:
        observations: array-like, observed data (e.g., data_sequence)
        predictions: array-like, model predictions (e.g., Bayesian predictions)
        regression_window: int, number of past timesteps to use as predictors.
        target_is_learning_rate: bool, if True target = prediction difference (learning rate),
                                 else target = current prediction.
        predictors_source: str, either 'observations' or 'prediction_errors'.
                         If 'prediction_errors', uses (observations - predictions) as predictors.
    
    Returns:
        X: 2D numpy array of shape (n_samples, regression_window)
        y: 1D numpy array of length n_samples
    """
    n = len(predictions)
    X = []
    y = []
    for t in range(regression_window, n):
        # Build predictor vector from past regression_window timesteps
        if predictors_source == 'observations':
            x_vec = observations[t-regression_window:t]
        elif predictors_source == 'prediction_errors':
            errors = observations[t-regression_window:t] - predictions[t-regression_window:t]
            x_vec = errors
        else:
            raise ValueError("predictors_source must be 'observations' or 'prediction_errors'")
        
        # Define target: either the learning rate (prediction[t] - prediction[t-1])
        # or the current prediction itself.
        if target_is_learning_rate:
            target_value = predictions[t] - predictions[t-1]
        else:
            target_value = predictions[t]
        
        X.append(x_vec)
        y.append(target_value)
    return np.array(X), np.array(y)

# Toggle parameters for regression
regression_window = 5
target_is_learning_rate = True  # Toggle: True -> learning rate; False -> model's current prediction
# predictors_source = 'observations'  # Toggle: 'observations' or 'prediction_errors'
predictors_source = 'prediction_errors'  # Toggle: 'observations' or 'prediction_errors'

# Prepare design matrix and target vector
X, y = prepare_regression_data(data_sequence, np.array(bayesian_predictions),
                               regression_window=regression_window,
                               target_is_learning_rate=target_is_learning_rate,
                               predictors_source=predictors_source)

# ----------------------------
# Step 3. Fit the Regression Model
# ----------------------------
reg_model = LinearRegression(fit_intercept=True)
reg_model.fit(X, y)
y_pred = reg_model.predict(X)

# ----------------------------
# Step 4. Visualization
# ----------------------------

# 4a. Plot the regression coefficients (alpha values) vs. time lag.
def visualize_coefficients(model, regression_window):
    coefficients = model.coef_
    lags = np.arange(regression_window)
    plt.figure(figsize=(10, 4))
    markerline, stemlines, baseline = plt.stem(lags, coefficients, basefmt=" ")
    plt.xlabel("Time Lag (steps into the past)")
    plt.ylabel("Regression Coefficient (α)")
    plt.title("Regression Coefficients vs. Past Time Steps")
    plt.show()

visualize_coefficients(reg_model, regression_window)

# 4b. Plot actual target vs. fitted target values for diagnostic.
plt.figure(figsize=(10, 3))
plt.plot(y, label="Actual Target", alpha=0.7)
plt.plot(y_pred, label="Fitted Target", linestyle="--", alpha=0.7)
plt.xlabel("Sample Index (time)")
if target_is_learning_rate:
    plt.ylabel("Learning Rate (Δ prediction)")
    plt.title("Actual vs. Fitted Learning Rates")
else:
    plt.ylabel("Model Prediction")
    plt.title("Actual vs. Fitted Model Predictions")
plt.legend()
plt.show()

# 4c. Optionally, scatter-plot of actual vs. predicted targets.
plt.figure(figsize=(2, 2))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Target")
plt.ylabel("Fitted Target")
plt.title("Scatter Plot of Actual vs. Fitted Targets")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.show()
#%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%
######################################################
################# Nassar style learning rates ########
######################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def extract_switch_segments(inputs, preds, latents, pre_window=5, post_window=30):
    """
    Identify block switches (points where latent changes) and extract segments of data centered around these switches.
    
    Parameters:
        inputs: array-like, observed inputs
        preds: array-like, model predictions
        latents: array-like, latent values
        pre_window: int, number of timesteps before the switch to include
        post_window: int, number of timesteps after the switch to include
    
    Returns:
        segments: list of dicts, each containing:
            'inputs': segment of inputs (length = pre_window + post_window)
            'preds': corresponding predictions
            'latents': corresponding latent values
            'switch_idx': index in the segment corresponding to the switch (should equal pre_window)
            'global_index': index in the original sequence where the switch occurred.
    """
    segments = []
    # Find indices where latent value changes: latent[t] != latent[t-1]
    for t in range(1, len(latents)):
        if latents[t] != latents[t-1]:
            # Only include switches where we have enough pre and post data.
            if t - pre_window >= 0 and t + post_window <= len(latents):
                segment = {
                    'inputs': inputs[t-pre_window:t+post_window],
                    'preds': preds[t-pre_window:t+post_window],
                    'latents': latents[t-pre_window:t+post_window],
                    'switch_idx': pre_window,  # The switch occurs at index `pre_window` in the segment
                    'global_index': t
                }
                segments.append(segment)
    return segments

def extract_period_data(segment, period_bounds):
    """
    Extract data from a segment for a given time period.
    
    Parameters:
        segment: dict, one segment as returned by extract_switch_segments
        period_bounds: tuple (start, end) relative to the switch time (e.g., (-5, 0), (0, 5), etc.)
            Note: start and end are relative indices. For example, (-5, 0) means last 5 timesteps before the switch.
    
    Returns:
        period_inputs: array of inputs for that period
        period_preds: array of predictions for that period
    """
    switch_idx = segment['switch_idx']
    # Convert relative bounds to indices in the segment
    start = switch_idx + period_bounds[0]
    end = switch_idx + period_bounds[1]
    # Check bounds
    if start < 0 or end > len(segment['inputs']):
        return None, None
    return segment['inputs'][start:end], segment['preds'][start:end]

def compute_deltas(values, abs=True):
    """
    Compute the first differences (delta) of a 1D array.
    Returns an array of length len(values)-1.
    """
    return np.abs(np.diff(values)) if abs else np.diff(values)

def estimate_slope(x, y):
    """
    Estimate the slope of a linear regression predicting y from x.
    
    Parameters:
        x: 1D numpy array (independent variable)
        y: 1D numpy array (dependent variable)
    
    Returns:
        slope: float, regression coefficient
    """
    # Reshape for sklearn
    x = x.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(x, y)
    return reg.coef_[0]

# Define periods of interest relative to the switch.
# Each tuple is (start, end) relative to the switch index.
# Example periods:
#   Pre-switch: (-5, 0)
#   Immediately post-switch: (0, 5)
#   Then successive windows: (5,10), (10,15), (15,20), (20,30)
periods = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
period_labels = ['Pre-switch', '0-5', '5-10', '10-15', '15-20', '20-30']

def analyze_learning_slopes(segments, periods, 
                            use_prediction_deltas=False):
    """
    For each defined period, compute the slope from the regression of the delta in model predictions
    on the delta in inputs. Optionally, one can use the changes in predictions directly or use prediction errors.
    
    Parameters:
        segments: list of segments (dicts) extracted by extract_switch_segments.
        periods: list of tuples defining the time periods relative to the switch.
        use_prediction_deltas: bool. If True, regress changes in predictions against changes in inputs.
                               If False, you could consider an alternative (e.g., using prediction errors)
                               but here we default to the deltas.
    
    Returns:
        period_slopes: dict with period labels as keys and lists of slopes (one per segment) as values.
    """
    period_slopes = {label: [] for label in period_labels}
    
    # Loop over each segment and each period
    for segment in segments:
        for (bounds, label) in zip(periods, period_labels):
            period_inputs, period_preds = extract_period_data(segment, bounds)
            # Skip if data not available
            if period_inputs is None or len(period_inputs) < 2:
                continue
            
            # Compute deltas: differences from one time step to the next.
            # For inputs and for predictions.
            input_deltas = compute_deltas(period_inputs)
            pred_deltas = compute_deltas(period_preds)
            
            # Optionally, if using prediction errors, one would compute errors here:
            # For now we use the prediction deltas directly.
            slope = estimate_slope(input_deltas, pred_deltas)
            period_slopes[label].append(slope)
    
    return period_slopes

def plot_learning_slopes(period_slopes):
    """
    Plot the average learning slope with error bars for each period.
    """
    labels = []
    means = []
    stds = []
    
    for label in period_labels:
        slopes = period_slopes[label]
        if len(slopes) > 0:
            labels.append(label)
            means.append(np.mean(slopes))
            stds.append(np.std(slopes))
    
    ax = plt.figure(figsize=(4, 3)).gca()
    ax.errorbar(labels, means, yerr=stds, fmt='o', capsize=5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Time Period Relative to Switch")
    ax.set_ylabel("Estimated Learning Slope")

# === Example Usage ===

# Assuming you have arrays of data from a model:
# Here, we use the Bayesian observer outputs as an example.
# (data_sequence, predictions, and latent_sequence should be numpy arrays)
# They might be generated as in our previous Bayesian observer simulation.

# For example:
# data_sequence: observed inputs (from dataset)
# predictions: Bayesian predictions computed earlier (as a numpy array)
# latent_sequence: corresponding latent values (from dataset)

# (These arrays are assumed to be available from your previous code.)
# For demonstration, let's assume they are already defined:
#   data_sequence, predictions, latent_sequence

# Convert latent_sequence to numpy if needed:
latent_sequence = np.array(latent_sequence)

# Extract segments centered on block switches.
segments = extract_switch_segments(data_sequence, np.array(bayesian_predictions), latent_sequence,
                                   pre_window=5, post_window=30)

# Analyze learning slopes in each period.
period_slopes = analyze_learning_slopes(segments, periods, use_prediction_deltas=True)

# Plot the estimated slopes.
plot_learning_slopes(period_slopes)

# %%
# rolling window estimate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Example Usage ===
# Assume you already have:
#   data_sequence, predictions, and latent_sequence from your model (e.g., Bayesian observer)
# And that they are numpy arrays.
# Also assume extract_switch_segments is defined as in the previous code.

# For demonstration, we use the arrays from the previous example:
# (data_sequence, predictions, latent_sequence)
segments = extract_switch_segments(data_sequence, bayesian_predictions, latent_sequence,
                                   pre_window=5, post_window=30)

# Now compute the rolling learning rate estimates across segments.
window_size = 5
unique_times, avg_slopes, std_slopes = compute_rolling_lr_estimates(segments,)
fig, axes = plt.subplots(1,1, figsize=cs.panel_small_size)
ax =axes
fig = plot_rolling_learning_rate(unique_times, avg_slopes, std_slopes, ax=ax)
plt.show()
# %%


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from plot_style import Color_scheme
from functions_adaptation_dynamics_analysis import *
from functions_and_utils import *

# Initialize color scheme
cs = Color_scheme()

# Extract segments
segments = extract_switch_segments(data_sequence, bayesian_predictions, latent_sequence,
                                   pre_window=5, post_window=30)

# Extract inputs, predictions, and compute errors
inputs = [seg['inputs'] for seg in segments]
preds = [seg['preds'] for seg in segments]
errors = [inp - pred for inp, pred in zip(inputs, preds)]

# Convert to numpy arrays for easier manipulation
inputs_np = np.array(inputs)
preds_np = np.array(preds)
errors_np = np.array(errors)

# Plot inputs, predictions, and errors in subplots
fig, axes = plt.subplots(3, 1, figsize=(3, 5))

# Plot inputs
axes[0].plot(inputs_np.T)
axes[0].set_title('Inputs')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Input Value')

# Plot predictions
axes[1].plot(preds_np.T)
axes[1].set_title('Predictions')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Prediction Value')

# Plot errors
axes[2].plot(errors_np.T)
axes[2].set_title('Errors (Inputs - Predictions)')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Error Value')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Compute rolling learning rate estimates across segments
window_size = 5
unique_times, avg_slopes, std_slopes = compute_rolling_lr_estimates(segments)

# Plot rolling learning rate
fig, ax = plt.subplots(1, 1, figsize=cs.panel_small_size)
fig = plot_rolling_learning_rate(unique_times, avg_slopes, std_slopes, ax=ax)

# Explore data container
explore_data_container(segments)

# Compute stable learning rate statistics for the segments
def compute_stable_lr_stats(segments, block_size=25, pre_window=5, post_window=30):
    """
    Computes the mean and SEM of the stable learning rate for the given segments.
    
    Parameters:
        segments: list of segments (dicts) extracted by extract_switch_segments.
        block_size: int, size of the block to consider for stable learning rate.
        pre_window: int, number of timesteps before the switch to include.
        post_window: int, number of timesteps after the switch to include.
    
    Returns:
        mean_lr: float, mean stable learning rate.
        sem_lr: float, standard error of the mean of the stable learning rate.
    """
    unique_times, avg_slopes, std_slopes = compute_rolling_lr_estimates(segments)
    mean_pre_switch_lr, std_pre_switch_lr = extract_pre_switch_lr(avg_slopes, block_size, k=5)
    
    lr_values = np.array(mean_pre_switch_lr)
    mean_lr = np.nanmean(lr_values)
    valid_n = np.sum(~np.isnan(lr_values))
    sem_lr = np.nan if valid_n <= 1 else np.nanstd(lr_values) / np.sqrt(valid_n)
    
    return mean_lr, sem_lr

# Example usage
mean_lr, sem_lr = compute_stable_lr_stats(segments, block_size = 16) # 16 is the min block size
print("Mean Stable Learning Rate:", mean_lr)
print("SEM of Stable Learning Rate:", sem_lr)

ax.axhline(mean_lr, color='black', linestyle='--', label='Mean Stable Learning Rate')


#%%import numpy as np
import torch
from datasets import TaskDataset
from configs import ContextualSwitchingTaskConfig
from scipy.stats import sem

# Define standard deviation values to test
std_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Function to compute stable learning rate statistics
def compute_stable_lr(segments):
    """
    Computes the mean and SEM of the stable learning rate for the given segments.
    """
    unique_times, avg_slopes, _ = compute_rolling_lr_estimates(segments)
    # mean_pre_switch_lr, _ = extract_pre_switch_lr(avg_slopes, block_size=16, k=5)  # Using minimum block size
    # this function assumes several blocks
    def extract_pre_switch_lr_asymptote(avg_slopes, min_block_size, k=5):
        pre_switch_slopes = []
        pre_window = 5
        pre_switch_slopes.extend(avg_slopes[pre_window + min_block_size - k: pre_window + min_block_size])

        return np.mean(pre_switch_slopes), np.std(pre_switch_slopes)
    
    mean_pre_switch_lr, _ = extract_pre_switch_lr_asymptote(avg_slopes, min_block_size=16, k=5)  # Using minimum block size
    lr_values = np.array(mean_pre_switch_lr)

    mean_lr = np.nanmean(lr_values)
    valid_n = np.sum(~np.isnan(lr_values))
    sem_lr = np.nan if valid_n <= 1 else np.nanstd(lr_values) / np.sqrt(valid_n)
    
    return mean_lr, sem_lr

# Function to compute mean absolute prediction error
def compute_mean_abs_error(predictions, data_sequence):
    return np.mean(np.abs(np.array(predictions) - np.array(data_sequence[:len(predictions)])))

# Storage for results
results = []
bayesian_mean_mses = {}
bayesian_mean_mses_sem = {}
naive_mean_mses = {}
naive_mean_mses_sem = {}

for std in std_values:
    # Set up the dataset configuration
    config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
    config.no_of_blocks = 400
    config.default_std = std
    dataset = TaskDataset(no_of_blocks=config.no_of_blocks, config=config)
    
    data_sequence = dataset.data_sequence
    latent_sequence = np.array(dataset.latent_sequence)
    
    # Bayesian model predictions
    latent_values = np.array(config.training_data_means)  # [0.2, 0.8]
    prior = np.array([0.5, 0.5])
    hazard_rate = 0.05
    training_std = 0.4
    # std_to_use = std
    std_to_use = training_std
    sigma = std_to_use ** 2 * np.sqrt(2 * np.pi)
    
    def normal_pdf(y, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((y - mu) / sigma) ** 2)
    
    posterior = prior.copy()
    bayesian_predictions = []
    for y in data_sequence:
        pred = np.sum(posterior * latent_values)
        bayesian_predictions.append(pred)
        likelihoods = np.array([normal_pdf(y, mu, sigma) for mu in latent_values])
        updated_posterior = (1 - hazard_rate) * posterior * likelihoods + hazard_rate * prior * likelihoods
        posterior = updated_posterior / np.sum(updated_posterior)
    
    # Naive model predictions
    naive_predictions = [(data_sequence[t - 1] + data_sequence[t - 2]) / 2 for t in range(2, len(data_sequence))]
    
    # Extract switch segments
    bayesian_segments = extract_switch_segments(data_sequence, np.array(bayesian_predictions), latent_sequence, pre_window=5, post_window=30)
    naive_segments = extract_switch_segments(data_sequence[2:], np.array(naive_predictions), latent_sequence[2:], pre_window=5, post_window=30)
    
    # Compute stable learning rate statistics
    bayesian_mean_lr, bayesian_sem_lr = compute_stable_lr(bayesian_segments)
    naive_mean_lr, naive_sem_lr = compute_stable_lr(naive_segments)
    from functions_and_utils_2 import calculate_error 
    # Create a temporary logger object. For consistency with RNN exp, will create this logger class to be able to use the same functions (Calculate error)
    class TempLogger:
        pass

    temp_logger = TempLogger()
    temp_logger.inputs = [np.array(data_sequence).reshape(-1, 1)]
    temp_logger.predicted_outputs = [np.array(bayesian_predictions).reshape(-1, 1)]
    temp_logger.llcids = [np.array(latent_sequence).reshape(-1, 1)]
    temp_logger.hlcids = [np.zeros_like(latent_sequence).reshape(-1, 1)]  # Assuming no higher-level latents
    temp_logger.phases = None  # Assuming no specific phases

    error_type = 'abs_from_mean'
    # Compute errors using calculate_error
    bayesian_error_array, bayesian_peri_switch_errors = calculate_error(temp_logger, error_type=error_type)
    bayesian_mae = np.mean(bayesian_error_array)

    # Compute peri-switch errors
    total_errors = [np.nanmean(strip) for strip in bayesian_peri_switch_errors]
    if len(total_errors) > 0:
        mean_total_error = np.nanmean(total_errors)
        bayesian_mean_mses[std]= mean_total_error
        bayesian_mean_mses_sem[std] = np.std(total_errors) / np.sqrt(len(total_errors))



    bayesian_segments = extract_switch_centered_segments(temp_logger, pre_window=5, 
                            post_window=30, phases_to_include=None)
    # Update logger for naive model
    temp_logger.inputs = [np.array(data_sequence[2:]).reshape(-1, 1)]
    temp_logger.predicted_outputs = [np.array(naive_predictions).reshape(-1, 1)]
    temp_logger.llcids = [np.array(latent_sequence[2:]).reshape(-1, 1)]

    # Compute errors for naive model
    naive_error_array, naive_peri_switch_errors = calculate_error(temp_logger, error_type='abs')
    naive_mae = np.mean(naive_error_array)

    # Compute peri-switch errors
    total_errors = [np.nanmean(strip) for strip in naive_peri_switch_errors]
    if len(total_errors) > 0:
        mean_total_error = np.nanmean(total_errors)
        naive_mean_mses[std]= mean_total_error
        naive_mean_mses_sem[std] = np.std(total_errors) / np.sqrt(len(total_errors))

    # Normalize learning rates
    # bayesian_mean_lr = bayesian_mae
    # naive_mean_lr = naive_mae
    
    # Store results
    results.append((std, bayesian_mean_lr, bayesian_sem_lr, naive_mean_lr, naive_sem_lr))

fig, ax = plt.subplots(1,1, figsize=cs.panel_small_size)
ax.plot(std_values, list(bayesian_mean_mses.values()), label='Bayesian Observer', color=cs.bayesian, marker='o', linestyle='-', linewidth=cs.linewidth)
ax.fill_between(std_values, 
                np.array(list(bayesian_mean_mses.values())) - np.array(list(bayesian_mean_mses_sem.values())), 
                np.array(list(bayesian_mean_mses.values())) + np.array(list(bayesian_mean_mses_sem.values())), 
                color=cs.bayesian, alpha=cs.alpha_shaded_regions)

# ax.plot(std_values, list(naive_mean_mses.values()), label='Naive Model', color=cs.naive, marker='o', linestyle='-', linewidth=cs.linewidth)
# ax.fill_between(std_values, 
#                 np.array(list(naive_mean_mses.values())) - np.array(list(naive_mean_mses_sem.values())), 
#                 np.array(list(naive_mean_mses.values())) + np.array(list(naive_mean_mses_sem.values())), 
#                 color=cs.naive, alpha=cs.alpha_shaded_regions)

# Convert results to numpy array for easy access
results = np.array(results)
#%%
# Print results as arrays that I can copy and paste elsewhere, spearately for bayesian and naive
print("#Bayesian results:")
# print("std_values =", "[" + ", ".join(map(str, results[:, 0])) + "]")
print("bayesian_mean_lr =", "[" + ", ".join(map(str, results[:, 1])) + "]")
# print("bayesian_sem_lr =", "[" + ", ".join(map(str, results[:, 2])) + "]")

print("#Naive results:")
# print("std_values =", "[" + ", ".join(map(str, results[:, 0])) + "]")
print("naive_mean_lr =", "[" + ", ".join(map(str, results[:, 3])) + "]")
# print("naive_sem_lr =", "[" + ", ".join(map(str, results[:, 4])) + "]")

# Plot results
plt.figure(figsize=(4, 3))
ax = plt.gca()
ax.errorbar(results[:, 0], results[:, 1], yerr=results[:, 2], label='Bayesian Observer', color='blue', marker='o')
ax.errorbar(results[:, 0], results[:, 3], yerr=results[:, 4], label='Naive Model', color='orange', marker='o')
ax.plot([1, 0], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Learning Rate')
