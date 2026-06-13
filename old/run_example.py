# %% 
''' 
a scratch script to train models and experiment. Trains a model and plots the behavior, latent dynamics, and gradients.
Notebook friendly, but can be run as a script. You can specify which config to run, and which panels to plot in the final figure.

'''

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
import plot_style
plot_style.set_plot_style()
from functions_and_utils import *
from configs import *
from train_and_infer_functions import *


## Choose the config to run.
config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
# config = seq_learnConfig(experiment_to_run='few_long_blocks')    

## Set some config parameters 
config.default_std = 0.1 # std lower than paper main experiments (0.3), just for visualization and clarity.
config.log_weights = False

# config.LU_lr = 0.8 
# config.l2_loss = 0.0003 #

# for RNN_short:
# config.LU_lr = 0.0

# for RNN_long:
# config.LU_lr = 0.0
# config.seq_len = 50

print('Running the model seed: ', config.env_seed)
config.save_model = False
config.load_saved_model = False
logger, model, config, figs = train_model(config, seed=config.env_seed, 
                    save_models=False, load_models=False,)

if config.dataset_name == 'seq_learn':
    fig = plot_seq_learn_behavior_and_overall_corrects(logger_train, config, include_gradients=True)
else:
    panel_order = ['task_illustration_and_hierarchies', 'behavior',  'latent', 'latent_2d', 'gradients', 'weights_grad_norm',]# 'loss'] # 'task_illustration_and_hierarchies',
    fig = plot_logger_panels(logger, config, panel_order,  x2=None, annotate_phases='behavior')

# final figure
# specify which panels to plot
panel_order = ['behavior',  'latent_2d', 'gradients', 'weights_grad_norm', 'latent_effective_lr']# 'loss'] # 'task_illustration_and_hierarchies',
panel_order = ['behavior',  'latent_2d', 'gradients',]# 'loss'] # 'task_illustration_and_hierarchies',
fig = plot_logger_panels(logger, config, panel_order,
            x2=260, dpi=300, subplot_height=.8, width = 2.5, rasterize=True)

# plot the whole data.
fig = plot_logger_panels(logger, config, panel_order, annotate_phases='behavior',
            x2=None, dpi=300, subplot_height=.8, rasterize=True)

print(f'figure saved to: {config.export_path}{config.dataset_name}_behavior_rasterized.pdf')

