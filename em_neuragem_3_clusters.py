# %% 
''' Qualitative comparison to EM demo data.
Here are the experiments to describe the z space and what is learned. 
'''

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
import torch
from torch.utils.data import Dataset, DataLoader   
import torch.nn as nn

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import copy
from sklearn.decomposition import PCA
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from functions_and_utils import *
# from acc_related_functions import *
from datasets import *
from configs import *
from plot_style import *
cs = Color_scheme()
set_plot_style()


from train_and_infer_functions import *


# %%
config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
config.use_EM_demo_data = True
config.dataset_name = 'contextual_switching_task_2D'
config.input_size = 2
config.output_size = 2
config.latent_dims = [2]
config.env_seed = 3 # 3  vs.  6 or 1 # Note results may vary from run to run. PCA is also randomly initialized. 
reference_epoch = -1 # -4  vs.  -1
config.eval_z_space_interval = 300
config.blocked_phase_length = 2000

folder = f'{config.export_path}{config.dataset_name}/z_space_evals/'
files = glob.glob(os.path.join(folder, '*'))
for f in files:
    os.remove(f)

logger, model, config, figs = train_model(config, seed=config.env_seed, 
                    save_models=False, load_models=False, run_test_phase=False,)

panel_order = ['behavior',  'latent_2d', 'gradients',]# 'loss'] # 'task_illustration_and_hierarchies',
plot_logger_panels(logger, config, panel_order,  x2=None, annotate_phases='behavior')

#%%
folder = os.path.join(config.export_path, config.dataset_name, 'z_space_evals')
files = sorted(glob.glob(os.path.join(folder, 'zlogger_*.npy')))

zloggers = {}
for filepath in files:
    epoch = int(os.path.basename(filepath).split('_')[1].split('.')[0])
    zloggers[epoch] = np.load(filepath, allow_pickle=True)

print(f"Loaded {len(zloggers)} zlogger files: {sorted(zloggers.keys())}")

sorted_epochs = sorted(zloggers.keys())
#%%
# PCA is performed on the latent from the end of training.
epoch_of_reference = sorted_epochs[reference_epoch]
zlogger = zloggers[epoch_of_reference].item()
latent_values = np.concatenate(zlogger.latent_values, axis=0)
latent_values = latent_values.reshape(-1, latent_values.shape[-1])


n_components =1
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
pca.fit(latent_values)

# Create a row of subplots for the chosen epochs
if config.env_seed ==3:
    chosen_epochs = [0, 600, 900, 1200, 1500, ]
elif config.env_seed == 1:
    chosen_epochs = [0, 300, 600, 900, 1500, ]
else:
    chosen_epochs = sorted_epochs
fig, axs = plt.subplots(1, len(chosen_epochs), figsize=(1.2*len(chosen_epochs), 1.4), sharex=True, sharey=True)

for ax, epoch in zip(axs, chosen_epochs):
    if epoch in zloggers:
        logger_epoch = zloggers[epoch].item()

        # Concatenate inputs and latent values (reshaping to maintain dimensions)
        inp = np.concatenate(logger_epoch.inputs, axis=0)
        inp = np.reshape(inp, (-1, inp.shape[-1]))
        latent = np.concatenate(logger_epoch.latent_values, axis=0)
        latent = latent.reshape(-1, latent.shape[-1])
        
        # Transform latent values using existing PCA (from the last evaluation)
        latent_pca = pca.transform(latent)
        latent_pca = latent_pca.clip(-1,1)
        scatter_obj = ax.scatter(inp[:, 0], inp[:, 1],
                                 c=latent_pca[:, 0], cmap='viridis', s=3, alpha=0.5, rasterized=True)
        ax.set_title(f'{epoch}', fontsize=8)
    else:
        ax.set_title(f'{epoch}\nNo data', )
    
    ax.set_xlabel('Data dim 1')
    ax.set_ylabel('Data dim 2')
    ax.label_outer()

plt.tight_layout()
if 'scatter_obj' in globals():
    fig.colorbar(scatter_obj, ax=axs, label='PCA component 1')
export_folder = os.path.join (config.export_path + config.dataset_name + '/z_space_evals/')
fn= os.path.join(export_folder, 'latent_values_pca.pdf')
os.makedirs(export_folder, exist_ok=True)
plt.savefig(fn, dpi=300, transparent=True)  
print(f'figure saved to {fn}')




# %%
