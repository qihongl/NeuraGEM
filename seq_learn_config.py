"""
Shared settings for the seq_len_short CSW experiments.
Keep this in sync across runners and analyzers by importing from here.
"""

# Experiment identifiers
run_name = "all_models2" 
export_base_path = f'./exports/seq_learn/experiments/{run_name}/'

# Base params by model and curriculum
rnn_seq_len = 6
rnn_long_phase = 2500
rnn_short_phase = 1250
ng_long_phase = 2500
ng_short_phase = 1250
neuragem_seq_len = 6

def apply_neuragem_overrides(config, param_combination):
    ''''Applies Neuragem-specific overrides to the config based on the param_combination.'''
    config.LU_lr = 0.2 #0.3 
    config.l2_loss = 0.00003
    if 'seq_len' in param_combination:
        if param_combination['seq_len'] >10:
            config.LU_lr = 0.3
            config.l2_loss = 0.00005
    config.no_of_blocks = config.blocked_phase_length // config.block_size
    config.pass_previous_latent = True
    # config.add_passive_learning_phase = False # Cannot use. See over ride below
    config.no_of_steps_in_latent_space = 1 


def get_base_params():
    return {
        'rnn': {
            'blocked':             {'seq_len': [rnn_seq_len],
                'blocked_phase_length': [rnn_long_phase],
                'interleaved_phase_length': [rnn_long_phase],
            },
            'interleaved':         {'seq_len': [rnn_seq_len],
                'blocked_phase_length': [rnn_long_phase],
                'interleaved_phase_length': [rnn_long_phase],
            },
            'interleaved_blocked': {'seq_len': [rnn_seq_len],
                'blocked_phase_length': [rnn_short_phase],
                'interleaved_phase_length': [700],
            },
            # 'blocked_interleaved': {'seq_len': [rnn_seq_len],
            #     'blocked_phase_length': [rnn_short_phase],
            #     'interleaved_phase_length': [rnn_short_phase],
            # },
        },
        'mrnn': {
            'blocked':             {'seq_len': [200],
                'blocked_phase_length': [1200],
                'interleaved_phase_length': [1200],
            },
            'interleaved':         {'seq_len': [200],
                'blocked_phase_length': [1200],
                'interleaved_phase_length': [1200],
            },
            'interleaved_blocked': {'seq_len': [200],
                'blocked_phase_length': [700],
                'interleaved_phase_length': [500],
            },
            # 'blocked_interleaved': {'seq_len': [200],
            #     'blocked_phase_length': [700],
            #     'interleaved_phase_length': [500],
            # },
        },
        'neuragem': {
            'blocked':             {'seq_len': [neuragem_seq_len],
                'blocked_phase_length': [ng_long_phase],
                'interleaved_phase_length': [ng_long_phase],
                'latent_updates_during_shuffle': [True, ],
            },
            'interleaved':         {'seq_len': [neuragem_seq_len],
                'blocked_phase_length': [ng_long_phase],
                'interleaved_phase_length': [ng_long_phase],
                'latent_updates_during_shuffle': [True, ],
                # 'shuffle_or_interleave': ['interleave', 'shuffle'],

            },
            'interleaved_blocked': {
                'seq_len': [neuragem_seq_len],
                'blocked_phase_length': [ng_long_phase],
                'interleaved_phase_length': [700],
                # 'interleaved_phase_length': [ 100, 200, 300, 400, 500, ],
                # 'interleaved_phase_length': [50, 600, 700,800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
                'latent_updates_during_shuffle': [False, True],

            },
            # 'blocked_interleaved': {
            #     'seq_len': [neuragem_seq_len],
            #     'blocked_phase_length': [ng_short_phase],
            #     'interleaved_phase_length': [ng_short_phase],
            #     'latent_updates_during_shuffle': [True, False]
            # },
        },
    }
