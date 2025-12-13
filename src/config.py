EXPERIMENT_CONFIG = {
    'node_counts': [30, 50, 60, 80, 100],

    'k_values': [4],

    'mixing_strategies': [
        'metropolis_hastings',
        'uniform',
    ],

    'topologies': ['ER', 'WS', 'BA'],

    'num_runs': 1,

    'seeds': [42],

    'num_rounds': 40,
    'local_epochs': 2,
    'learning_rate': 0.01,

    'alpha': 1.0,

    'convergence_threshold': 70.0,
    'dispersion_threshold': 5.0,

    'results_dir': 'results',
    'data_dir': 'experiment_data',

    'device': 'cuda',
}

DATASET_CONFIG = {
    'name': 'FashionMNIST',
    'data_root': './data',
    'normalization_mean': 0.5,
    'normalization_std': 0.5,
}

VISUALIZATION_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'pdf',
    'results_dir': 'results',
}
