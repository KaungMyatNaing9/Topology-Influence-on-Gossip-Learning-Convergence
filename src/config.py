EXPERIMENT_CONFIG = {
    'node_counts': [30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200],

    'k_values': [2, 4, 6, 8, 10],

    'mixing_strategies': [
        'metropolis_hastings',
        'uniform',
        'degree_normalized',
        'laplacian'
    ],

    'topologies': ['ER', 'WS', 'BA'],

    'num_runs': 3,

    'seeds': [42, 123, 456],

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
