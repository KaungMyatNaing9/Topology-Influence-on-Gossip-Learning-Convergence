import os
import time
from datetime import datetime
import torch
import numpy as np
from torchvision import datasets, transforms

from config import EXPERIMENT_CONFIG, DATASET_CONFIG
from experiments import run_experiments_matched_degree


def setup_pytorch_threads():
    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    torch.set_num_threads(num_cores)

    torch.set_num_interop_threads(max(1, num_cores // 2))

    print(f"PyTorch configured to use {num_cores} threads")
    print(f"  Intra-op threads: {torch.get_num_threads()}")
    print(f"  Inter-op threads: {torch.get_num_interop_threads()}")


def main():
    setup_pytorch_threads()

    node_counts = EXPERIMENT_CONFIG['node_counts']
    k_values = EXPERIMENT_CONFIG['k_values']
    mixing_strategies = EXPERIMENT_CONFIG['mixing_strategies']
    seeds = EXPERIMENT_CONFIG['seeds']
    num_rounds = EXPERIMENT_CONFIG['num_rounds']
    local_epochs = EXPERIMENT_CONFIG['local_epochs']
    lr = EXPERIMENT_CONFIG['learning_rate']
    alpha = EXPERIMENT_CONFIG['alpha']
    convergence_threshold = EXPERIMENT_CONFIG['convergence_threshold']
    dispersion_threshold = EXPERIMENT_CONFIG['dispersion_threshold']
    data_dir = EXPERIMENT_CONFIG['data_dir']
    device = EXPERIMENT_CONFIG['device']

    os.makedirs(data_dir, exist_ok=True)

    experiment_start_time = datetime.now()
    print("=" * 80)
    print(f"GOSSIP LEARNING EXPERIMENT SUITE")
    print(f"Started at: {experiment_start_time}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Node counts: {node_counts}")
    print(f"  K values: {k_values}")
    print(f"  Mixing strategies: {mixing_strategies}")
    print(f"  Seeds: {seeds}")
    print(f"  Runs per config: {len(seeds)}")
    print(f"  Max rounds: {num_rounds}")
    print("=" * 80)

    print("\nLOADING FASHION-MNIST DATASET")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (DATASET_CONFIG['normalization_mean'],),
            (DATASET_CONFIG['normalization_std'],)
        )
    ])

    train_dataset = datasets.FashionMNIST(
        root=DATASET_CONFIG['data_root'],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=DATASET_CONFIG['data_root'],
        train=False,
        download=True,
        transform=transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    total_experiments = len(node_counts) * len(mixing_strategies)
    experiment_counter = 0

    for num_nodes in node_counts:
        for mixing_strategy in mixing_strategies:
            experiment_counter += 1
            print("\n" + "=" * 80)
            print(f"EXPERIMENT {experiment_counter}/{total_experiments}")
            print(f"Nodes: {num_nodes} | Mixing: {mixing_strategy}")
            print("=" * 80)

            exp_start_time = time.time()

            df_graph, df_nodes, df_rounds, df_node_rounds, df_distribution = run_experiments_matched_degree(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_nodes=num_nodes,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                lr=lr,
                alpha=alpha,
                convergence_threshold=convergence_threshold,
                dispersion_threshold=dispersion_threshold,
                seeds=seeds,
                k_values=k_values,
                mixing_strategy=mixing_strategy,
                device=device,
            )

            filename_base = f"nodes{num_nodes}_mixing{mixing_strategy}"
            graph_csv = os.path.join(data_dir, f"{filename_base}_graph.csv")
            nodes_csv = os.path.join(data_dir, f"{filename_base}_nodes.csv")
            rounds_csv = os.path.join(data_dir, f"{filename_base}_rounds.csv")
            node_rounds_csv = os.path.join(data_dir, f"{filename_base}_node_rounds.csv")
            distribution_csv = os.path.join(data_dir, f"{filename_base}_distribution.csv")

            df_graph.to_csv(graph_csv, index=False)
            df_nodes.to_csv(nodes_csv, index=False)
            df_rounds.to_csv(rounds_csv, index=False)
            df_node_rounds.to_csv(node_rounds_csv, index=False)
            df_distribution.to_csv(distribution_csv, index=False)

            exp_duration = time.time() - exp_start_time
            print(f"\nSaved results:")
            print(f"  Graph-level: {graph_csv}")
            print(f"  Node-level: {nodes_csv}")
            print(f"  Round-level: {rounds_csv}")
            print(f"  Node-Round-level: {node_rounds_csv}")
            print(f"  Distribution: {distribution_csv}")
            print(f"  Duration: {exp_duration/60:.2f} minutes")

    total_duration = (datetime.now() - experiment_start_time).total_seconds()
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Total experiments: {total_experiments}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Results saved to: {data_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
