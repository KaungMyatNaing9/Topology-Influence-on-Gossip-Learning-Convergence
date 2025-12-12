import torch
import numpy as np
from torchvision import datasets, transforms

from experiments import run_experiments_matched_degree
from visualize import plot_accuracy_stripplots, plot_node_distributions, visualize_topologies


def main():
    NUM_NODES = 12
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 2
    LEARNING_RATE = 0.01
    ALPHA = 1.0
    SEEDS = (0, 1, 2)
    K_VALUES = (2, 4, 8)

    torch.manual_seed(42)
    np.random.seed(42)

    print("LOADING FASHION-MNIST DATASET")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset_full = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"Total training samples (full): {len(train_dataset_full)}")
    print(f"Total test samples: {len(test_dataset)}")
    train_dataset = train_dataset_full
    print(f"Using FULL training dataset: {len(train_dataset)} samples")

    df_graph, df_nodes, df_rounds, df_node_rounds, df_distribution = run_experiments_matched_degree(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_nodes=NUM_NODES,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LEARNING_RATE,
        alpha=ALPHA,
        convergence_threshold=70.0,
        dispersion_threshold=5.0,
        seeds=SEEDS,
        k_values=K_VALUES,
    )

    df_graph.to_csv("gossip_graph_matched_degree.csv", index=False)
    df_nodes.to_csv("gossip_node_matched_degree.csv", index=False)
    df_rounds.to_csv("gossip_rounds_matched_degree.csv", index=False)
    df_node_rounds.to_csv("gossip_node_rounds_matched_degree.csv", index=False)
    df_distribution.to_csv(
        "gossip_distribution_matched_degree.csv", index=False)
    print("\nSaved all CSV files: graph, nodes, rounds, node_rounds, and distribution")

    print("\nGenerating visualizations...")
    plot_accuracy_stripplots(df_graph, output_dir='results')
    plot_node_distributions(df_nodes, output_dir='results')
    visualize_topologies(
        num_nodes=NUM_NODES,
        k_values=K_VALUES,
        seed=SEEDS[0],
        output_dir='results'
    )

    print("\nAll experiments and visualizations complete!")


if __name__ == "__main__":
    main()
