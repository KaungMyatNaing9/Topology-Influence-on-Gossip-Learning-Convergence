import numpy as np
import pandas as pd
import torch
import networkx as nx

from data import partition_data
from models import SimpleMLP
from network import GossipNetwork
from training import train_gossip_learning


def run_experiments_matched_degree(
    train_dataset,
    test_dataset,
    num_nodes=12,
    num_rounds=20,
    local_epochs=2,
    lr=0.01,
    alpha=1.0,
    convergence_threshold=70.0,
    dispersion_threshold=5.0,
    seeds=(0, 1, 2),
    k_values=(2, 4, 8),
    mixing_strategy='metropolis_hastings',
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Partitioning data for {
          num_nodes} nodes (non-IID Dirichlet, alpha={alpha})")
    node_datasets, distribution_info = partition_data(
        train_dataset, num_nodes, alpha=alpha)

    graph_rows = []
    node_rows = []
    round_rows = []
    node_round_rows = []
    distribution_rows = []

    for k_target in k_values:
        print(f"\n===== Target average degree <k> = {k_target} =====")

        for topology in ["ER", "WS", "BA"]:
            for seed in seeds:
                print(f"\n=== Running {topology} (seed={seed}, k_target={
                      k_target}, mixing={mixing_strategy}) ===")

                torch.manual_seed(seed)
                np.random.seed(seed)

                if topology == "ER":
                    p = k_target / (num_nodes - 1)
                    graph = nx.erdos_renyi_graph(n=num_nodes, p=p)
                    topo_params = {"p": p, "k_ws": np.nan, "m": np.nan}

                elif topology == "WS":
                    k_ws = int(k_target)
                    if k_ws % 2 != 0:
                        k_ws = max(2, k_ws - 1)
                    if k_ws >= num_nodes:
                        k_ws = num_nodes - 2 if num_nodes > 2 else 2
                    p_rewire = 0.3
                    graph = nx.watts_strogatz_graph(
                        n=num_nodes, k=k_ws, p=p_rewire)
                    topo_params = {"p": p_rewire, "k_ws": k_ws, "m": np.nan}

                elif topology == "BA":
                    m = max(1, min(num_nodes - 1, int(round(k_target / 2))))
                    graph = nx.barabasi_albert_graph(n=num_nodes, m=m)
                    topo_params = {"p": np.nan, "k_ws": np.nan, "m": m}

                else:
                    continue

                model_template = SimpleMLP()
                network = GossipNetwork(
                    graph,
                    model_template,
                    node_datasets,
                    device=device,
                    mixing_strategy=mixing_strategy
                )

                graph_metrics = network.get_graph_metrics()

                _, metrics = train_gossip_learning(
                    network=network,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    test_dataset=test_dataset,
                    convergence_threshold=convergence_threshold,
                    dispersion_threshold=dispersion_threshold,
                )

                final_stats = network.evaluate_all_nodes(test_dataset)
                node_metrics = network.get_node_metrics()

                for dist_info in distribution_info:
                    distribution_rows.append({
                        "topology": topology,
                        "mixing_strategy": mixing_strategy,
                        "num_nodes": num_nodes,
                        "seed": seed,
                        "k_target": k_target,
                        **dist_info
                    })

                graph_rows.append({
                    "topology": topology,
                    "mixing_strategy": mixing_strategy,
                    "num_nodes": num_nodes,
                    "seed": seed,
                    "k_target": k_target,
                    **topo_params,
                    **graph_metrics,
                    "converged": metrics["converged"],
                    "convergence_round": metrics["convergence_round"],
                    "final_mean_accuracy": metrics["mean_accuracy"][-1],
                    "final_std_accuracy": metrics["std_accuracy"][-1],
                })

                for node_id in range(num_nodes):
                    nm = node_metrics[node_id]
                    acc = final_stats["all_accuracies"][node_id]
                    node_rows.append({
                        "topology": topology,
                        "mixing_strategy": mixing_strategy,
                        "num_nodes": num_nodes,
                        "seed": seed,
                        "k_target": k_target,
                        "node_id": node_id,
                        "degree": nm["degree"],
                        "clustering": nm["clustering"],
                        "avg_neighbor_degree": nm["avg_neighbor_degree"],
                        "final_accuracy": acc,
                        "delta_from_mean": acc - final_stats["mean_accuracy"],
                    })

                for round_idx in range(len(metrics['rounds'])):
                    round_rows.append({
                        "topology": topology,
                        "mixing_strategy": mixing_strategy,
                        "num_nodes": num_nodes,
                        "seed": seed,
                        "k_target": k_target,
                        "round": metrics['rounds'][round_idx],
                        "mean_accuracy": metrics['mean_accuracy'][round_idx],
                        "std_accuracy": metrics['std_accuracy'][round_idx],
                        "min_accuracy": metrics['min_accuracy'][round_idx],
                        "max_accuracy": metrics['max_accuracy'][round_idx],
                        "communication_mb": metrics['communication_mb'][round_idx],
                        "time_per_round": metrics['time_per_round'][round_idx],
                    })

                    for node_id in range(num_nodes):
                        node_acc = metrics['node_accuracies'][round_idx][node_id]
                        node_round_rows.append({
                            "topology": topology,
                            "mixing_strategy": mixing_strategy,
                            "num_nodes": num_nodes,
                            "seed": seed,
                            "k_target": k_target,
                            "round": metrics['rounds'][round_idx],
                            "node_id": node_id,
                            "accuracy": node_acc,
                            "degree": node_metrics[node_id]["degree"],
                        })

    df_graph = pd.DataFrame(graph_rows)
    df_nodes = pd.DataFrame(node_rows)
    df_rounds = pd.DataFrame(round_rows)
    df_node_rounds = pd.DataFrame(node_round_rows)
    df_distribution = pd.DataFrame(distribution_rows)

    print("\n=== Graph-level summary (head) ===")
    print(df_graph.head())
    print("\n=== Node-level summary (head) ===")
    print(df_nodes.head())
    print("\n=== Round-level summary (head) ===")
    print(df_rounds.head())
    print("\n=== Node-Round-level summary (head) ===")
    print(df_node_rounds.head())
    print("\n=== Data distribution summary (head) ===")
    print(df_distribution.head())

    return df_graph, df_nodes, df_rounds, df_node_rounds, df_distribution
