import numpy as np
import torch
import networkx as nx

from node import Node


class GossipNetwork:
    def __init__(self, graph, model_template, datasets, device='cpu', mixing_strategy='metropolis_hastings'):
        self.graph = graph
        self.device = device
        self.nodes = {}
        self.mixing_strategy = mixing_strategy

        for node_id in graph.nodes():
            neighbors = list(graph.neighbors(node_id))
            node_model = type(model_template)()
            node_model.load_state_dict(model_template.state_dict())

            self.nodes[node_id] = Node(
                node_id=node_id,
                model=node_model,
                dataset=datasets[node_id],
                neighbors=neighbors,
                device=device
            )

        self.mixing_matrix = self.compute_mixing_matrix()

        self.total_bytes_sent = 0
        self.round_communication = []

    def compute_mixing_matrix(self):
        if self.mixing_strategy == 'metropolis_hastings':
            return self._compute_metropolis_hastings()
        elif self.mixing_strategy == 'uniform':
            return self._compute_uniform()
        elif self.mixing_strategy == 'degree_normalized':
            return self._compute_degree_normalized()
        elif self.mixing_strategy == 'laplacian':
            return self._compute_laplacian()
        else:
            raise ValueError(f"Unknown mixing strategy: {
                             self.mixing_strategy}")

    def _compute_metropolis_hastings(self):
        n = self.graph.number_of_nodes()
        W = np.zeros((n, n))
        degrees = dict(self.graph.degree())

        for i in self.graph.nodes():
            for j in self.graph.neighbors(i):
                W[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))

        for i in self.graph.nodes():
            W[i, i] = 1.0 - sum(W[i, j] for j in self.graph.neighbors(i))

        return W

    def _compute_uniform(self):
        n = self.graph.number_of_nodes()
        W = np.zeros((n, n))
        degrees = dict(self.graph.degree())

        for i in self.graph.nodes():
            num_neighbors = degrees[i]
            if num_neighbors > 0:
                for j in self.graph.neighbors(i):
                    W[i, j] = 1.0 / (num_neighbors + 1)
                W[i, i] = 1.0 / (num_neighbors + 1)
            else:
                W[i, i] = 1.0

        return W

    def _compute_degree_normalized(self):
        n = self.graph.number_of_nodes()
        W = np.zeros((n, n))
        degrees = dict(self.graph.degree())

        for i in self.graph.nodes():
            for j in self.graph.neighbors(i):
                W[i, j] = 1.0 / max(degrees[i], 1)

        for i in self.graph.nodes():
            row_sum = sum(W[i, j] for j in self.graph.neighbors(i))
            if row_sum < 1.0:
                W[i, i] = 1.0 - row_sum
            else:
                for j in self.graph.neighbors(i):
                    W[i, j] /= row_sum

        return W

    def _compute_laplacian(self):
        n = self.graph.number_of_nodes()
        L = nx.normalized_laplacian_matrix(self.graph).toarray()
        alpha = 0.5
        W = np.eye(n) - alpha * L

        W = np.maximum(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / np.maximum(row_sums, 1e-10)

        return W

    def gossip_round(self, local_epochs=1, lr=0.01):
        for node in self.nodes.values():
            node.local_train(epochs=local_epochs, lr=lr)

        old_params = {node_id: node.get_model_params()
                      for node_id, node in self.nodes.items()}

        round_bytes = 0
        param_size_bytes = next(iter(old_params.values())).numel() * 4

        for i, node in self.nodes.items():
            new_params = torch.zeros_like(old_params[i])

            for j in range(len(self.nodes)):
                weight = self.mixing_matrix[i][j]
                if weight > 0:
                    new_params += weight * old_params[j]

                    if i != j and j in node.neighbors:
                        round_bytes += param_size_bytes
                        node.bytes_received += param_size_bytes
                        self.nodes[j].bytes_sent += param_size_bytes

            node.set_model_params(new_params)

        self.total_bytes_sent += round_bytes
        self.round_communication.append(round_bytes)

        return round_bytes

    def evaluate_all_nodes(self, test_dataset=None):
        """Evaluate all nodes and return statistics."""
        accuracies = []
        losses = []

        for node in self.nodes.values():
            loss, acc = node.evaluate(test_dataset)
            accuracies.append(acc)
            losses.append(loss)

        stats = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'all_accuracies': accuracies,
            'mean_loss': float(np.mean(losses)),
        }
        return stats

    def get_node_metrics(self):
        """Compute node-level metrics for each node in the graph."""
        G = self.graph
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        avg_neighbor_degree = nx.average_neighbor_degree(G)

        node_metrics = {}
        for node in G.nodes():
            node_metrics[node] = {
                "degree": degrees[node],
                "clustering": clustering[node],
                "avg_neighbor_degree": avg_neighbor_degree[node],
            }
        return node_metrics

    def get_graph_metrics(self):
        """Compute and return graph-level metrics."""
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': float(np.mean([d for _, d in self.graph.degree()])),
            'avg_clustering': float(nx.average_clustering(self.graph)),
        }

        if nx.is_connected(self.graph):
            metrics['diameter'] = nx.diameter(self.graph)
            metrics['avg_path_length'] = float(
                nx.average_shortest_path_length(self.graph))
        else:
            metrics['diameter'] = float('inf')
            metrics['avg_path_length'] = float('inf')

        return metrics
