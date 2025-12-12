import numpy as np
from torch.utils.data import Subset


def partition_data(dataset, num_nodes, alpha=0.5, num_classes=10):
    print(f"Partitioning data into {
          num_nodes} nodes with Dirichlet(alpha={alpha})")

    if isinstance(dataset, Subset):
        labels_full = np.array(dataset.dataset.targets)
        base_indices = np.array(dataset.indices)
        labels = labels_full[base_indices]
    else:
        labels = np.array(dataset.targets)
        base_indices = np.arange(len(dataset))

    class_indices = []
    for c in range(num_classes):
        cls_idx = base_indices[labels == c]
        class_indices.append(cls_idx)

    node_datasets = {i: [] for i in range(num_nodes)}

    for c_idx in class_indices:
        c_idx = np.array(c_idx)
        if len(c_idx) == 0:
            continue

        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_nodes))
        proportions = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        splits = np.split(c_idx, proportions)

        for node_id, split in enumerate(splits):
            node_datasets[node_id].extend(split.tolist())

    datasets_dict = {}
    distribution_info = []

    for node_id in range(num_nodes):
        indices_node = node_datasets[node_id]
        np.random.shuffle(indices_node)
        datasets_dict[node_id] = Subset(dataset, indices_node)

        node_labels = labels[np.isin(base_indices, indices_node)]
        class_counts = {c: int(np.sum(node_labels == c)) for c in range(num_classes)}

        distribution_info.append({
            'node_id': node_id,
            'total_samples': len(indices_node),
            **{f'class_{c}_count': class_counts[c] for c in range(num_classes)}
        })

        print(f"Node {node_id}: {len(indices_node)} samples")

    return datasets_dict, distribution_info
