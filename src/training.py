import time


def train_gossip_learning(
    network,
    num_rounds=20,
    local_epochs=1,
    lr=0.01,
    test_dataset=None,
    convergence_threshold=70.0,
    dispersion_threshold=5.0,
):
    metrics = {
        'rounds': [],
        'mean_accuracy': [],
        'std_accuracy': [],
        'min_accuracy': [],
        'max_accuracy': [],
        'communication_mb': [],
        'time_per_round': [],
        'node_accuracies': []
    }

    print("Starting Gossip Learning Simulation")
    print(f"Network: {network.graph.number_of_nodes()} nodes, "
          f"{network.graph.number_of_edges()} edges")
    print(f"Parameters: {num_rounds} rounds (max), {
          local_epochs} local epoch(s), lr={lr}")

    converged = False
    convergence_round = None

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")

        start_time = time.time()

        round_bytes = network.gossip_round(local_epochs=local_epochs, lr=lr)

        stats = network.evaluate_all_nodes(test_dataset)

        round_time = time.time() - start_time

        metrics['rounds'].append(round_num + 1)
        metrics['mean_accuracy'].append(stats['mean_accuracy'])
        metrics['std_accuracy'].append(stats['std_accuracy'])
        metrics['min_accuracy'].append(stats['min_accuracy'])
        metrics['max_accuracy'].append(stats['max_accuracy'])
        metrics['communication_mb'].append(
            network.total_bytes_sent / (1024 ** 2))
        metrics['time_per_round'].append(round_time)
        metrics['node_accuracies'].append(stats['all_accuracies'])

        print(f"  Accuracy: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}% "
              f"[{stats['min_accuracy']:.2f}%, {stats['max_accuracy']:.2f}%]")
        print(f"  Communication: {round_bytes / (1024 ** 2):.3f} MB, "
              f"{network.total_bytes_sent / (1024 ** 2):.3f} MB total")
        print(f"  Time: {round_time:.2f}s\n")

        if (
            not converged
            and stats['mean_accuracy'] >= convergence_threshold
            and stats['std_accuracy'] <= dispersion_threshold
        ):
            converged = True
            convergence_round = round_num + 1
            print(f"CONVERGED at round {convergence_round}! "
                  f"Accuracy: {stats['mean_accuracy']:.2f}%\n")

    print("Training Complete")
    print(f"Final Accuracy: {metrics['mean_accuracy'][-1]:.2f}% ± "
          f"{metrics['std_accuracy'][-1]:.2f}%")
    print(f"Total Communication: {metrics['communication_mb'][-1]:.2f} MB")
    print(f"Total Time: {sum(metrics['time_per_round']):.2f}s")
    if converged:
        print(f"Converged at round: {convergence_round}")
    else:
        print("Did not meet convergence criteria within max rounds.")

    metrics['converged'] = converged
    metrics['convergence_round'] = convergence_round

    return network, metrics
