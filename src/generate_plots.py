import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import EXPERIMENT_CONFIG, VISUALIZATION_CONFIG


def load_all_results(data_dir):
    graph_files = glob.glob(os.path.join(data_dir, "*_graph.csv"))
    node_files = glob.glob(os.path.join(data_dir, "*_nodes.csv"))

    print(f"Found {len(graph_files)} graph result files")
    print(f"Found {len(node_files)} node result files")

    df_graph_list = []
    df_nodes_list = []

    for graph_file in graph_files:
        df = pd.read_csv(graph_file)
        df_graph_list.append(df)

    for node_file in node_files:
        df = pd.read_csv(node_file)
        df_nodes_list.append(df)

    df_graph_all = pd.concat(
        df_graph_list, ignore_index=True) if df_graph_list else pd.DataFrame()
    df_nodes_all = pd.concat(
        df_nodes_list, ignore_index=True) if df_nodes_list else pd.DataFrame()

    print(f"\nTotal graph-level records: {len(df_graph_all)}")
    print(f"Total node-level records: {len(df_nodes_all)}")

    return df_graph_all, df_nodes_all


def plot_convergence_by_mixing(df_graph, output_dir):
    print("\nGenerating convergence comparison plots...")

    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        data=df_graph,
        x='mixing_strategy',
        y='convergence_round',
        hue='topology',
        ax=axes[0]
    )
    axes[0].set_title('Convergence Speed by Mixing Strategy',
                      fontweight='bold')
    axes[0].set_xlabel('Mixing Strategy')
    axes[0].set_ylabel('Convergence Round')
    axes[0].tick_params(axis='x', rotation=45)

    sns.lineplot(
        data=df_graph,
        x='num_nodes',
        y='convergence_round',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        dashes=False,
        ax=axes[1]
    )
    axes[1].set_title('Convergence vs Network Size', fontweight='bold')
    axes[1].set_xlabel('Number of Nodes')
    axes[1].set_ylabel('Convergence Round')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'convergence_comparison.pdf')
    plt.savefig(output_file, format='pdf',
                dpi=VISUALIZATION_CONFIG['figure_dpi'])
    plt.close()
    print(f"  Saved: {output_file}")


def plot_accuracy_by_mixing(df_graph, output_dir):
    print("\nGenerating accuracy comparison plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.violinplot(
        data=df_graph,
        x='mixing_strategy',
        y='final_mean_accuracy',
        hue='topology',
        split=False,
        ax=axes[0]
    )
    axes[0].set_title('Final Accuracy by Mixing Strategy', fontweight='bold')
    axes[0].set_xlabel('Mixing Strategy')
    axes[0].set_ylabel('Final Mean Accuracy (%)')
    axes[0].tick_params(axis='x', rotation=45)

    sns.lineplot(
        data=df_graph,
        x='num_nodes',
        y='final_mean_accuracy',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        dashes=False,
        ax=axes[1]
    )
    axes[1].set_title('Accuracy vs Network Size', fontweight='bold')
    axes[1].set_xlabel('Number of Nodes')
    axes[1].set_ylabel('Final Mean Accuracy (%)')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'accuracy_comparison.pdf')
    plt.savefig(output_file, format='pdf',
                dpi=VISUALIZATION_CONFIG['figure_dpi'])
    plt.close()
    print(f"  Saved: {output_file}")


def plot_k_value_analysis(df_graph, output_dir):
    print("\nGenerating k-value analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.lineplot(
        data=df_graph,
        x='k_target',
        y='convergence_round',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title(
        'Convergence vs K (Average Degree)', fontweight='bold')
    axes[0, 0].set_xlabel('K (Target Average Degree)')
    axes[0, 0].set_ylabel('Convergence Round')

    sns.lineplot(
        data=df_graph,
        x='k_target',
        y='final_mean_accuracy',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Accuracy vs K (Average Degree)', fontweight='bold')
    axes[0, 1].set_xlabel('K (Target Average Degree)')
    axes[0, 1].set_ylabel('Final Mean Accuracy (%)')

    pivot_conv = df_graph.groupby(['k_target', 'mixing_strategy'])[
        'convergence_round'].mean().unstack()
    sns.heatmap(pivot_conv, annot=True, fmt='.1f',
                cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title(
        'Avg Convergence Round by K and Mixing', fontweight='bold')
    axes[1, 0].set_xlabel('Mixing Strategy')
    axes[1, 0].set_ylabel('K Value')

    pivot_acc = df_graph.groupby(['k_target', 'mixing_strategy'])[
        'final_mean_accuracy'].mean().unstack()
    sns.heatmap(pivot_acc, annot=True, fmt='.2f', cmap='YlGn', ax=axes[1, 1])
    axes[1, 1].set_title(
        'Avg Final Accuracy by K and Mixing', fontweight='bold')
    axes[1, 1].set_xlabel('Mixing Strategy')
    axes[1, 1].set_ylabel('K Value')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'k_value_analysis.pdf')
    plt.savefig(output_file, format='pdf',
                dpi=VISUALIZATION_CONFIG['figure_dpi'])
    plt.close()
    print(f"  Saved: {output_file}")


def plot_topology_comparison(df_graph, output_dir):
    print("\nGenerating topology comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.boxplot(
        data=df_graph,
        x='topology',
        y='convergence_round',
        hue='mixing_strategy',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Convergence by Topology', fontweight='bold')
    axes[0, 0].set_xlabel('Topology')
    axes[0, 0].set_ylabel('Convergence Round')

    sns.boxplot(
        data=df_graph,
        x='topology',
        y='final_mean_accuracy',
        hue='mixing_strategy',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Accuracy by Topology', fontweight='bold')
    axes[0, 1].set_xlabel('Topology')
    axes[0, 1].set_ylabel('Final Mean Accuracy (%)')

    sns.violinplot(
        data=df_graph,
        x='topology',
        y='final_std_accuracy',
        hue='mixing_strategy',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Accuracy Dispersion by Topology', fontweight='bold')
    axes[1, 0].set_xlabel('Topology')
    axes[1, 0].set_ylabel('Std Accuracy (%)')

    conv_rate = df_graph.groupby(['topology', 'mixing_strategy'])[
        'converged'].mean().unstack() * 100
    conv_rate.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Convergence Rate by Topology', fontweight='bold')
    axes[1, 1].set_xlabel('Topology')
    axes[1, 1].set_ylabel('Convergence Rate (%)')
    axes[1, 1].legend(title='Mixing Strategy')
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'topology_comparison.pdf')
    plt.savefig(output_file, format='pdf',
                dpi=VISUALIZATION_CONFIG['figure_dpi'])
    plt.close()
    print(f"  Saved: {output_file}")


def plot_scalability_analysis(df_graph, output_dir):
    print("\nGenerating scalability analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for mixing in df_graph['mixing_strategy'].unique():
        subset = df_graph[df_graph['mixing_strategy'] == mixing]
        sns.lineplot(
            data=subset,
            x='num_nodes',
            y='convergence_round',
            hue='topology',
            markers=True,
            ax=axes[0, 0],
            label=f"{mixing}"
        )
    axes[0, 0].set_title(
        'Scalability: Convergence vs Network Size', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Convergence Round')

    sns.lineplot(
        data=df_graph,
        x='num_nodes',
        y='final_mean_accuracy',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title(
        'Scalability: Accuracy vs Network Size', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Nodes')
    axes[0, 1].set_ylabel('Final Mean Accuracy (%)')

    sns.lineplot(
        data=df_graph,
        x='num_nodes',
        y='final_std_accuracy',
        hue='mixing_strategy',
        style='topology',
        markers=True,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title(
        'Scalability: Dispersion vs Network Size', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Nodes')
    axes[1, 0].set_ylabel('Std Accuracy (%)')

    summary = df_graph.groupby(['num_nodes', 'mixing_strategy']).agg({
        'convergence_round': 'mean',
        'final_mean_accuracy': 'mean'
    }).reset_index()

    summary['efficiency'] = summary['final_mean_accuracy'] / \
        (summary['convergence_round'] + 1)

    sns.lineplot(
        data=summary,
        x='num_nodes',
        y='efficiency',
        hue='mixing_strategy',
        markers=True,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title(
        'Efficiency: Accuracy / Convergence Time', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Nodes')
    axes[1, 1].set_ylabel('Efficiency Score')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'scalability_analysis.pdf')
    plt.savefig(output_file, format='pdf',
                dpi=VISUALIZATION_CONFIG['figure_dpi'])
    plt.close()
    print(f"  Saved: {output_file}")


def generate_summary_statistics(df_graph, output_dir):
    print("\nGenerating summary statistics...")

    summary_mixing = df_graph.groupby('mixing_strategy').agg({
        'convergence_round': ['mean', 'std', 'min', 'max'],
        'final_mean_accuracy': ['mean', 'std', 'min', 'max'],
        'converged': 'mean'
    }).round(2)

    summary_file = os.path.join(output_dir, 'summary_by_mixing.csv')
    summary_mixing.to_csv(summary_file)
    print(f"  Saved: {summary_file}")

    summary_topology = df_graph.groupby('topology').agg({
        'convergence_round': ['mean', 'std'],
        'final_mean_accuracy': ['mean', 'std'],
        'converged': 'mean'
    }).round(2)

    summary_file = os.path.join(output_dir, 'summary_by_topology.csv')
    summary_topology.to_csv(summary_file)
    print(f"  Saved: {summary_file}")


def main():
    data_dir = EXPERIMENT_CONFIG['data_dir']
    results_dir = VISUALIZATION_CONFIG['results_dir']

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("GENERATING VISUALIZATIONS FROM EXPERIMENT RESULTS")
    print("=" * 80)

    df_graph, df_nodes = load_all_results(data_dir)

    if df_graph.empty:
        print("\nERROR: No experiment results found!")
        print(f"Please run experiments first: python src/run_experiments.py")
        return

    plot_convergence_by_mixing(df_graph, results_dir)
    plot_accuracy_by_mixing(df_graph, results_dir)
    plot_k_value_analysis(df_graph, results_dir)
    plot_topology_comparison(df_graph, results_dir)
    plot_scalability_analysis(df_graph, results_dir)

    generate_summary_statistics(df_graph, results_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
