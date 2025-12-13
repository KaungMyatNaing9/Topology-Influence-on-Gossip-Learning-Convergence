import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import EXPERIMENT_CONFIG, VISUALIZATION_CONFIG


def load_all_results(data_dir):
    """Load all CSV files from experiment_data directory"""
    graph_files = glob.glob(os.path.join(data_dir, "*_graph.csv"))
    rounds_files = glob.glob(os.path.join(data_dir, "*_rounds.csv"))

    print(f"Found {len(graph_files)} graph result files")
    print(f"Found {len(rounds_files)} rounds result files")

    df_graph_list = []
    df_rounds_list = []

    for graph_file in graph_files:
        df = pd.read_csv(graph_file)
        df_graph_list.append(df)

    for rounds_file in rounds_files:
        df = pd.read_csv(rounds_file)
        df_rounds_list.append(df)

    df_graph_all = pd.concat(df_graph_list, ignore_index=True) if df_graph_list else pd.DataFrame()
    df_rounds_all = pd.concat(df_rounds_list, ignore_index=True) if df_rounds_list else pd.DataFrame()

    print(f"\nTotal graph-level records: {len(df_graph_all)}")
    print(f"Total round-level records: {len(df_rounds_all)}")

    return df_graph_all, df_rounds_all


def plot_topology_accuracy_over_rounds_by_mixing(df_rounds, output_dir, node_count=30):
    """Create separate plots for each mixing strategy showing topology comparison over rounds"""
    print(f"\nGenerating topology accuracy plots for {node_count} nodes...")

    df = df_rounds[df_rounds['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    sns.set_theme(style="whitegrid", context="paper")

    for mixing in df['mixing_strategy'].unique():
        df_mix = df[df['mixing_strategy'] == mixing]

        fig, ax = plt.subplots(figsize=(8, 6))

        for topology in ['ER', 'WS', 'BA']:
            df_topo = df_mix[df_mix['topology'] == topology]
            if not df_topo.empty:
                ax.plot(df_topo['round'], df_topo['mean_accuracy'],
                       marker='o', linewidth=2, markersize=4,
                       label=topology, alpha=0.8)

        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
        ax.set_title(f'Topology Comparison - {mixing.replace("_", " ").title()}\n({node_count} nodes)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Topology', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'accuracy_over_rounds_{mixing}_{node_count}nodes.pdf')
        plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")


def plot_mixing_strategy_comparison(df_rounds, output_dir, node_count=30):
    """Compare the two mixing strategies across all topologies"""
    print(f"\nGenerating mixing strategy comparison for {node_count} nodes...")

    df = df_rounds[df_rounds['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8, 6))

    for mixing in df['mixing_strategy'].unique():
        df_mix = df[df['mixing_strategy'] == mixing]
        avg_by_round = df_mix.groupby('round')['mean_accuracy'].mean().reset_index()
        ax.plot(avg_by_round['round'], avg_by_round['mean_accuracy'],
               marker='o', linewidth=2.5, markersize=5,
               label=mixing.replace('_', ' ').title(), alpha=0.8)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax.set_title(f'Mixing Strategy Comparison\n({node_count} nodes, averaged across topologies)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mixing_comparison_{node_count}nodes.pdf')
    plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'metropolis_hastings': 'C0', 'uniform': 'C1'}
    markers = {'ER': 'o', 'WS': 's', 'BA': '^'}
    linestyles = {'metropolis_hastings': '-', 'uniform': '--'}

    for mixing in df['mixing_strategy'].unique():
        for topology in ['ER', 'WS', 'BA']:
            df_subset = df[(df['mixing_strategy'] == mixing) & (df['topology'] == topology)]
            if not df_subset.empty:
                label = f"{topology} - {mixing.replace('_', ' ').title()}"
                ax.plot(df_subset['round'], df_subset['mean_accuracy'],
                       marker=markers[topology], color=colors.get(mixing, 'gray'),
                       linestyle=linestyles.get(mixing, '-'),
                       linewidth=2, markersize=4, label=label, alpha=0.7)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax.set_title(f'Mixing Strategy × Topology Comparison\n({node_count} nodes)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mixing_topology_detailed_{node_count}nodes.pdf')
    plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_convergence_speed_comparison(df_graph, output_dir, node_count=30):
    """Compare convergence speed across topologies and mixing strategies"""
    print(f"\nGenerating convergence speed comparison for {node_count} nodes...")

    df = df_graph[df_graph['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8, 6))

    df_pivot = df.pivot_table(values='convergence_round',
                               index='topology',
                               columns='mixing_strategy')

    df_pivot.plot(kind='bar', ax=ax, width=0.7, alpha=0.8)

    ax.set_xlabel('Topology', fontsize=12)
    ax.set_ylabel('Convergence Round', fontsize=12)
    ax.set_title(f'Convergence Speed Comparison\n({node_count} nodes)',
                fontsize=14, fontweight='bold')
    ax.legend(title='Mixing Strategy', labels=[s.replace('_', ' ').title() for s in df_pivot.columns])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'convergence_speed_{node_count}nodes.pdf')
    plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_final_accuracy_comparison(df_graph, output_dir, node_count=30):
    """Compare final accuracy across topologies and mixing strategies"""
    print(f"\nGenerating final accuracy comparison for {node_count} nodes...")

    df = df_graph[df_graph['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8, 6))

    df_pivot = df.pivot_table(values='final_mean_accuracy',
                               index='topology',
                               columns='mixing_strategy')

    df_pivot.plot(kind='bar', ax=ax, width=0.7, alpha=0.8)

    ax.set_xlabel('Topology', fontsize=12)
    ax.set_ylabel('Final Mean Accuracy (%)', fontsize=12)
    ax.set_title(f'Final Accuracy Comparison\n({node_count} nodes)',
                fontsize=14, fontweight='bold')
    ax.legend(title='Mixing Strategy', labels=[s.replace('_', ' ').title() for s in df_pivot.columns])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'final_accuracy_{node_count}nodes.pdf')
    plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_accuracy_dispersion(df_rounds, output_dir, node_count=30):
    """Show how accuracy variance evolves over rounds"""
    print(f"\nGenerating accuracy dispersion plot for {node_count} nodes...")

    df = df_rounds[df_rounds['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'metropolis_hastings': 'C0', 'uniform': 'C1'}
    markers = {'ER': 'o', 'WS': 's', 'BA': '^'}

    for mixing in df['mixing_strategy'].unique():
        df_mix = df[df['mixing_strategy'] == mixing]
        avg_std_by_round = df_mix.groupby('round')['std_accuracy'].mean().reset_index()
        ax.plot(avg_std_by_round['round'], avg_std_by_round['std_accuracy'],
               marker='o', color=colors.get(mixing, 'gray'),
               linewidth=2.5, markersize=5,
               label=mixing.replace('_', ' ').title(), alpha=0.8)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Std. Deviation of Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy Dispersion Over Rounds\n({node_count} nodes, averaged across topologies)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'accuracy_dispersion_{node_count}nodes.pdf')
    plt.savefig(output_file, format='pdf', dpi=VISUALIZATION_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def generate_summary_table(df_graph, output_dir, node_count=30):
    """Generate a summary statistics table"""
    print(f"\nGenerating summary statistics table for {node_count} nodes...")

    df = df_graph[df_graph['num_nodes'] == node_count].copy()

    if df.empty:
        print(f"  WARNING: No data found for {node_count} nodes")
        return

    summary = df.groupby(['topology', 'mixing_strategy']).agg({
        'convergence_round': 'mean',
        'final_mean_accuracy': 'mean',
        'final_std_accuracy': 'mean'
    }).round(2)

    summary.columns = ['Convergence Round', 'Final Mean Accuracy (%)', 'Final Std Accuracy (%)']

    output_file = os.path.join(output_dir, f'summary_table_{node_count}nodes.csv')
    summary.to_csv(output_file)
    print(f"  Saved: {output_file}")
    print("\nSummary Statistics:")
    print(summary)


def main():
    data_dir = EXPERIMENT_CONFIG['data_dir']
    results_dir = VISUALIZATION_CONFIG['results_dir']

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("GENERATING LATEX-READY VISUALIZATIONS FROM EXPERIMENT RESULTS")
    print("=" * 80)

    df_graph, df_rounds = load_all_results(data_dir)

    if df_graph.empty or df_rounds.empty:
        print("\nERROR: No experiment results found!")
        print(f"Please check: {data_dir}/")
        return

    node_counts = sorted(df_graph['num_nodes'].unique())
    print(f"\nNode counts found in data: {node_counts}")

    for node_count in node_counts:
        print(f"\n{'='*80}")
        print(f"Generating plots for {node_count} nodes")
        print(f"{'='*80}")

        plot_topology_accuracy_over_rounds_by_mixing(df_rounds, results_dir, node_count)
        plot_mixing_strategy_comparison(df_rounds, results_dir, node_count)
        plot_convergence_speed_comparison(df_graph, results_dir, node_count)
        plot_final_accuracy_comparison(df_graph, results_dir, node_count)
        plot_accuracy_dispersion(df_rounds, results_dir, node_count)
        generate_summary_table(df_graph, results_dir, node_count)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
