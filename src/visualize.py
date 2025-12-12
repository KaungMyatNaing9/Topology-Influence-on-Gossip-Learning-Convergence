import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

try:
    from IPython.display import display, Image
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


def plot_accuracy_stripplots(df_graph, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams['figure.dpi'] = 130
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['axes.titlesize'] = 16

    k_palette = sns.color_palette("tab10", df_graph["k_target"].nunique())

    plt.figure(figsize=(9, 6))
    sns.stripplot(
        data=df_graph,
        x="topology",
        y="final_mean_accuracy",
        hue="k_target",
        dodge=True,
        jitter=0.25,
        size=7,
        palette=k_palette
    )
    plt.xlabel("Topology")
    plt.ylabel("Final Mean Accuracy (%)")
    plt.title("Final Gossip Accuracy by Topology and <k> Target", weight="bold")
    plt.legend(title="Target <k>")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_stripplot.pdf'))
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 6))
    sns.stripplot(
        data=df_graph,
        x="topology",
        y="convergence_round",
        hue="k_target",
        dodge=True,
        jitter=0.25,
        size=7,
        palette=k_palette
    )
    plt.xlabel("Topology")
    plt.ylabel("Convergence Round")
    plt.title("Convergence Speed by Topology and <k> Target", weight="bold")
    plt.legend(title="Target <k>")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_stripplot.pdf'))
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

    print(f"Saved stripplots to {output_dir}/")


def plot_node_distributions(df_nodes, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    topology_palette = sns.color_palette("colorblind", 3)

    plt.figure(figsize=(9, 6))
    sns.ecdfplot(
        data=df_nodes,
        x="final_accuracy",
        hue="topology",
        palette=topology_palette,
        linewidth=2
    )
    plt.xlabel("Node Final Accuracy (%)")
    plt.ylabel("ECDF")
    plt.title("ECDF of Node-Level Accuracy by Topology", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_ecdf.pdf'))
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 7))
    sns.kdeplot(
        data=df_nodes,
        x="degree",
        y="final_accuracy",
        hue="topology",
        palette=topology_palette,
        fill=True,
        alpha=0.45,
        thresh=0.1,
        linewidth=1.5
    )
    plt.xlabel("Node Degree")
    plt.ylabel("Node Final Accuracy (%)")
    plt.title("2D KDE: Node Degree vs Accuracy by Topology", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_accuracy_kde.pdf'))
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

    print(f"Saved distribution plots to {output_dir}/")


def visualize_topologies(num_nodes=12, k_values=(2, 4, 8), seed=0, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    FIG_WIDTH = 5.5
    FIG_HEIGHT = 5.0

    print(f"Matching experiment parameters: NUM_NODES={num_nodes}, K_VALUES={k_values}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    for k_target in k_values:
        print(f"\n===== Target average degree <k> = {k_target} =====")
        p_er = k_target / (num_nodes - 1)
        er_graph = nx.erdos_renyi_graph(n=num_nodes, p=p_er, seed=seed)

        k_ws = int(k_target)
        p_rewire = 0.3
        ws_graph = nx.watts_strogatz_graph(n=num_nodes, k=k_ws, p=p_rewire, seed=seed)

        m_ba = max(1, min(num_nodes - 1, int(round(k_target / 2))))
        ba_graph = nx.barabasi_albert_graph(n=num_nodes, m=m_ba, seed=seed)

        topologies = {
            'ER': {
                'graph': er_graph,
                'name': 'Erdos-Renyi',
                'subtitle': f'Random (p={p_er:.3f})',
                'layout': 'spring',
                'filename': os.path.join(output_dir, f'topology_erdos_renyi_k{k_target}.pdf')
            },
            'WS': {
                'graph': ws_graph,
                'name': 'Watts-Strogatz',
                'subtitle': f'Small-World (k={k_ws}, p={p_rewire})',
                'layout': 'circular',
                'filename': os.path.join(output_dir, f'topology_watts_strogatz_k{k_target}.pdf')
            },
            'BA': {
                'graph': ba_graph,
                'name': 'Barabasi-Albert',
                'subtitle': f'Scale-Free (m={m_ba})',
                'layout': 'spring',
                'filename': os.path.join(output_dir, f'topology_barabasi_albert_k{k_target}.pdf')
            }
        }

        for key, topo in topologies.items():
            graph = topo['graph']

            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

            num_edges = graph.number_of_edges()
            avg_degree = np.mean([d for _, d in graph.degree()])
            avg_clustering = nx.average_clustering(graph)

            if nx.is_connected(graph):
                diameter = nx.diameter(graph)
                avg_path_length = nx.average_shortest_path_length(graph)
            else:
                diameter = float('inf')
                avg_path_length = float('inf')

            if topo['layout'] == 'circular':
                pos = nx.circular_layout(graph)
            else:
                pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=seed)

            node_degrees = dict(graph.degree())
            node_colors = [node_degrees[node] for node in graph.nodes()]

            nodes = nx.draw_networkx_nodes(graph, pos, ax=ax,
                                            node_color=node_colors,
                                            node_size=500,
                                            cmap='viridis',
                                            vmin=min(node_colors) if node_colors else 0,
                                            vmax=max(node_colors) if node_colors else 1,
                                            edgecolors='black',
                                            linewidths=1.0)

            nx.draw_networkx_edges(graph, pos, ax=ax,
                                  alpha=0.4,
                                  width=1.5,
                                  edge_color='#333333')

            nx.draw_networkx_labels(graph, pos, ax=ax,
                                   font_size=11,
                                   font_weight='bold',
                                   font_color='white',
                                   font_family='sans-serif')

            title = f"{topo['name']} ({topo['subtitle']})"
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

            ax.axis('off')
            ax.set_aspect('equal')

            cbar = plt.colorbar(nodes, ax=ax, orientation='horizontal',
                                fraction=0.05, pad=0.02, aspect=30)
            cbar.set_label('Node Degree', fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)

            path_len_str = f"{avg_path_length:.2f}" if avg_path_length != float('inf') else "∞"
            diam_str = str(diameter) if diameter != float('inf') else "∞"

            metrics_text = (
                f"n = {num_nodes}   |   "
                f"m = {num_edges}   |   "
                f"⟨k⟩ = {avg_degree:.2f}   |   "
                f"C = {avg_clustering:.3f}   |   "
                f"D = {diam_str}   |   "
                f"ℓ = {path_len_str}"
            )

            fig.text(0.5, 0.02, metrics_text,
                     fontsize=9,
                     ha='center',
                     va='bottom',
                     family='sans-serif',
                     style='italic')

            plt.subplots_adjust(bottom=0.15)

            filename = topo['filename']
            plt.savefig(filename, format='pdf', bbox_inches='tight',
                        dpi=300, pad_inches=0.05)

            print(f"Saved: {filename}")
            print(f"  {topo['name']}: {num_edges} edges, "
                  f"avg degree = {avg_degree:.2f}, "
                  f"clustering = {avg_clustering:.3f}")

            if IN_NOTEBOOK:
                plt.show()

            plt.close()

    print("\n" + "=" * 60)
    print("All topology visualizations saved as PDFs")
