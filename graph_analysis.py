import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
import ast
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

from src.modeling import prepare_raw_modeling_data, CORE_ASPECTS
from src.network_builder import construct_partial_correlation_network, select_best_precision_ebic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = 'graph_analysis'
DATA_PATH = 'data/Seminar_Amazon_Results_FULL.csv'
RANDOM_STATE = 42

def setup_output():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.rcParams.update({'font.size': 10})

def load_full_data():
    logger.info(f"Loading full dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"File {DATA_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    # Handle string representation of list in aspect_sentiments
    if isinstance(df['aspect_sentiments'].iloc[0], str):
        df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    
    # Prepare feature matrix using existing logic
    feature_matrix = prepare_raw_modeling_data(df)
    # Drop rating column as we only need aspect scores for the network
    feature_matrix = feature_matrix[CORE_ASPECTS]
    
    return feature_matrix

def estimate_precision_matrix(feature_matrix, gamma=0.5):
    """Estimates precision matrix using the EBIC selected lambda for consistency."""
    active_aspects = [a for a in feature_matrix.columns if feature_matrix[a].std() > 0]
    X = feature_matrix[active_aspects]
    N, P = X.shape
    X_scaled = StandardScaler().fit_transform(X)
    S = np.cov(X_scaled.T, bias=True)
    
    logger.info("Estimating precision matrix using EBIC (GLASSO)...")
    best_precision, _ = select_best_precision_ebic(S, N, P, gamma=gamma)
    
    # Create full precision matrix for all CORE_ASPECTS
    full_precision = pd.DataFrame(0.0, index=CORE_ASPECTS, columns=CORE_ASPECTS)
    if best_precision is not None:
        for i, a1 in enumerate(active_aspects):
            for j, a2 in enumerate(active_aspects):
                full_precision.loc[a1, a2] = best_precision[i, j]
            
    return full_precision

def compute_node_metrics(G):
    logger.info("Computing node-level metrics...")
    
    # A1. Degree Centrality
    degree_cent = nx.degree_centrality(G)
    
    # A2. Weighted Degree (Strength)
    strength = dict(G.degree(weight='weight'))
    
    # A3. PageRank
    pagerank = nx.pagerank(G, weight='weight', alpha=0.85)
    
    # A4. Betweenness Centrality
    # Use 1/weight as distance
    dist_G = G.copy()
    for u, v, d in dist_G.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)
    betweenness = nx.betweenness_centrality(dist_G, weight='distance', normalized=True)
    
    # A5. Closeness Centrality
    closeness = nx.closeness_centrality(dist_G, distance='distance')
    
    # A6. Eigenvector Centrality
    try:
        eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except Exception as e:
        logger.warning(f"Eigenvector centrality failed: {e}. Using zero values.")
        eigenvector = {node: 0.0 for node in G.nodes()}
        
    # A7. Clustering Coefficient
    clustering = nx.clustering(G, weight='weight')
    
    # A8. Node Eccentricity
    eccentricity = {}
    if nx.is_connected(dist_G):
        eccentricity = nx.eccentricity(dist_G, sp=dict(nx.all_pairs_dijkstra_path_length(dist_G, weight='distance')))
    else:
        logger.warning("Graph is disconnected. Computing eccentricity per component.")
        for component in nx.connected_components(dist_G):
            subgraph = dist_G.subgraph(component)
            comp_ecc = nx.eccentricity(subgraph, sp=dict(nx.all_pairs_dijkstra_path_length(subgraph, weight='distance')))
            eccentricity.update(comp_ecc)

    # C5. Community Detection (Louvain)
    partition = community_louvain.best_partition(G, weight='weight', random_state=RANDOM_STATE)
    
    node_df = pd.DataFrame({
        'node': CORE_ASPECTS,
        'degree_centrality': [degree_cent.get(a, 0) for a in CORE_ASPECTS],
        'weighted_degree': [strength.get(a, 0) for a in CORE_ASPECTS],
        'pagerank': [pagerank.get(a, 0) for a in CORE_ASPECTS],
        'betweenness_centrality': [betweenness.get(a, 0) for a in CORE_ASPECTS],
        'closeness_centrality': [closeness.get(a, 0) for a in CORE_ASPECTS],
        'eigenvector_centrality': [eigenvector.get(a, 0) for a in CORE_ASPECTS],
        'clustering_coefficient': [clustering.get(a, 0) for a in CORE_ASPECTS],
        'eccentricity': [eccentricity.get(a, np.nan) for a in CORE_ASPECTS],
        'community': [partition.get(a, -1) for a in CORE_ASPECTS]
    })
    
    return node_df, partition, eccentricity

def compute_edge_metrics(G, precision_matrix):
    logger.info("Computing edge-level metrics...")
    
    edges = list(G.edges(data=True))
    edge_data = []
    
    max_weight = max([d['weight'] for u, v, d in edges]) if edges else 1.0
    total_weight = sum([d['weight'] for u, v, d in edges]) if edges else 1.0
    
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    bridges = list(nx.bridges(G))
    
    # Pre-calculate partial correlation for all pairs to match B1
    diag = np.diag(precision_matrix)
    d = np.sqrt(diag)
    
    for u, v, data in edges:
        raw_prec = precision_matrix.loc[u, v]
        raw_partial_corr = -raw_prec / (d[CORE_ASPECTS.index(u)] * d[CORE_ASPECTS.index(v)])
        
        weight = data['weight']
        edge_data.append({
            'node_1': u,
            'node_2': v,
            'raw_partial_correlation': raw_partial_corr,
            'weight': weight,
            'tie_strength': weight / max_weight,
            'normalized_weight': weight / total_weight,
            'edge_betweenness': edge_betweenness.get((u, v), edge_betweenness.get((v, u), 0)),
            'is_bridge': (u, v) in bridges or (v, u) in bridges
        })
        
    return pd.DataFrame(edge_data)

def compute_graph_metrics(G, node_df, edge_df, partition, eccentricity):
    logger.info("Computing graph-level metrics...")
    
    # C1. Density
    density = nx.density(G)
    
    # C2. Average Clustering
    avg_clustering = nx.average_clustering(G, weight='weight')
    
    # C3. Average Shortest Path
    dist_G = G.copy()
    for u, v, d in dist_G.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)
        
    if nx.is_connected(dist_G):
        avg_path_length = nx.average_shortest_path_length(dist_G, weight='distance')
        diameter = nx.diameter(dist_G, e=eccentricity)
    else:
        # Handle disconnected
        path_lengths = []
        diameters = []
        for component in nx.connected_components(dist_G):
            if len(component) > 1:
                subgraph = dist_G.subgraph(component)
                path_lengths.append(nx.average_shortest_path_length(subgraph, weight='distance'))
                # Need per-component eccentricity for diameter
                comp_ecc = {n: eccentricity[n] for n in component}
                diameters.append(nx.diameter(subgraph, e=comp_ecc))
        avg_path_length = np.mean(path_lengths) if path_lengths else np.nan
        diameter = max(diameters) if diameters else np.nan
        
    # C5. Modularity
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    # C6. Assortativity
    assortativity = nx.degree_assortativity_coefficient(G)
    
    # C7. Global Efficiency
    # Requirement: "Average inverse shortest path length"
    # nx.global_efficiency uses 1/d(u,v). We should ensure d(u,v) uses 'distance' (1/weight).
    # Since nx.global_efficiency doesn't accept a weight parameter for the distance calculation,
    # we compute it manually as the average inverse of weighted shortest path lengths.
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(dist_G, weight='distance'))
    inv_distances = []
    for u in shortest_paths:
        for v in shortest_paths[u]:
            if u != v and shortest_paths[u][v] > 0:
                inv_distances.append(1.0 / shortest_paths[u][v])
    global_efficiency = np.mean(inv_distances) if inv_distances else 0.0
    
    # C8. Small-World Coefficient (Sigma)
    logger.info("Computing small-world sigma (generating 1000 random graphs)...")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes > 1 and n_edges > 0:
        # Literal Requirement: Use weighted metrics for Sigma
        C_real = avg_clustering
        L_real = avg_path_length
        
        c_randoms = []
        l_randoms = []
        for _ in range(1000):
            # For random graphs, we preserve N and M. Since weight is not defined for standard random 
            # graphs in this context, we assign the mean edge weight of G to the random edges
            # to make the comparison meaningful.
            R = nx.gnm_random_graph(n_nodes, n_edges, seed=None)
            mean_weight = edge_df['weight'].mean()
            for u, v in R.edges():
                R[u][v]['weight'] = mean_weight
                R[u][v]['distance'] = 1.0 / (mean_weight + 1e-9)
            
            c_randoms.append(nx.average_clustering(R, weight='weight'))
            if nx.is_connected(R):
                l_randoms.append(nx.average_shortest_path_length(R, weight='distance'))
            else:
                # Per component if disconnected
                comp_lens = [nx.average_shortest_path_length(R.subgraph(c), weight='distance') 
                             for c in nx.connected_components(R) if len(c) > 1]
                if comp_lens: l_randoms.append(np.mean(comp_lens))
        
        C_random = np.mean(c_randoms)
        L_random = np.mean(l_randoms) if l_randoms else np.nan
        
        sigma = (C_real / C_random) / (L_real / L_random) if not np.isnan(L_random) and L_random != 0 else np.nan
    else:
        sigma = np.nan

    graph_metrics = {
        'density': density,
        'average_clustering': avg_clustering,
        'average_shortest_path_length': avg_path_length,
        'graph_diameter': diameter,
        'modularity': modularity,
        'assortativity': assortativity,
        'global_efficiency': global_efficiency,
        'small_world_sigma': sigma
    }
    
    return pd.DataFrame([graph_metrics])

def generate_visualizations(G, node_df, edge_df, precision_matrix, partition):
    logger.info("Generating visualizations...")
    
    # D1. Network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=RANDOM_STATE, k=1.5)
    
    node_sizes = [node_df.loc[node_df['node'] == n, 'pagerank'].values[0] * 10000 for n in G.nodes()]
    node_colors = [partition[n] for n in G.nodes()]
    
    edge_widths = [edge_df[((edge_df['node_1']==u) & (edge_df['node_2']==v)) | 
                           ((edge_df['node_1']==v) & (edge_df['node_2']==u))]['tie_strength'].values[0] * 5 for u, v in G.edges()]
    
    edge_colors = []
    for u, v in G.edges():
        corr = edge_df[((edge_df['node_1']==u) & (edge_df['node_2']==v)) | 
                       ((edge_df['node_1']==v) & (edge_df['node_2']==u))]['raw_partial_correlation'].values[0]
        edge_colors.append('blue' if corr > 0 else 'red')
        
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Set3, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("GLASSO Aspect Network\nNode size: PageRank, Color: Community, Edge: Tie Strength & Sign")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'network_graph.png'))
    plt.close()

    # D2. Centrality comparison bar chart
    metrics_to_plot = ['degree_centrality', 'weighted_degree', 'pagerank', 'betweenness_centrality', 'eigenvector_centrality']
    plot_df = node_df[['node'] + metrics_to_plot].copy()
    
    # Normalize to [0,1]
    for col in metrics_to_plot:
        if plot_df[col].max() != plot_df[col].min():
            plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())
        else:
            plot_df[col] = 1.0 if plot_df[col].max() > 0 else 0.0
            
    plot_df = plot_df.sort_values(by='pagerank', ascending=False)
    
    melted_df = plot_df.melt(id_vars='node', var_name='Metric', value_name='Normalized Value')
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='node', y='Normalized Value', hue='Metric')
    plt.xticks(rotation=45)
    plt.title("Centrality Metrics Comparison (Normalized [0,1])")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'centrality_comparison.png'))
    plt.close()

    # D3. Edge weight heatmap
    # Need to reconstruct signed partial correlation matrix
    diag = np.diag(precision_matrix)
    d = np.sqrt(diag)
    p_corr_matrix = pd.DataFrame(0.0, index=CORE_ASPECTS, columns=CORE_ASPECTS)
    for i, a1 in enumerate(CORE_ASPECTS):
        for j, a2 in enumerate(CORE_ASPECTS):
            if i == j:
                p_corr_matrix.loc[a1, a2] = np.nan
            else:
                p_corr_matrix.loc[a1, a2] = -precision_matrix.loc[a1, a2] / (d[i] * d[j])
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_corr_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0, square=True)
    plt.title("GLASSO Partial Correlation Network — Aspect Relationships")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'edge_heatmap.png'))
    plt.close()

    # D4. Community structure plot
    plt.figure(figsize=(12, 10))
    # Group by community
    # For a small graph (7 nodes), we can just use kamada_kawai which often highlights clusters
    pos_comm = nx.kamada_kawai_layout(G, weight='weight')
    
    nx.draw_networkx_nodes(G, pos_comm, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Set3, alpha=0.9)
    nx.draw_networkx_edges(G, pos_comm, width=edge_widths, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos_comm, font_size=10)
    
    # Add community labels
    for comm_id in set(partition.values()):
        nodes_in_comm = [n for n, c in partition.items() if c == comm_id]
        if nodes_in_comm:
            # Calculate centroid of nodes in this community
            x_coords = [pos_comm[n][0] for n in nodes_in_comm]
            y_coords = [pos_comm[n][1] for n in nodes_in_comm]
            plt.text(np.mean(x_coords), np.mean(y_coords) + 0.1, f"Comm {comm_id}", 
                     fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    plt.title("Community Structure (Nodes Arranged by Connection Strength)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'community_plot.png'))
    plt.close()

def save_run_summary(G, node_df, edge_df, graph_metrics, partition):
    logger.info("Saving run summary...")
    
    with open(os.path.join(OUTPUT_DIR, 'run_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("GLASSO GRAPH ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of nodes: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n\n")
        
        f.write("EDGES AND WEIGHTS:\n")
        sorted_edges = edge_df.sort_values(by='weight', ascending=False)
        for _, row in sorted_edges.iterrows():
            f.write(f"  {row['node_1']} <-> {row['node_2']}: {row['raw_partial_correlation']:.4f} (weight={row['weight']:.4f})\n")
        
        f.write("\nCOMMUNITY ASSIGNMENTS:\n")
        for comm_id in sorted(set(partition.values())):
            members = [n for n, c in partition.items() if c == comm_id]
            f.write(f"  Community {comm_id}: {', '.join(members)}\n")
            
        f.write("\nTOP 3 NODES BY PAGERANK:\n")
        top_nodes = node_df.sort_values(by='pagerank', ascending=False).head(3)
        for _, row in top_nodes.iterrows():
            f.write(f"  {row['node']}: {row['pagerank']:.4f}\n")
            
        sigma = graph_metrics['small_world_sigma'].values[0]
        f.write(f"\nSMALL-WORLD STRUCTURE: {'Detected' if sigma > 1 else 'Not Detected'} (Sigma = {sigma:.4f})\n")
        
        bridges = edge_df[edge_df['is_bridge'] == True]
        if not bridges.empty:
            f.write(f"\nBRIDGE EDGES FOUND ({len(bridges)}):\n")
            for _, row in bridges.iterrows():
                f.write(f"  {row['node_1']} <-> {row['node_2']}\n")
        else:
            f.write("\nNo bridge edges found.\n")
        f.write("="*60 + "\n")

def main():
    setup_output()
    
    # 1. Load data
    feature_matrix = load_full_data()
    
    # 2. Reconstruct network
    # For metrics, we use the existing function to ensure consistency
    logger.info("Constructing NetworkX graph...")
    G = construct_partial_correlation_network(feature_matrix)
    
    # Also get raw precision matrix for B1 and E4
    precision_matrix = estimate_precision_matrix(feature_matrix)
    precision_matrix.to_csv(os.path.join(OUTPUT_DIR, 'precision_matrix.csv'))
    
    # 3. Compute Metrics
    node_df, partition, eccentricity = compute_node_metrics(G)
    node_df.to_csv(os.path.join(OUTPUT_DIR, 'node_metrics.csv'), index=False)
    
    edge_df = compute_edge_metrics(G, precision_matrix)
    edge_df.to_csv(os.path.join(OUTPUT_DIR, 'edge_metrics.csv'), index=False)
    
    graph_metrics_df = compute_graph_metrics(G, node_df, edge_df, partition, eccentricity)
    graph_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'graph_metrics.csv'), index=False)
    
    # 4. Visualizations
    generate_visualizations(G, node_df, edge_df, precision_matrix, partition)
    
    # 5. Run Summary
    save_run_summary(G, node_df, edge_df, graph_metrics_df, partition)
    
    logger.info("Graph analysis complete. All outputs saved to 'graph_analysis/' folder.")

if __name__ == "__main__":
    main()
