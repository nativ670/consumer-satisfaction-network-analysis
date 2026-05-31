import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import ast
import logging
import json
from src.network_builder import construct_partial_correlation_network
from src.network_builder_ebic import construct_partial_correlation_network_ebic
from src.modeling import prepare_raw_modeling_data
from src.nlp_extraction import CORE_ASPECTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_networks():
    # 1. Load data
    data_path = 'data/Seminar_Amazon_Results_FULL.csv'
    if not os.path.exists(data_path):
        logger.error(f"Error: {data_path} not found. Ensure the NLP extraction has been run and saved.")
        return

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    logger.info("Parsing aspect_sentiments...")
    df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    
    logger.info("Applying modeling sparsity filter...")
    # This returns a pivoted feature matrix with only reviews that have at least one aspect
    # and includes 'rating' column (which we'll ignore for network building)
    filtered_data = prepare_raw_modeling_data(df)
    feature_matrix = filtered_data[CORE_ASPECTS]
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    # 2. Construct networks
    logger.info("Constructing network using CV method...")
    # We need to capture the lambda if possible, but GraphicalLassoCV doesn't easily expose it 
    # unless we check model.alpha_
    # Wait, the current construct_partial_correlation_network doesn't return the model.
    # I might need to slightly modify it or just accept I won't have the lambda easily 
    # unless I re-implement the CV call here.
    # Actually, I'll just run it as is.
    G_cv = construct_partial_correlation_network(feature_matrix)
    
    logger.info("Constructing network using EBIC method...")
    G_ebic, selection_info = construct_partial_correlation_network_ebic(feature_matrix)
    
    # 3. Compare networks
    comparison_dir = 'network_comparison'
    os.makedirs(comparison_dir, exist_ok=True)
    
    cv_edges = set()
    for u, v, d in G_cv.edges(data=True):
        cv_edges.add(tuple(sorted((u, v))))
        
    ebic_edges = set()
    for u, v, d in G_ebic.edges(data=True):
        ebic_edges.add(tuple(sorted((u, v))))
        
    dropped_edges = cv_edges - ebic_edges
    added_edges = ebic_edges - cv_edges
    shared_edges = cv_edges & ebic_edges
    
    # Edge weights for shared edges
    weights_cv = []
    weights_ebic = []
    for u, v in shared_edges:
        w_cv = G_cv[u][v]['partial_correlation']
        w_ebic = G_ebic[u][v]['partial_correlation']
        weights_cv.append(w_cv)
        weights_ebic.append(w_ebic)
        
    correlation = np.corrcoef(weights_cv, weights_ebic)[0, 1] if len(shared_edges) > 1 else 1.0
    
    overlap_pct = (len(shared_edges) / len(cv_edges | ebic_edges)) * 100 if (cv_edges | ebic_edges) else 100
    conclusion = "identical" if cv_edges == ebic_edges and np.allclose(weights_cv, weights_ebic) else \
                 "nearly identical (>80% edge overlap)" if overlap_pct > 80 else \
                 "substantially different (<80% overlap)"
                 
    # 4. Save Report
    report_path = os.path.join(comparison_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("NETWORK COMPARISON REPORT: CV vs EBIC\n")
        f.write("="*40 + "\n\n")
        f.write(f"Number of observations (N): {len(feature_matrix)}\n")
        f.write(f"EBIC Selected Lambda: {selection_info['best_lambda']}\n")
        f.write(f"EBIC value: {selection_info['best_ebic']}\n\n")
        
        f.write(f"CV Network Edges: {len(cv_edges)}\n")
        f.write(f"EBIC Network Edges: {len(ebic_edges)}\n\n")
        
        f.write("CV NETWORK EDGES (u, v, weight):\n")
        for u, v, d in sorted(G_cv.edges(data=True), key=lambda x: abs(x[2]['partial_correlation']), reverse=True):
            f.write(f"  {u} <-> {v}: {d['partial_correlation']}\n")
        f.write("\n")
        
        f.write("EBIC NETWORK EDGES (u, v, weight):\n")
        for u, v, d in sorted(G_ebic.edges(data=True), key=lambda x: abs(x[2]['partial_correlation']), reverse=True):
            f.write(f"  {u} <-> {v}: {d['partial_correlation']}\n")
        f.write("\n")
        
        f.write(f"DROPPED EDGES (Present in CV, not EBIC): {len(dropped_edges)}\n")
        for u, v in dropped_edges:
            f.write(f"  {u} <-> {v}\n")
        f.write("\n")
        
        f.write(f"ADDED EDGES (Present in EBIC, not CV): {len(added_edges)}\n")
        for u, v in added_edges:
            f.write(f"  {u} <-> {v}\n")
        f.write("\n")
        
        f.write(f"SHARED EDGES: {len(shared_edges)}\n")
        for u, v in shared_edges:
            w_cv = G_cv[u][v]['partial_correlation']
            w_ebic = G_ebic[u][v]['partial_correlation']
            f.write(f"  {u} <-> {v}: CV={w_cv}, EBIC={w_ebic}\n")
        f.write("\n")
        
        f.write(f"Edge Weight Correlation (Shared Edges): {correlation:.4f}\n")
        f.write(f"Edge Overlap: {overlap_pct:.2f}%\n")
        f.write(f"Conclusion: {conclusion}\n")

    logger.info(f"Comparison report saved to {report_path}")
    
    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Fix positions
    pos = nx.circular_layout(CORE_ASPECTS)
    
    def draw_network(G, ax, title):
        # Nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgrey', node_size=1000)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        # Edges
        if G.number_of_edges() > 0:
            edges = G.edges(data=True)
            colors = ['blue' if d['partial_correlation'] > 0 else 'red' for u, v, d in edges]
            weights = [abs(d['partial_correlation']) * 10 for u, v, d in edges]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=colors, width=weights, alpha=0.6)
            
        ax.set_title(title)
        ax.axis('off')

    draw_network(G_cv, axes[0], "CV Method (GraphicalLassoCV)")
    draw_network(G_ebic, axes[1], f"EBIC Method (lambda={selection_info['best_lambda']:.4f})")
    
    plt.tight_layout()
    viz_path = os.path.join(comparison_dir, 'network_comparison.png')
    plt.savefig(viz_path)
    logger.info(f"Visualization saved to {viz_path}")
    
if __name__ == "__main__":
    compare_networks()
