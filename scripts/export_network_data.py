import pandas as pd
import numpy as np
import ast
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

from src.network_builder import build_and_analyze_network, construct_partial_correlation_network, calculate_network_metrics
from src.modeling import prepare_raw_modeling_data

def export_network_data():
    data_path = 'data/Seminar_Amazon_Results_FULL.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print("Loading full dataset...")
    df = pd.read_csv(data_path)
    
    print("Parsing aspect_sentiments...")
    # aspect_sentiments is stored as a string representation of a list of tuples
    df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    
    print("Applying modeling sparsity filter (dropping empty reviews)...")
    # This prepares the pivoted feature matrix AND drops sparse rows
    filtered_data = prepare_raw_modeling_data(df)
    
    print("Building and analyzing network (GLasso) on filtered data...")
    # We pass the raw filtered dataframe to build_and_analyze_network 
    # BUT wait, build_and_analyze_network expects a dataframe with 'aspect_sentiments' 
    # to pivot it. Since we already pivoted it in prepare_raw_modeling_data, 
    # we should construct the network directly.
    
    # Extract only the aspect columns
    aspect_cols = [c for c in filtered_data.columns if c != 'rating']
    feature_matrix = filtered_data[aspect_cols]
    
    G = construct_partial_correlation_network(feature_matrix)
    metrics = calculate_network_metrics(G)
    
    # Prepare Nodes Data for D3
    # We use Eigenvector Centrality (stored in metrics['eigenvector']) as pageRank proxy
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            "id": node,
            "pageRank": float(round(metrics['eigenvector'].get(node, 0.0), 4))
        })
        
    # Prepare Links Data for D3
    links_data = []
    for u, v, data in G.edges(data=True):
        links_data.append({
            "source": u,
            "target": v,
            "value": float(round(abs(data['partial_correlation']), 4))
        })
        
    # Output to visualizations/network_data.js
    output_path = 'visualizations/network_data.js'
    with open(output_path, 'w') as f:
        f.write(f"const nodesData = {json.dumps(nodes_data, indent=4)};\n")
        f.write(f"const linksData = {json.dumps(links_data, indent=4)};\n")
        
    print(f"Successfully exported network data to {output_path}")
    print(f"Generated {len(links_data)} edges for the visualization.")
    
    # Print ranges for manual scale adjustment if needed
    pr_vals = [n['pageRank'] for n in nodes_data]
    link_vals = [l['value'] for l in links_data]
    print(f"PageRank range: {min(pr_vals)} - {max(pr_vals)}")
    if links_data:
        print(f"Link Value range: {min(link_vals)} - {max(link_vals)}")
    else:
        print("Link Value range: No edges found.")

if __name__ == "__main__":
    export_network_data()
