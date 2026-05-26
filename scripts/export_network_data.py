import pandas as pd
import numpy as np
import ast
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

from src.network_builder import build_and_analyze_network

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
    
    print("Building and analyzing network (GLasso)...")
    results = build_and_analyze_network(df)
    
    G = results['graph']
    metrics = results['metrics']
    
    # Prepare Nodes Data for D3
    # We use Eigenvector Centrality (stored in metrics['eigenvector']) as pageRank proxy
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            "id": node,
            "pageRank": float(round(metrics['eigenvector'][node], 4))
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
    
    # Print ranges for manual scale adjustment if needed
    pr_vals = [n['pageRank'] for n in nodes_data]
    link_vals = [l['value'] for l in links_data]
    print(f"PageRank range: {min(pr_vals)} - {max(pr_vals)}")
    print(f"Link Value range: {min(link_vals)} - {max(link_vals)}")

if __name__ == "__main__":
    export_network_data()
