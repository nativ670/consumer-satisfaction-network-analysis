import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import json
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import graphical_lasso

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network_builder import pivot_aspect_sentiments, CORE_ASPECTS

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def select_best_precision_ebic_custom(S, N, P, lambdas, gamma=0.5):
    best_ebic = float('inf')
    best_lambda = None
    best_precision = None
    
    for lam in lambdas:
        try:
            _, precision = graphical_lasso(S, alpha=lam, max_iter=500)
            E = (np.sum(np.abs(precision) > 1e-10) - P) / 2
            sign, logdet = np.linalg.slogdet(precision)
            if sign <= 0: continue
                
            ll = logdet - np.trace(S @ precision)
            ebic = -N * ll + E * np.log(N) + 4 * gamma * E * np.log(P)
            
            if ebic < best_ebic:
                best_ebic = ebic
                best_lambda = lam
                best_precision = precision
        except:
            continue
            
    if best_precision is not None:
        E_best = (np.sum(np.abs(best_precision) > 1e-10) - P) / 2
        if E_best == 0:
            for lam in lambdas:
                try:
                    _, prec = graphical_lasso(S, alpha=lam, max_iter=500)
                    if (np.sum(np.abs(prec) > 1e-10) - P) / 2 > 0:
                        best_lambda, best_precision = lam, prec
                        break
                except: continue
                
    return best_precision, best_lambda

def build_network_custom(feature_matrix, lambdas, threshold):
    active_aspects = [a for a in feature_matrix.columns if feature_matrix[a].std() > 0]
    
    G = nx.Graph()
    for aspect in CORE_ASPECTS:
        series = feature_matrix[aspect]
        mentions = series[series != 0]
        frequency = len(mentions)
        avg_sentiment = mentions.mean() if frequency > 0 else 0.0
        G.add_node(aspect, avg_sentiment=avg_sentiment, frequency=frequency)

    if not active_aspects:
        return G, None, None

    X = feature_matrix[active_aspects]
    N, P = X.shape
    X_scaled = StandardScaler().fit_transform(X)
    S = np.cov(X_scaled.T, bias=True)
    
    best_precision, best_lambda = select_best_precision_ebic_custom(S, N, P, lambdas=lambdas)

    if best_precision is None:
        return G, best_lambda, None
        
    diag_indices = np.diag_indices_from(best_precision)
    d = np.sqrt(best_precision[diag_indices])
    d[d == 0] = 1.0
    partial_corr = -best_precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    
    for i in range(len(active_aspects)):
        for j in range(i + 1, len(active_aspects)):
            p_corr = partial_corr[i, j]
            if abs(p_corr) > threshold:
                G.add_edge(active_aspects[i], active_aspects[j], 
                           weight=abs(p_corr), 
                           partial_correlation=round(p_corr, 4),
                           sign='positive' if p_corr > 0 else 'negative')
    return G, best_lambda, best_precision

def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'glasso_test_graphs_output'))
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Seminar_Amazon_Results_FULL.csv'))
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info("Parsing aspect_sentiments...")
    # Handle NaNs and evaluate strings
    df['aspect_sentiments'] = df['aspect_sentiments'].fillna('[]')
    df['aspect_sentiments'] = df['aspect_sentiments'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    logger.info("Pivoting feature matrix...")
    feature_matrix = pivot_aspect_sentiments(df)
    
    tests = [
        {"name": "test_1", "lambdas": np.logspace(-4, 0, 100), "threshold": 0.02, "desc": "lambda 0.0001 to 1; threshold 0.02"},
        {"name": "test_2", "lambdas": np.logspace(-3, 0, 100), "threshold": 0.0, "desc": "lambda 0.001 to 1; threshold 0"},
        {"name": "test_3", "lambdas": np.logspace(-4, 0, 100), "threshold": 0.0, "desc": "lambda 0.0001 to 1; threshold 0"}
    ]
    
    for t in tests:
        logger.info(f"Running {t['name']}: {t['desc']}")
        G, best_lambda, best_precision = build_network_custom(feature_matrix, t['lambdas'], t['threshold'])
        
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "partial_correlation": data['partial_correlation'],
                "weight": data['weight'],
                "sign": data['sign']
            })
            
        nodes = []
        for n, data in G.nodes(data=True):
            nodes.append({
                "aspect": n,
                "avg_sentiment": data['avg_sentiment'],
                "frequency": data['frequency']
            })
            
        results = {
            "test_name": t['name'],
            "description": t['desc'],
            "best_lambda": float(best_lambda) if best_lambda is not None else None,
            "num_edges": len(edges),
            "edges": edges,
            "nodes": nodes
        }
        
        output_file = os.path.join(output_dir, f"{t['name']}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved {t['name']} to {output_file}")

if __name__ == '__main__':
    main()
