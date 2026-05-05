import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.preprocessing import StandardScaler
import logging
from src.nlp_extraction import CORE_ASPECTS, extract_aspects_and_sentiments
from src.data_preprocessing import load_and_clean_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pivot_aspect_sentiments(df):
    """
    Pivots the 'aspect_sentiments' column into a feature matrix.
    Rows = Reviews, Columns = CORE_ASPECTS.
    Missing mentions are filled with 0.0 (neutral).
    
    Args:
        df (pd.DataFrame): DataFrame with 'aspect_sentiments' column.
        
    Returns:
        pd.DataFrame: Feature matrix of shape (n_reviews, n_core_aspects).
    """
    if df.empty or 'aspect_sentiments' not in df.columns:
        return pd.DataFrame(0.0, index=df.index, columns=CORE_ASPECTS)

    # Explode the list of (aspect, sentiment) tuples
    exploded = df[['aspect_sentiments']].explode('aspect_sentiments')
    
    # Filter out empty rows (reviews with no aspects)
    exploded = exploded.dropna(subset=['aspect_sentiments'])
    
    if exploded.empty:
        return pd.DataFrame(0.0, index=df.index, columns=CORE_ASPECTS)
    
    # Split tuples into separate columns
    aspect_sent = pd.DataFrame(exploded['aspect_sentiments'].tolist(), index=exploded.index)
    aspect_sent.columns = ['aspect', 'sentiment']
    
    # Group by index and aspect, average the sentiment
    pivoted = aspect_sent.groupby([aspect_sent.index, 'aspect'])['sentiment'].mean().unstack(fill_value=0.0)
    
    # Ensure all CORE_ASPECTS are present and in the correct order
    feature_matrix = pivoted.reindex(columns=CORE_ASPECTS, fill_value=0.0)
    
    # Reindex to original dataframe index to include reviews with no aspects
    feature_matrix = feature_matrix.reindex(df.index, fill_value=0.0)
    
    return feature_matrix

def construct_partial_correlation_network(feature_matrix, threshold=0.02):
    """
    Estimates the partial correlation network using Graphical Lasso.
    
    Args:
        feature_matrix (pd.DataFrame): The pivoted sentiment matrix.
        threshold (float): Minimum absolute partial correlation to create an edge.
        
    Returns:
        nx.Graph: A NetworkX graph representing the partial correlation network.
    """
    # Identify aspects that have at least some variation
    # GraphicalLasso needs some variation to work correctly
    active_aspects = [a for a in feature_matrix.columns if feature_matrix[a].std() > 0]
    inactive_aspects = [a for a in feature_matrix.columns if a not in active_aspects]
    
    G = nx.Graph()
    
    # Add all nodes initially
    for aspect in CORE_ASPECTS:
        series = feature_matrix[aspect]
        mentions = series[series != 0]
        frequency = len(mentions)
        avg_sentiment = mentions.mean() if frequency > 0 else 0.0
        G.add_node(aspect, avg_sentiment=avg_sentiment, frequency=frequency)

    if not active_aspects:
        logger.warning("No active aspects found with variation. Returning isolated nodes.")
        return G

    X = feature_matrix[active_aspects]
    
    logger.info(f"Standardizing feature matrix for {len(active_aspects)} active aspects...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info("Estimating precision matrix using GraphicalLasso...")
    # GraphicalLassoCV is preferred but can be unstable if N is small or data is very sparse.
    # We use GraphicalLasso with a small alpha as a fallback or if CV is too slow.
    try:
        # Try CV first for optimal alpha
        model = GraphicalLassoCV(max_iter=500)
        model.fit(X_scaled)
    except Exception as e:
        logger.warning(f"GraphicalLassoCV failed: {e}. Falling back to standard GraphicalLasso.")
        model = GraphicalLasso(alpha=0.1, max_iter=500)
        model.fit(X_scaled)
    
    precision_matrix = model.precision_
    
    # Convert precision matrix to partial correlation matrix
    # rho_ij = -k_ij / sqrt(k_ii * k_jj)
    diag_indices = np.diag_indices_from(precision_matrix)
    d = np.sqrt(precision_matrix[diag_indices])
    # Avoid division by zero
    d[d == 0] = 1.0
    
    partial_corr = -precision_matrix / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    
    # Add edges for active aspects
    for i in range(len(active_aspects)):
        for j in range(i + 1, len(active_aspects)):
            p_corr = partial_corr[i, j]
            if abs(p_corr) > threshold:
                G.add_edge(active_aspects[i], active_aspects[j], 
                           weight=abs(p_corr), 
                           partial_correlation=round(p_corr, 4),
                           sign='positive' if p_corr > 0 else 'negative')
                
    return G

def calculate_network_metrics(G):
    """
    Calculates centrality metrics for the graph.
    
    Args:
        G (nx.Graph): The NetworkX graph.
        
    Returns:
        dict: A dictionary of metrics for each node.
    """
    # Degree Centrality (Weighted - "Strength")
    strength = dict(G.degree(weight='weight'))
    
    # Betweenness Centrality (using 1/weight as distance for meaningful metrics)
    # If partial correlation is high, distance is low.
    distance_G = G.copy()
    for u, v, d in distance_G.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-6)
    
    betweenness = nx.betweenness_centrality(distance_G, weight='distance', normalized=True)
    
    # Eigenvector Centrality
    try:
        eigenvector = nx.pagerank(G, weight='weight')
    except Exception as e:
        logger.warning(f"Eigenvector centrality calculation failed: {e}")
        eigenvector = {node: 0.0 for node in G.nodes()}
        
    return {
        'degree': strength,
        'betweenness': betweenness,
        'eigenvector': eigenvector
    }

def build_and_analyze_network(df):
    """
    Orchestrates the network construction and analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with 'aspect_sentiments'.
        
    Returns:
        dict: Graph, metrics, and sorted edges.
    """
    feature_matrix = pivot_aspect_sentiments(df)
    G = construct_partial_correlation_network(feature_matrix)
    metrics = calculate_network_metrics(G)
    
    # Prepare sorted edges list
    edges_list = []
    for u, v, data in G.edges(data=True):
        edges_list.append((u, v, data['partial_correlation']))
    
    sorted_edges = sorted(edges_list, key=lambda x: abs(x[2]), reverse=True)
    
    return {
        'graph': G,
        'metrics': metrics,
        'sorted_edges': sorted_edges
    }

if __name__ == '__main__':
    try:
        # Load sample data (1500 rows)
        logger.info("Loading 1500 rows for network analysis verification...")
        data = load_and_clean_data().head(1500)
        
        # Run NLP extraction (This might take a while depending on hardware)
        logger.info("Running NLP Aspect Extraction and Sentiment Analysis...")
        processed_data = extract_aspects_and_sentiments(data)
        
        # Build network
        logger.info("Constructing Partial Correlation Network...")
        results = build_and_analyze_network(processed_data)
        G = results['graph']
        metrics = results['metrics']
        sorted_edges = results['sorted_edges']
        
        # Print Summary
        print("\n" + "="*60)
        print("CONSUMER SATISFACTION NETWORK: PARTIAL CORRELATION ANALYSIS")
        print("="*60)
        
        print("\nNODES (Aspects):")
        header = f"{'Core Aspect':<25} | {'Freq':<5} | {'Avg Sent':<10} | {'Eigen Centrality':<15}"
        print(header)
        print("-" * len(header))
        for node in G.nodes(data=True):
            aspect = node[0]
            attr = node[1]
            ec = metrics['eigenvector'].get(aspect, 0.0)
            print(f"{aspect:<25} | {attr['frequency']:<5} | {attr['avg_sentiment']:<10.4f} | {ec:<15.4f}")
            
        print("\nTOP 5 EDGES (Cognitive Trade-offs/Interactions):")
        print(f"{'Interaction (Edge)':<45} | {'Partial Corr':<15}")
        print("-" * 65)
        for u, v, weight in sorted_edges[:5]:
            edge_label = f"{u} <-> {v}"
            print(f"{edge_label:<45} | {weight:<15.4f}")
            
        print("\nInterpretation Hint: A positive partial correlation indicates features ")
        print("that vary together after controlling for other aspects. A negative correlation ")
        print("suggests a trade-off in the consumer's mental model.")
        print("="*60)
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
