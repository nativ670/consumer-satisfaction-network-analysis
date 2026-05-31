import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import graphical_lasso
from sklearn.preprocessing import StandardScaler
import logging
from src.nlp_extraction import CORE_ASPECTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def construct_partial_correlation_network_ebic(feature_matrix, threshold=0.02, gamma=0.5):
    """
    Estimates the partial correlation network using Graphical Lasso with EBIC model selection.
    
    Args:
        feature_matrix (pd.DataFrame): The pivoted sentiment matrix.
        threshold (float): Minimum absolute partial correlation to create an edge.
        gamma (float): EBIC hyperparameter (default 0.5).
        
    Returns:
        nx.Graph: A NetworkX graph representing the partial correlation network.
        dict: Selection info (best_lambda, best_ebic).
    """
    # Identify aspects that have at least some variation
    active_aspects = [a for a in feature_matrix.columns if feature_matrix[a].std() > 0]
    
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
        return G, {"best_lambda": None, "best_ebic": None}

    X = feature_matrix[active_aspects]
    N = X.shape[0]
    P = len(active_aspects)
    
    logger.info(f"Standardizing feature matrix for {P} active aspects (N={N})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute empirical covariance matrix
    S = np.cov(X_scaled.T, bias=True)
    
    # Test 100 lambda values between 1e-3 and 1.0 (log-spaced)
    lambdas = np.logspace(-3, 0, 100)
    
    best_ebic = float('inf')
    best_lambda = None
    best_precision = None
    
    logger.info(f"Starting EBIC selection over 100 lambda values (gamma={gamma})...")
    
    for lam in lambdas:
        try:
            # graphical_lasso returns (covariance, precision)
            _, precision = graphical_lasso(S, alpha=lam, max_iter=500)
            
            # Number of non-zero edges in upper triangle (abs > 1e-10)
            # Exclude diagonal
            E = (np.sum(np.abs(precision) > 1e-10) - P) / 2
            
            # Log-likelihood (per sample, up to a constant)
            # LL = logdet(precision) - trace(S @ precision)
            sign, logdet = np.linalg.slogdet(precision)
            if sign <= 0:
                # Precision matrix should be positive definite
                continue
                
            ll = logdet - np.trace(S @ precision)
            
            # EBIC = -2*LL_total + E*log(N) + 4*gamma*E*log(P)
            # LL_total = (N/2) * ll
            ebic = -N * ll + E * np.log(N) + 4 * gamma * E * np.log(P)
            
            if ebic < best_ebic:
                best_ebic = ebic
                best_lambda = lam
                best_precision = precision
                
        except Exception as e:
            logger.debug(f"Graphical Lasso failed for lambda={lam}: {e}")
            continue

    # Handle edge case: if EBIC selects an empty network (no edges)
    # The requirement says: log a warning and fall back to the least sparse lambda that produces at least one edge
    if best_precision is not None:
        E_best = (np.sum(np.abs(best_precision) > 1e-10) - P) / 2
        if E_best == 0:
            logger.warning("EBIC selected an empty network. Falling back to the least sparse lambda with at least one edge.")
            # Search for the smallest lambda (least sparse) that gives at least one edge
            fallback_lambda = None
            fallback_precision = None
            for lam in lambdas: # From 1e-3 up to 1.0
                try:
                    _, prec = graphical_lasso(S, alpha=lam, max_iter=500)
                    E = (np.sum(np.abs(prec) > 1e-10) - P) / 2
                    if E > 0:
                        fallback_lambda = lam
                        fallback_precision = prec
                        break
                except:
                    continue
            
            if fallback_precision is not None:
                best_lambda = fallback_lambda
                best_precision = fallback_precision
                # Re-calculate EBIC for the fallback if needed, but the requirement just says log and fallback
                sign, logdet = np.linalg.slogdet(best_precision)
                ll = logdet - np.trace(S @ best_precision)
                E = (np.sum(np.abs(best_precision) > 1e-10) - P) / 2
                best_ebic = -N * ll + E * np.log(N) + 4 * gamma * E * np.log(P)
            else:
                logger.error("Could not find any lambda that produces a non-empty network.")

    if best_precision is None:
        logger.error("EBIC selection failed to find a valid precision matrix.")
        return G, {"best_lambda": None, "best_ebic": None}

    logger.info(f"Selected lambda: {best_lambda:.6f} with EBIC: {best_ebic:.4f}")
    
    # Convert precision matrix to partial correlation matrix
    diag_indices = np.diag_indices_from(best_precision)
    d = np.sqrt(best_precision[diag_indices])
    d[d == 0] = 1.0
    
    partial_corr = -best_precision / np.outer(d, d)
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
                
    return G, {"best_lambda": best_lambda, "best_ebic": best_ebic}
