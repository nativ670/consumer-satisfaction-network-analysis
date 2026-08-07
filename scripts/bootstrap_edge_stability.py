import os
import sys
import ast
import time
import argparse
import logging
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.nlp_extraction import CORE_ASPECTS
from src.modeling import prepare_raw_modeling_data
from src.network_builder import select_best_precision_ebic, construct_partial_correlation_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    DATA_PATH = "data/Seminar_Amazon_Results_FULL.csv"
    logger.info("Loading dataset from %s ...", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error(f"File not found: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    if isinstance(df["aspect_sentiments"].iloc[0], str):
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(ast.literal_eval)
    return df

def run_bootstrap_iteration(iteration_index, df_features, base_seed, lambda_grid_size, active_aspects):
    # Deterministic rng per iteration for reproducibility regardless of n_jobs
    rng = np.random.default_rng(base_seed + iteration_index)
    N = len(df_features)
    
    # Resample with replacement
    indices = rng.choice(N, size=N, replace=True)
    df_boot = df_features.iloc[indices]
    
    # Determine active aspects for this bootstrap sample
    boot_active = [a for a in active_aspects if df_boot[a].std() > 0]
    
    # Initialize dictionary for all 21 pairs to 0.0
    all_pairs = list(itertools.combinations(sorted(CORE_ASPECTS), 2))
    pcorr_dict = {pair: 0.0 for pair in all_pairs}
    
    if not boot_active:
        return pcorr_dict
        
    X = df_boot[boot_active]
    N_active, P = X.shape
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Empirical covariance
    S = np.cov(X_scaled.T, bias=True)
    
    # EBIC lambda selection
    lambdas = np.logspace(-3, 0, lambda_grid_size)
    best_precision, _ = select_best_precision_ebic(S, N_active, P, lambdas=lambdas, gamma=0.5)
    
    if best_precision is None:
        return pcorr_dict
        
    # Convert precision matrix to partial correlation matrix
    diag_indices = np.diag_indices_from(best_precision)
    d = np.sqrt(best_precision[diag_indices])
    d[d == 0] = 1.0
    partial_corr = -best_precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    
    # Record values
    for i in range(len(boot_active)):
        for j in range(i + 1, len(boot_active)):
            a, b = boot_active[i], boot_active[j]
            pair = tuple(sorted([a, b]))
            pcorr_dict[pair] = partial_corr[i, j]
            
    return pcorr_dict

def run_bootstrap_worker(i, df_features, base_seed, lambda_grid_size, active_aspects):
    if i % 50 == 0 and i > 0:
        logger.info(f"Completed {i} bootstrap iterations...")
    return run_bootstrap_iteration(i, df_features, base_seed, lambda_grid_size, active_aspects)

def main():
    parser = argparse.ArgumentParser(description="Bootstrap edge stability for GLASSO network.")
    parser.add_argument("--B", type=int, default=1000, help="Number of bootstrap iterations.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers.")
    parser.add_argument("--lambda-grid-size", type=int, default=50, help="Size of lambda grid for EBIC.")
    args = parser.parse_args()
    
    OUTPUT_DIR = "bootstrap_stability"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load and prepare data
    df_raw = load_data()
    df_features = prepare_raw_modeling_data(df_raw)
    
    active_aspects = [a for a in CORE_ASPECTS if df_features[a].std() > 0]
    
    # 2. Get Original Network to determine the "11 current edges" and their signs
    logger.info("Building original network on full sample to identify true edges...")
    G_orig = construct_partial_correlation_network(df_features[CORE_ASPECTS], threshold=0.02)
    
    original_edges = {}
    for u, v, data in G_orig.edges(data=True):
        pair = tuple(sorted([u, v]))
        original_edges[pair] = np.sign(data['partial_correlation'])
    logger.info(f"Original network has {len(original_edges)} edges.")
    
    # 3. Bootstrap Resampling
    logger.info(f"Starting {args.B} bootstrap iterations with n_jobs={args.n_jobs}...")
    start_time = time.time()
    
    results = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(run_bootstrap_worker)(i, df_features, args.base_seed, args.lambda_grid_size, active_aspects) 
        for i in range(args.B)
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Bootstrap finished in {elapsed:.2f} seconds.")
    
    # 4. Compile results
    all_pairs = list(itertools.combinations(sorted(CORE_ASPECTS), 2))
    
    records = []
    for row_idx, res_dict in enumerate(results):
        row = {"bootstrap_iter": row_idx}
        for pair in all_pairs:
            col_name = f"{pair[0]}__{pair[1]}"
            row[col_name] = res_dict[pair]
        records.append(row)
        
    df_raw_pcorrs = pd.DataFrame(records)
    raw_pcorrs_path = os.path.join(OUTPUT_DIR, "bootstrap_raw_partial_corrs.csv")
    df_raw_pcorrs.to_csv(raw_pcorrs_path, index=False)
    
    # Compute stability statistics
    stats_records = []
    threshold = 0.02
    
    for pair in all_pairs:
        col_name = f"{pair[0]}__{pair[1]}"
        pcorrs = df_raw_pcorrs[col_name].values
        
        in_original = pair in original_edges
        orig_sign = original_edges.get(pair, np.nan)
        
        inclusion_frequency = np.mean(np.abs(pcorrs) > threshold)
        mean_pcorr = np.mean(pcorrs)
        std_pcorr = np.std(pcorrs)
        ci_lower = np.percentile(pcorrs, 2.5)
        ci_upper = np.percentile(pcorrs, 97.5)
        
        if in_original:
            sign_consistency = np.mean(np.sign(pcorrs) == orig_sign)
        else:
            sign_consistency = np.nan
            
        stats_records.append({
            "node_1": pair[0],
            "node_2": pair[1],
            "in_original_network": in_original,
            "inclusion_frequency": inclusion_frequency,
            "mean_partial_corr": mean_pcorr,
            "std_partial_corr": std_pcorr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "sign_consistency": sign_consistency
        })
        
    df_stats = pd.DataFrame(stats_records)
    df_stats = df_stats.sort_values("inclusion_frequency", ascending=False)
    
    stats_path = os.path.join(OUTPUT_DIR, "edge_stability.csv")
    df_stats.to_csv(stats_path, index=False)
    
    # Generate run_summary.txt
    stable_edges = df_stats[(df_stats['in_original_network'] == True) & (df_stats['inclusion_frequency'] >= 0.90)]
    unstable_edges = df_stats[(df_stats['in_original_network'] == True) & (df_stats['inclusion_frequency'] < 0.90)]
    near_threshold = df_stats[(df_stats['in_original_network'] == False) & (df_stats['inclusion_frequency'] >= 0.50)]
    
    summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write("BOOTSTRAP EDGE STABILITY SUMMARY\n")
        f.write("================================\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Bootstrap iterations (B): {args.B}\n")
        f.write(f"Base seed: {args.base_seed}\n")
        f.write(f"N per bootstrap: {len(df_features)}\n")
        f.write(f"Lambda grid size: {args.lambda_grid_size}\n\n")
        
        f.write("STABLE EDGES (inclusion >= 0.90)\n")
        f.write("-" * 40 + "\n")
        for _, row in stable_edges.iterrows():
            f.write(f"{row['node_1']} <-> {row['node_2']}: freq={row['inclusion_frequency']:.3f}, mean_pcorr={row['mean_partial_corr']:.4f}\n")
            
        f.write("\nUNSTABLE EDGES (in original network but inclusion < 0.90)\n")
        f.write("-" * 60 + "\n")
        for _, row in unstable_edges.iterrows():
            f.write(f"{row['node_1']} <-> {row['node_2']}: freq={row['inclusion_frequency']:.3f}, mean_pcorr={row['mean_partial_corr']:.4f}\n")
            
        f.write("\nNEAR-THRESHOLD EDGES (not in original network but inclusion >= 0.50)\n")
        f.write("-" * 70 + "\n")
        for _, row in near_threshold.iterrows():
            f.write(f"{row['node_1']} <-> {row['node_2']}: freq={row['inclusion_frequency']:.3f}, mean_pcorr={row['mean_partial_corr']:.4f}\n")
            
    # Generate edge_stability_plot.png
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 10})
    
    df_plot = df_stats.sort_values("inclusion_frequency", ascending=True).copy()
    
    labels = [f"{row['node_1']} - {row['node_2']}" for _, row in df_plot.iterrows()]
    freqs = df_plot['inclusion_frequency'].values
    colors = ['#1f77b4' if row['in_original_network'] else '#cccccc' for _, row in df_plot.iterrows()]
    
    plt.barh(labels, freqs, color=colors)
    plt.axvline(x=0.90, color='red', linestyle='--', alpha=0.7, label='90% Stability Threshold')
    plt.axvline(x=0.50, color='orange', linestyle=':', alpha=0.7, label='50% Threshold')
    
    plt.xlabel('Inclusion Frequency (|partial_corr| > 0.02)')
    plt.title('Bootstrap Edge Stability (Nonparametric)')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Selected in Original Network'),
        Patch(facecolor='#cccccc', label='Not Selected in Original')
    ]
    plt.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='90% Stability Threshold'),
        plt.Line2D([0], [0], color='orange', linestyle=':', lw=1.5, label='50% Threshold')
    ], loc='lower right')
    
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "edge_stability_plot.png"), dpi=300)
    plt.close()
    
    # 5. Print Summary Table to stdout
    print("\n" + "="*60)
    print("FINAL SUMMARY: BOOTSTRAP STABILITY")
    print("="*60)
    print(f"Original Edges (Total: {len(original_edges)}):")
    print(f"  - Stable (>= 0.90): {len(stable_edges)}")
    print(f"  - Unstable (< 0.90): {len(unstable_edges)}")
    print(f"Non-Selected Pairs (Total: {21 - len(original_edges)}):")
    print(f"  - Near-Threshold (>= 0.50): {len(near_threshold)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
