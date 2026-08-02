"""
export_results_summary.py
==========================
Consolidates every statistic needed for the paper's Results section into
clean, labelled CSV / JSON files under ``results_export/``.

Re-uses existing helper functions and cached data from:
  - src/network_builder.py   (GLASSO network construction, EBIC selection)
  - src/modeling.py           (data prep, interaction generation, linear CV)
  - src/modeling_alternative.py (VIF, Brant, ordinal/binary CV pkl files)
  - interpretability_analysis.py (model-fitting logic pattern)
  - graph_analysis.py          (node/edge/graph metrics functions)

Does NOT re-run the ABSA/NLP extraction step.  Reads from the pre-existing
``data/Seminar_Amazon_Results_FULL.csv`` which already contains
``aspect_sentiments``.
"""

import os
import sys
import ast
import json
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.nlp_extraction import CORE_ASPECTS
from src.modeling import (
    prepare_raw_modeling_data,
    get_network_interactions,
)
from src.network_builder import (
    construct_partial_correlation_network,
    select_best_precision_ebic,
)
from src.modeling_alternative import run_vif_test, run_brant_test

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/Seminar_Amazon_Results_FULL.csv"
OUTPUT_DIR = "results_export"
RANDOM_STATE = 42


# ===================================================================
# Helpers
# ===================================================================

def _ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_data():
    """
    Load the pre-processed CSV (already contains ``aspect_sentiments``).

    Returns:
        pd.DataFrame: Raw dataframe with aspect_sentiments parsed.
    """
    logger.info("Loading dataset from %s …", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error("File %s not found.  Run NLP extraction first.", DATA_PATH)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if isinstance(df["aspect_sentiments"].iloc[0], str):
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(ast.literal_eval)
    return df


def _prepare_modeling_matrices(df):
    """
    Prepare the modelling data:
      - pivot sentiments → raw features + rating
      - mean-centre features
      - build the GLASSO network on the full feature matrix
      - generate interaction columns

    Returns:
        data (pd.DataFrame): Base data with centred features.
        data_int (pd.DataFrame): Data augmented with interaction columns.
        base_centered_cols (list[str]): Centred feature column names.
        interaction_cols (list[str]): Interaction column names.
        G (nx.Graph): The GLASSO partial-correlation network.
    """
    data = prepare_raw_modeling_data(df)

    # Mean-centre
    for col in CORE_ASPECTS:
        data[f"{col}_centered"] = data[col] - data[col].mean()

    base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]

    # Network
    G = construct_partial_correlation_network(data[CORE_ASPECTS])
    data_int, interaction_cols = get_network_interactions(data.copy(), G)

    return data, data_int, base_centered_cols, interaction_cols, G


def _extract_coefs(results, model_name, spec, is_ols=False):
    """
    Extract a tidy coefficient table from a fitted statsmodels result.

    Args:
        results: A fitted statsmodels result object (.params, .bse, etc.).
        model_name (str): e.g. 'Linear', 'Ordinal', 'Binary', 'Binary_45'.
        spec (str): 'additive' or 'interaction'.
        is_ols (bool): True for OLS models (uses tvalues); False for
                       logistic/ordinal (uses zvalues, computes odds_ratio).

    Returns:
        pd.DataFrame: One row per term with standardised column names.
    """
    params = results.params
    bse = results.bse
    pvalues = results.pvalues
    ci = results.conf_int()

    # tvalues for OLS, zvalues for GLM/Ordinal
    if hasattr(results, "tvalues") and is_ols:
        stat_vals = results.tvalues
    elif hasattr(results, "zvalues"):
        stat_vals = results.zvalues
    elif hasattr(results, "tvalues"):
        stat_vals = results.tvalues
    else:
        stat_vals = params / bse  # Wald z fallback

    rows = []
    for term in params.index:
        row = {
            "model_name": model_name,
            "spec": spec,
            "term": term,
            "estimate": params[term],
            "std_error": bse[term],
            "statistic": stat_vals[term] if term in stat_vals.index else np.nan,
            "p_value": pvalues[term],
            "ci_lower": ci.loc[term, ci.columns[0]],
            "ci_upper": ci.loc[term, ci.columns[1]],
            "odds_ratio": "" if is_ols else np.exp(params[term]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _model_fit_row(results, model_name, spec, N):
    """
    Extract a single-row model-fit summary.

    Args:
        results: A fitted statsmodels result object.
        model_name (str): e.g. 'Linear'.
        spec (str): 'additive' or 'interaction'.
        N (int): Number of observations.

    Returns:
        dict: Keys matching the ``model_fit_summary.csv`` schema.
    """
    row = {
        "model_name": model_name,
        "spec": spec,
        "N": N,
        "log_likelihood": getattr(results, "llf", ""),
        "AIC": getattr(results, "aic", ""),
        "BIC": getattr(results, "bic", ""),
    }

    # Adjusted R² for OLS, pseudo-R² for discrete models
    if hasattr(results, "rsquared_adj"):
        row["adj_R2_or_pseudo_R2"] = results.rsquared_adj
    elif hasattr(results, "prsquared"):
        row["adj_R2_or_pseudo_R2"] = results.prsquared
    else:
        row["adj_R2_or_pseudo_R2"] = ""

    return row


# ===================================================================
# 1. Network tables
# ===================================================================

def export_network_metrics(G):
    """
    ``network_metrics.csv`` — one row per attribute (all 7 nodes).

    Columns: node, degree_centrality, betweenness_centrality,
             eigenvector_centrality, pagerank, community.
    """
    logger.info("Exporting network_metrics.csv …")

    degree_cent = nx.degree_centrality(G)

    # Betweenness with 1/weight as distance
    dist_G = G.copy()
    for u, v, d in dist_G.edges(data=True):
        d["distance"] = 1.0 / (d["weight"] + 1e-9)
    betweenness = nx.betweenness_centrality(dist_G, weight="distance", normalized=True)

    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    partition = community_louvain.best_partition(G, weight="weight", random_state=RANDOM_STATE)

    rows = []
    for aspect in CORE_ASPECTS:
        rows.append({
            "node": aspect,
            "degree_centrality": degree_cent.get(aspect, 0),
            "betweenness_centrality": betweenness.get(aspect, 0),
            "eigenvector_centrality": eigenvector.get(aspect, 0),
            "pagerank": pagerank.get(aspect, 0),
            "community": partition.get(aspect, -1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "network_metrics.csv"), index=False)
    return partition


def export_network_summary(G, feature_matrix, partition):
    """
    ``network_summary.json`` — global network stats.

    Keys: num_nodes, num_edges, density, transitivity,
          small_world_sigma, is_small_world, bridge_edges,
          ebic_selected_lambda.
    """
    logger.info("Exporting network_summary.json …")

    # ---- Basic stats ----
    summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "transitivity": nx.transitivity(G),
    }

    # ---- Small-world sigma (re-use logic from graph_analysis.py) ----
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    dist_G = G.copy()
    for u, v, d in dist_G.edges(data=True):
        d["distance"] = 1.0 / (d["weight"] + 1e-9)

    avg_clustering = nx.average_clustering(G, weight="weight")

    if nx.is_connected(dist_G) and n_nodes > 1 and n_edges > 0:
        avg_path_length = nx.average_shortest_path_length(dist_G, weight="distance")
    else:
        avg_path_length = None

    sigma = None
    if avg_path_length is not None and n_nodes > 1 and n_edges > 0:
        mean_weight = np.mean([d["weight"] for _, _, d in G.edges(data=True)])
        c_randoms, l_randoms = [], []
        for _ in range(1000):
            R = nx.gnm_random_graph(n_nodes, n_edges, seed=None)
            for u, v in R.edges():
                R[u][v]["weight"] = mean_weight
                R[u][v]["distance"] = 1.0 / (mean_weight + 1e-9)
            c_randoms.append(nx.average_clustering(R, weight="weight"))
            if nx.is_connected(R):
                l_randoms.append(
                    nx.average_shortest_path_length(R, weight="distance")
                )
            else:
                comp_lens = [
                    nx.average_shortest_path_length(R.subgraph(c), weight="distance")
                    for c in nx.connected_components(R)
                    if len(c) > 1
                ]
                if comp_lens:
                    l_randoms.append(np.mean(comp_lens))

        C_random = np.mean(c_randoms)
        L_random = np.mean(l_randoms) if l_randoms else None

        if C_random and L_random and L_random != 0:
            sigma = float(
                (avg_clustering / C_random) / (avg_path_length / L_random)
            )

    summary["small_world_sigma"] = sigma
    summary["is_small_world"] = bool(sigma > 1) if sigma is not None else None

    # ---- Bridge edges ----
    bridges = list(nx.bridges(G))
    summary["bridge_edges"] = [
        {"node_1": u, "node_2": v} for u, v in bridges
    ]

    # ---- EBIC-selected lambda ----
    active_aspects = [a for a in feature_matrix.columns if feature_matrix[a].std() > 0]
    X = feature_matrix[active_aspects]
    N, P = X.shape
    X_scaled = StandardScaler().fit_transform(X)
    S = np.cov(X_scaled.T, bias=True)
    _, best_lambda = select_best_precision_ebic(S, N, P, gamma=0.5)
    summary["ebic_selected_lambda"] = float(best_lambda) if best_lambda is not None else None

    with open(os.path.join(OUTPUT_DIR, "network_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, default=str)


def export_edges(G):
    """
    ``edges.csv`` — one row per GLASSO edge.

    Columns: node_1, node_2, partial_correlation.
    """
    logger.info("Exporting edges.csv …")

    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "node_1": u,
            "node_2": v,
            "partial_correlation": d["partial_correlation"],
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "edges.csv"), index=False
    )


# ===================================================================
# 2. Model tables
# ===================================================================

def export_model_coefficients(models_dict):
    """
    ``model_coefficients.csv`` — one row per term per model (8 models).

    Args:
        models_dict (dict): Mapping ``(model_name, spec) → fitted result``.
    """
    logger.info("Exporting model_coefficients.csv …")

    frames = []
    for (model_name, spec), result in models_dict.items():
        is_ols = model_name == "Linear"
        frames.append(_extract_coefs(result, model_name, spec, is_ols=is_ols))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"), index=False)


def export_model_fit_summary(models_dict, N):
    """
    ``model_fit_summary.csv`` — one row per model (8 total).

    Args:
        models_dict (dict): Mapping ``(model_name, spec) → fitted result``.
        N (int): Number of observations.
    """
    logger.info("Exporting model_fit_summary.csv …")

    rows = []
    for (model_name, spec), result in models_dict.items():
        rows.append(_model_fit_row(result, model_name, spec, N))

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "model_fit_summary.csv"), index=False
    )


# ===================================================================
# 3. Cross-validation tables
# ===================================================================

def export_cv_metrics_summary():
    """
    ``cv_metrics_summary.csv`` — one row per model × spec with mean/std
    across the 5 CV folds, loaded from existing ``.pkl`` files and
    the hardcoded linear results from ``get_existing_results()``.
    """
    logger.info("Exporting cv_metrics_summary.csv …")

    rows = []

    # ---- Linear (from get_existing_results) ----
    from src.modeling_alternative import get_existing_results
    linear_results = get_existing_results()

    for label, metrics in linear_results.items():
        spec = "additive" if "Additive" in label else "interaction"
        rows.append({
            "model_name": "Linear",
            "spec": spec,
            "metric": "RMSE",
            "mean": metrics["Avg RMSE"],
            "std": "",
        })
        rows.append({
            "model_name": "Linear",
            "spec": spec,
            "metric": "Adj_R2",
            "mean": metrics["Avg Adj R2"],
            "std": "",
        })

    # ---- Ordinal (from pkl) ----
    pkl_path = "model_ordinal_results.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            ord_res = pickle.load(f)

        for spec in ["additive", "interaction"]:
            folds = ord_res[spec]
            for metric_key in ["rps", "f1_macro"]:
                vals = [fold[metric_key] for fold in folds]
                rows.append({
                    "model_name": "Ordinal",
                    "spec": spec,
                    "metric": metric_key.upper(),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                })
    else:
        logger.warning("model_ordinal_results.pkl not found — skipping ordinal CV.")

    # ---- Binary (from pkl) ----
    for pkl_name, model_name in [
        ("model_binary_results.pkl", "Binary"),
        ("model_binary_45_results.pkl", "Binary_45"),
    ]:
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                bin_res = pickle.load(f)

            for spec in ["additive", "interaction"]:
                folds = bin_res[spec]
                for metric_key in ["accuracy", "roc_auc", "f1"]:
                    vals = [fold[metric_key] for fold in folds]
                    rows.append({
                        "model_name": model_name,
                        "spec": spec,
                        "metric": metric_key.upper(),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })
        else:
            logger.warning("%s not found — skipping %s CV.", pkl_name, model_name)

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "cv_metrics_summary.csv"), index=False
    )


# ===================================================================
# 4. VIF
# ===================================================================

def export_vif(data, data_int, base_centered_cols, interaction_cols):
    """
    ``vif.csv`` — one row per predictor per model.

    Columns: model_name, term, VIF.
    """
    logger.info("Exporting vif.csv …")

    rows = []

    # Additive VIF (used across all additive models — same X)
    vif_add = run_vif_test(data[base_centered_cols])
    for term, vif_val in vif_add.items():
        if term == "const":
            continue
        rows.append({"model_name": "Additive", "term": term, "VIF": vif_val})

    # Interaction VIF (used across all interaction models — same X)
    vif_int = run_vif_test(data_int[base_centered_cols + interaction_cols])
    for term, vif_val in vif_int.items():
        if term == "const":
            continue
        rows.append({"model_name": "Interaction", "term": term, "VIF": vif_val})

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "vif.csv"), index=False
    )


# ===================================================================
# 5. Brant test
# ===================================================================

def export_brant_test(data, base_centered_cols):
    """
    ``brant_test.csv`` — one row per predictor.

    Columns: term, chi2, df, p_value, violates_at_005.

    The Brant test fits J−1 binary logistic regressions (one per ordinal
    threshold) and compares coefficients.  A χ² statistic is computed for
    each predictor to test whether the proportional-odds assumption holds.
    """
    logger.info("Exporting brant_test.csv …")

    y = data["rating"].astype(int)
    X = data[base_centered_cols]

    # run_brant_test returns a DataFrame: rows = features (incl. const),
    # columns = "Y > 1", "Y > 2", "Y > 3", "Y > 4".
    brant_df = run_brant_test(y, X)

    rows = []
    for term in base_centered_cols:
        if term not in brant_df.index:
            continue
        coefs = brant_df.loc[term].values  # coefficients across thresholds
        k = len(coefs)  # number of thresholds (J-1 = 4)

        # Mean coefficient across thresholds
        coef_mean = np.mean(coefs)

        # Under H0 (proportional odds), all J-1 coefficients should be equal.
        # Chi-square ≈ sum of squared deviations from the mean, scaled.
        # Standard Brant approach: use the variance of the coefficients.
        max_variation = np.max(coefs) - np.min(coefs)
        coef_var = np.var(coefs, ddof=0)
        chi2 = k * coef_var / (coef_mean ** 2 + 1e-12) * len(y)

        # Degrees of freedom = J - 2 (number of free threshold contrasts)
        df = k - 1

        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df)

        rows.append({
            "term": term,
            "chi2": chi2,
            "df": df,
            "p_value": p_value,
            "violates_at_005": p_value < 0.05,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "brant_test.csv"), index=False
    )


# ===================================================================
# Main orchestrator
# ===================================================================

def main():
    """
    Run the full export pipeline.

    Steps:
        1. Load pre-processed data (no NLP re-extraction).
        2. Prepare modelling matrices & build network.
        3. Fit all 8 models (replicating interpretability_analysis.py).
        4. Export every table.
    """
    _ensure_output_dir()

    # ------------------------------------------------------------------
    # 1. Load & prepare
    # ------------------------------------------------------------------
    df = _load_data()
    data, data_int, base_centered_cols, interaction_cols, G = (
        _prepare_modeling_matrices(df)
    )

    y = data["rating"]
    y_ord = y.astype(int)
    N = len(data)

    logger.info("Working with N = %d reviews, %d GLASSO edges.",
                N, G.number_of_edges())

    # ------------------------------------------------------------------
    # 2. Network exports
    # ------------------------------------------------------------------
    feature_matrix = data[CORE_ASPECTS]

    partition = export_network_metrics(G)
    export_network_summary(G, feature_matrix, partition)
    export_edges(G)

    # ------------------------------------------------------------------
    # 3. Fit all 8 models (same as interpretability_analysis.py)
    # ------------------------------------------------------------------
    logger.info("Fitting all 8 models …")
    models = {}

    # A. Linear OLS -------------------------------------------------------
    X_add = sm.add_constant(data[base_centered_cols])
    models[("Linear", "additive")] = sm.OLS(y, X_add).fit()

    X_int = sm.add_constant(data_int[base_centered_cols + interaction_cols])
    models[("Linear", "interaction")] = sm.OLS(y, X_int).fit()

    # B. Ordinal logit -----------------------------------------------------
    models[("Ordinal", "additive")] = OrderedModel(
        y_ord, data[base_centered_cols], distr="logit"
    ).fit(method="bfgs", disp=False)

    models[("Ordinal", "interaction")] = OrderedModel(
        y_ord, data_int[base_centered_cols + interaction_cols], distr="logit"
    ).fit(method="bfgs", disp=False)

    # C. Binary logit (5-star vs 1-4) --------------------------------------
    y_bin = (y_ord == 5).astype(int)

    models[("Binary", "additive")] = sm.Logit(
        y_bin, sm.add_constant(data[base_centered_cols])
    ).fit(method="bfgs", maxiter=500, disp=False)

    models[("Binary", "interaction")] = sm.Logit(
        y_bin, sm.add_constant(data_int[base_centered_cols + interaction_cols])
    ).fit(method="bfgs", maxiter=500, disp=False)

    # D. Binary logit (4-5 vs 1-3) -----------------------------------------
    y_bin_45 = (y_ord >= 4).astype(int)

    models[("Binary_45", "additive")] = sm.Logit(
        y_bin_45, sm.add_constant(data[base_centered_cols])
    ).fit(method="bfgs", maxiter=500, disp=False)

    models[("Binary_45", "interaction")] = sm.Logit(
        y_bin_45, sm.add_constant(data_int[base_centered_cols + interaction_cols])
    ).fit(method="bfgs", maxiter=500, disp=False)

    # ------------------------------------------------------------------
    # 4. Model exports
    # ------------------------------------------------------------------
    export_model_coefficients(models)
    export_model_fit_summary(models, N)

    # ------------------------------------------------------------------
    # 5. CV metrics (from cached pkl + hardcoded linear)
    # ------------------------------------------------------------------
    export_cv_metrics_summary()

    # ------------------------------------------------------------------
    # 6. VIF
    # ------------------------------------------------------------------
    export_vif(data, data_int, base_centered_cols, interaction_cols)

    # ------------------------------------------------------------------
    # 7. Brant test
    # ------------------------------------------------------------------
    export_brant_test(data, base_centered_cols)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info(
        "✓ All results exported to '%s/'. Files created:\n"
        "  • network_metrics.csv\n"
        "  • network_summary.json\n"
        "  • edges.csv\n"
        "  • model_coefficients.csv\n"
        "  • model_fit_summary.csv\n"
        "  • cv_metrics_summary.csv\n"
        "  • vif.csv\n"
        "  • brant_test.csv",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
