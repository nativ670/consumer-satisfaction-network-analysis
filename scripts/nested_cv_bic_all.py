import os
import sys
import ast
import json
import logging
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.nlp_extraction import CORE_ASPECTS
from src.modeling import prepare_raw_modeling_data, get_network_interactions
from src.network_builder import construct_partial_correlation_network

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/Seminar_Amazon_Results_FULL.csv"
OUTPUT_DIR = "results_export"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nested_cv_bic_all_models.json")

def load_data():
    logger.info("Loading dataset from %s ...", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error("File not found.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    if isinstance(df["aspect_sentiments"].iloc[0], str):
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(ast.literal_eval)
    return df

def fit_models(X, y_lin, y_ord, y_bin, y_bin45):
    # Add constant for OLS and Logit (OrderedModel adds its own thresholds instead of a constant)
    X_sm = sm.add_constant(X)
    
    # 1. Linear
    m_lin = sm.OLS(y_lin, X_sm).fit()
    bic_lin = m_lin.bic
    
    # 2. Ordinal
    try:
        m_ord = OrderedModel(y_ord, X, distr='logit').fit(method='bfgs', disp=False)
        bic_ord = m_ord.bic
    except Exception as e:
        logger.error(f"Ordinal model failed to fit: {e}")
        bic_ord = np.nan
        
    # 3. Binary (5 vs 1-4)
    try:
        m_bin = sm.Logit(y_bin, X_sm).fit(disp=False)
        bic_bin = m_bin.bic
    except Exception as e:
        logger.error(f"Binary model failed to fit: {e}")
        bic_bin = np.nan
        
    # 4. Binary 4-5 vs 1-3
    try:
        m_bin45 = sm.Logit(y_bin45, X_sm).fit(disp=False)
        bic_bin45 = m_bin45.bic
    except Exception as e:
        logger.error(f"Binary45 model failed to fit: {e}")
        bic_bin45 = np.nan
    
    return bic_lin, bic_ord, bic_bin, bic_bin45

def run_nested_cv_bic_all(df):
    raw_data = prepare_raw_modeling_data(df)
    
    raw_data['rating_lin'] = raw_data['rating']
    raw_data['rating_ord'] = raw_data['rating'].astype(int)
    raw_data['rating_bin'] = (raw_data['rating'] == 5).astype(int)
    raw_data['rating_bin45'] = (raw_data['rating'] >= 4).astype(int)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "Linear": {"baseline": [], "network": []},
        "Ordinal": {"baseline": [], "network": []},
        "Binary": {"baseline": [], "network": []},
        "Binary45": {"baseline": [], "network": []}
    }
    
    logger.info("Starting Nested CV for Fold-Averaged BIC (All Models)...")
    
    fold = 1
    for train_index, test_index in kf.split(raw_data):
        logger.info(f"Processing Fold {fold}/5...")
        
        train_df = raw_data.iloc[train_index].copy()
        
        # Centering on train folds only
        train_means = train_df[CORE_ASPECTS].mean()
        
        base_centered_cols = []
        for col in CORE_ASPECTS:
            c_col = f"{col}_centered"
            train_df[c_col] = train_df[col] - train_means[col]
            base_centered_cols.append(c_col)
            
        y_lin = train_df['rating_lin']
        y_ord = train_df['rating_ord']
        y_bin = train_df['rating_bin']
        y_bin45 = train_df['rating_bin45']
        
        # 1. Baseline Models Evaluation on Train Fold
        X_train_base = train_df[base_centered_cols]
        
        bic_lin_base, bic_ord_base, bic_bin_base, bic_bin45_base = fit_models(
            X_train_base, y_lin, y_ord, y_bin, y_bin45
        )
        
        results["Linear"]["baseline"].append(bic_lin_base)
        results["Ordinal"]["baseline"].append(bic_ord_base)
        results["Binary"]["baseline"].append(bic_bin_base)
        results["Binary45"]["baseline"].append(bic_bin45_base)
        
        # 2. Network Models (Structure Discovery on TRAIN ONLY)
        G_fold = construct_partial_correlation_network(train_df[CORE_ASPECTS])
        
        train_df_int, interaction_cols = get_network_interactions(train_df, G_fold)
        
        X_train_net = train_df_int[base_centered_cols + interaction_cols]
        
        bic_lin_net, bic_ord_net, bic_bin_net, bic_bin45_net = fit_models(
            X_train_net, y_lin, y_ord, y_bin, y_bin45
        )
        
        results["Linear"]["network"].append(bic_lin_net)
        results["Ordinal"]["network"].append(bic_ord_net)
        results["Binary"]["network"].append(bic_bin_net)
        results["Binary45"]["network"].append(bic_bin45_net)
        
        logger.info(f"Fold {fold} BICs | Lin Base: {bic_lin_base:.1f}, Net: {bic_lin_net:.1f} | Ord Base: {bic_ord_base:.1f}, Net: {bic_ord_net:.1f}")
        fold += 1
        
    final_summary = {}
    for model_name, bics in results.items():
        avg_base = np.nanmean(bics["baseline"])
        avg_net = np.nanmean(bics["network"])
        final_summary[model_name] = {
            "avg_baseline_bic": avg_base,
            "avg_network_bic": avg_net,
            "bic_improvement": avg_base - avg_net,
            "fold_bics": bics
        }
        logger.info(f"{model_name} -> Avg Baseline BIC: {avg_base:.2f}, Avg Network BIC: {avg_net:.2f}, Impr: {avg_base - avg_net:.2f}")
        
    return final_summary

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    results = run_nested_cv_bic_all(df)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Results saved to {OUTPUT_FILE}")
