import os
import sys
import ast
import json
import logging
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nested_cv_bic.json")

def load_data():
    logger.info("Loading dataset from %s ...", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error("File not found.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    if isinstance(df["aspect_sentiments"].iloc[0], str):
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(ast.literal_eval)
    return df

def run_nested_cv_bic(df):
    raw_data = prepare_raw_modeling_data(df)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    baseline_bics = []
    network_bics = []
    
    logger.info("Starting Nested CV for Fold-Averaged BIC...")
    
    fold = 1
    for train_index, test_index in kf.split(raw_data):
        logger.info(f"Processing Fold {fold}/5...")
        
        train_df = raw_data.iloc[train_index].copy()
        test_df = raw_data.iloc[test_index].copy()
        
        # Centering on train folds only
        train_means = train_df[CORE_ASPECTS].mean()
        
        base_centered_cols = []
        for col in CORE_ASPECTS:
            c_col = f"{col}_centered"
            train_df[c_col] = train_df[col] - train_means[col]
            base_centered_cols.append(c_col)
            
        y_train = train_df['rating']
        
        # 1. Baseline Model Evaluation on Train Fold
        X_train_base = train_df[base_centered_cols]
        X_train_base_sm = sm.add_constant(X_train_base)
        m_base = sm.OLS(y_train, X_train_base_sm).fit()
        
        baseline_bics.append(m_base.bic)
        
        # 2. Network Model (Structure Discovery on TRAIN ONLY)
        G_fold = construct_partial_correlation_network(train_df[CORE_ASPECTS])
        
        train_df_int, interaction_cols = get_network_interactions(train_df, G_fold)
        
        X_train_net = train_df_int[base_centered_cols + interaction_cols]
        X_train_net_sm = sm.add_constant(X_train_net)
        m_net = sm.OLS(y_train, X_train_net_sm).fit()
        
        network_bics.append(m_net.bic)
        
        logger.info(f"Fold {fold} | Baseline BIC: {m_base.bic:.2f} | Network BIC: {m_net.bic:.2f}")
        fold += 1
        
    avg_base_bic = np.mean(baseline_bics)
    avg_net_bic = np.mean(network_bics)
    
    results = {
        "baseline_fold_bics": baseline_bics,
        "network_fold_bics": network_bics,
        "avg_baseline_bic": avg_base_bic,
        "avg_network_bic": avg_net_bic,
        "bic_improvement": avg_base_bic - avg_net_bic
    }
    
    logger.info("Nested CV Complete.")
    logger.info(f"Avg Baseline BIC: {avg_base_bic:.2f}")
    logger.info(f"Avg Network BIC: {avg_net_bic:.2f}")
    
    return results

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    results = run_nested_cv_bic(df)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Results saved to {OUTPUT_FILE}")
