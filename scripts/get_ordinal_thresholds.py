import os
import sys
import pandas as pd
import numpy as np
import ast
import logging
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

from src.modeling import prepare_raw_modeling_data, get_network_interactions, CORE_ASPECTS
from src.network_builder import construct_partial_correlation_network

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = 'data/Seminar_Amazon_Results_FULL.csv'

def get_thresholds():
    logger.info(f"Loading dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"File {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    if isinstance(df['aspect_sentiments'].iloc[0], str):
        df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    
    # Prepare data
    data = prepare_raw_modeling_data(df)
    y = data['rating'].astype(int)
    
    # Mean-center aspect scores
    for col in CORE_ASPECTS:
        data[f"{col}_centered"] = data[col] - data[col].mean()
    
    base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
    
    print("\n" + "="*50)
    print("ORDINAL MODEL THRESHOLD EXTRACTION")
    print("="*50)

    # 1. Additive Model
    logger.info("Fitting Additive Ordinal Model...")
    model_add = OrderedModel(y, data[base_centered_cols], distr='logit')
    res_add = model_add.fit(method='bfgs', disp=False)
    
    print("\n[1] Additive Model Analysis:")
    print("-" * 30)
    print("ALL parameters:")
    for name, value in res_add.params.items():
        print(f"  {name:<45}: {value:8.4f}")
            
    # Probabilities at Mean
    mean_vec_add = pd.DataFrame([[0] * len(base_centered_cols)], columns=base_centered_cols)
    probs_add = res_add.predict(mean_vec_add).values[0]
    print("\nPredicted Probabilities at Mean (Centered=0):")
    for i, p in enumerate(probs_add):
        print(f"  P(Rating={i+1}): {p:8.4f}")

    # 2. Interaction Model
    logger.info("Fitting Interaction Ordinal Model...")
    G = construct_partial_correlation_network(data[CORE_ASPECTS])
    data_int, interaction_cols = get_network_interactions(data, G)
    X_int = data_int[base_centered_cols + interaction_cols]
    
    model_int = OrderedModel(y, X_int, distr='logit')
    res_int = model_int.fit(method='bfgs', disp=False)
    
    print("\n[2] Interaction Model Analysis:")
    print("-" * 30)
    print("Thresholds:")
    for name, value in res_int.params.items():
        if '/' in name:
            print(f"  {name:<10}: {value:8.4f}")
            
    # Probabilities at Mean
    mean_vec_int = pd.DataFrame([[0] * len(X_int.columns)], columns=X_int.columns)
    probs_int = res_int.predict(mean_vec_int).values[0]
    print("\nPredicted Probabilities at Mean (Centered=0):")
    for i, p in enumerate(probs_int):
        print(f"  P(Rating={i+1}): {p:8.4f}")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    get_thresholds()
