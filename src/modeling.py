import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import load_and_clean_data
from src.nlp_extraction import extract_aspects_and_sentiments, CORE_ASPECTS
from src.network_builder import (
    build_and_analyze_network, 
    pivot_aspect_sentiments, 
    construct_partial_correlation_network
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_raw_modeling_data(df):
    """
    Pivots sentiments and prepares the target variable without centering.
    
    Args:
        df (pd.DataFrame): DataFrame with 'aspect_sentiments' and 'rating'.
        
    Returns:
        pd.DataFrame: A DataFrame with raw base features and 'rating'.
    """
    logger.info("Pivoting raw sentiment features...")
    
    # 1. Pivot sentiments
    feature_matrix = pivot_aspect_sentiments(df)
    
    # 2. Add rating
    feature_matrix['rating'] = df['rating'].values
    
    # 3. Drop rows with missing rating
    feature_matrix = feature_matrix.dropna(subset=['rating'])
    
    # 4. Methodological Filtering: Drop rows with 0.0 across all CORE_ASPECTS (Sparsity Filter)
    initial_count = len(feature_matrix)
    mask = (feature_matrix[CORE_ASPECTS] != 0.0).any(axis=1)
    feature_matrix = feature_matrix[mask].copy()
    
    dropped_count = initial_count - len(feature_matrix)
    logger.info(f"Sparsity Filter: Dropped {dropped_count} empty reviews. Remaining valid reviews: {len(feature_matrix)}")
    
    return feature_matrix

def prepare_modeling_data(df):
    """
    Legacy wrapper that pivots and centers sentiments.
    """
    feature_matrix = prepare_raw_modeling_data(df)
    
    # Mean Center the base sentiment columns
    for col in CORE_ASPECTS:
        feature_matrix[f"{col}_centered"] = feature_matrix[col] - feature_matrix[col].mean()
        
    return feature_matrix

def get_network_interactions(feature_matrix, G):
    """
    Dynamically creates interaction features for every edge in the Graphical Lasso network.
    
    Args:
        feature_matrix (pd.DataFrame): DataFrame with centered features.
        G (nx.Graph): The partial correlation network.
        
    Returns:
        pd.DataFrame: Updated DataFrame with interaction terms.
        list: Names of the interaction columns created.
    """
    logger.info(f"Generating interaction features from {G.number_of_edges()} network edges...")
    
    interaction_cols = []
    
    for u, v in G.edges():
        # Use centered versions for interaction to avoid structural multicollinearity
        col_name = f"int_{u}_{v}".replace('/', '_').replace(' ', '_')
        feature_matrix[col_name] = feature_matrix[f"{u}_centered"] * feature_matrix[f"{v}_centered"]
        interaction_cols.append(col_name)
        
    return feature_matrix, interaction_cols

def calculate_adjusted_r2(r2, n, p):
    """Calculates Adjusted R-squared."""
    if n - p - 1 == 0:
        return 0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def run_bulletproof_cv(df):
    """
    Performs leakage-free cross-validation by:
    1. Scaling/Centering on train folds only.
    2. Discovering Network topology (GLASSO) on train folds only.
    3. Evaluating both OLS models on unseen test folds.
    
    Args:
        df (pd.DataFrame): DataFrame with 'aspect_sentiments' and 'rating'.
        
    Returns:
        tuple: (baseline_metrics, network_metrics)
    """
    raw_data = prepare_raw_modeling_data(df)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    baseline_results = []
    network_results = []
    
    logger.info("Starting Bulletproof Cross-Validation (Leakage-Free)...")
    
    fold = 1
    for train_index, test_index in kf.split(raw_data):
        logger.info(f"Processing Fold {fold}/5...")
        
        train_df = raw_data.iloc[train_index].copy()
        test_df = raw_data.iloc[test_index].copy()
        
        # 1. Centering (Preventing scalar leakage)
        # Calculate means on TRAIN ONLY
        train_means = train_df[CORE_ASPECTS].mean()
        
        for col in CORE_ASPECTS:
            train_df[f"{col}_centered"] = train_df[col] - train_means[col]
            test_df[f"{col}_centered"] = test_df[col] - train_means[col]
            
        y_train = train_df['rating']
        y_test = test_df['rating']
        
        base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
        
        # 2. Baseline Model Evaluation
        X_train_base = train_df[base_centered_cols]
        X_test_base = test_df[base_centered_cols]
        
        reg_base = LinearRegression().fit(X_train_base, y_train)
        preds_base = reg_base.predict(X_test_base)
        
        rmse_base = np.sqrt(mean_squared_error(y_test, preds_base))
        r2_base = r2_score(y_test, preds_base)
        adj_r2_base = calculate_adjusted_r2(r2_base, len(y_test), len(base_centered_cols))
        baseline_results.append((rmse_base, adj_r2_base))
        
        # 3. Network Model (Structure Discovery on TRAIN ONLY)
        G_fold = construct_partial_correlation_network(train_df[CORE_ASPECTS])
        
        # Generate interactions for both train and test based on DISCOVERED structure
        train_df_int, interaction_cols = get_network_interactions(train_df, G_fold)
        test_df_int, _ = get_network_interactions(test_df, G_fold)
        
        # Train Network Model
        X_train_net = train_df_int[base_centered_cols + interaction_cols]
        X_test_net = test_df_int[base_centered_cols + interaction_cols]
        
        reg_net = LinearRegression().fit(X_train_net, y_train)
        preds_net = reg_net.predict(X_test_net)
        
        rmse_net = np.sqrt(mean_squared_error(y_test, preds_net))
        r2_net = r2_score(y_test, preds_net)
        adj_r2_net = calculate_adjusted_r2(r2_net, len(y_test), len(base_centered_cols) + len(interaction_cols))
        network_results.append((rmse_net, adj_r2_net))
        
        fold += 1
        
    avg_rmse_base = np.mean([r[0] for r in baseline_results])
    avg_adj_r2_base = np.mean([r[1] for r in baseline_results])
    
    avg_rmse_net = np.mean([r[0] for r in network_results])
    avg_adj_r2_net = np.mean([r[1] for r in network_results])
    
    return (avg_rmse_base, avg_adj_r2_base), (avg_rmse_net, avg_adj_r2_net)

def compare_models(df, G):
    """
    Executes the comparison between Baseline and Network models using leakage-free CV.
    """
    # 1. True Predictive Performance (CV)
    (rmse1, adj_r2_1), (rmse2, adj_r2_2) = run_bulletproof_cv(df)
    
    # 2. Final Parameter Estimates (Full Dataset fit for interpretability/BIC)
    data = prepare_modeling_data(df)
    data, interaction_features = get_network_interactions(data, G)
    
    base_features = [f"{col}_centered" for col in CORE_ASPECTS]
    name_map = {f"{col}_centered": col.replace('/', '_').replace(' ', '_') + "_c" for col in CORE_ASPECTS}
    data = data.rename(columns=name_map)
    base_features_sanitized = list(name_map.values())
    
    y = data['rating']
    
    # Model 1: Baseline
    X1 = data[base_features_sanitized]
    X1_sm = sm.add_constant(X1)
    sm_model1 = sm.OLS(y, X1_sm).fit()
    
    # Model 2: Network-Informed
    X2 = data[base_features_sanitized + interaction_features]
    X2_sm = sm.add_constant(X2)
    sm_model2 = sm.OLS(y, X2_sm).fit()
    
    results = {
        'Baseline': {'RMSE': rmse1, 'AdjR2': adj_r2_1, 'AIC': sm_model1.aic, 'BIC': sm_model1.bic},
        'Network': {'RMSE': rmse2, 'AdjR2': adj_r2_2, 'AIC': sm_model2.aic, 'BIC': sm_model2.bic}
    }
    
    # Print Comparison Table
    print("\n" + "="*80)
    print("BULLETPROOF COMPARISON: CROSS-VALIDATED PERFORMANCE (No Leakage)")
    print("="*80)
    print(f"{'Model':<15} | {'Avg RMSE':<12} | {'Avg Adj R2':<12} | {'Full AIC':<12} | {'Full BIC':<12}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<15} | {m['RMSE']:<12.4f} | {m['AdjR2']:<12.4f} | {m['AIC']:<12.1f} | {m['BIC']:<12.1f}")
    print("="*80)
    
    print("\nInterpretation: RMSE and Adj R2 are calculated using leakage-free cross-validation.")
    print("AIC and BIC are calculated on the full-dataset fit for model selection purposes.")
    
    return {"baseline": sm_model1, "network": sm_model2}

if __name__ == "__main__":
    try:
        # Load sample
        logger.info("Loading 2500 reviews for modeling verification...")
        df = load_and_clean_data().head(2500)
        
        # NLP Extraction
        df = extract_aspects_and_sentiments(df)
        
        # Build Network
        network_results = build_and_analyze_network(df)
        G = network_results['graph']
        
        # Run Comparison
        compare_models(df, G)
        
    except Exception as e:
        logger.error(f"Modeling failed: {e}")
        import traceback
        traceback.print_exc()
