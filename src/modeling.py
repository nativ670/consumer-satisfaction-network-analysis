import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import load_and_clean_data
from src.nlp_extraction import extract_aspects_and_sentiments, CORE_ASPECTS
from src.network_builder import build_and_analyze_network, pivot_aspect_sentiments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_modeling_data(df):
    """
    Pivots sentiments, centers them, and prepares the target variable.
    
    Args:
        df (pd.DataFrame): DataFrame with 'aspect_sentiments' and 'rating'.
        
    Returns:
        pd.DataFrame: A DataFrame with centered base features and 'rating'.
    """
    logger.info("Pivoting and centering sentiment features...")
    
    # 1. Pivot sentiments
    feature_matrix = pivot_aspect_sentiments(df)
    
    # 2. Add rating
    feature_matrix['rating'] = df['rating'].values
    
    # 3. Drop rows with missing rating (should be none due to cleaning, but safe)
    feature_matrix = feature_matrix.dropna(subset=['rating'])
    
    # 4. Methodological Filtering: Drop rows with 0.0 across all CORE_ASPECTS (Sparsity Filter)
    initial_count = len(feature_matrix)
    mask = (feature_matrix[CORE_ASPECTS] != 0.0).any(axis=1)
    feature_matrix = feature_matrix[mask].copy()
    
    dropped_count = initial_count - len(feature_matrix)
    logger.info(f"Sparsity Filter: Dropped {dropped_count} empty reviews. Remaining valid reviews: {len(feature_matrix)}")
    
    # 5. Mean Center the base sentiment columns
    # We keep the original names for now, will sanitize for statsmodels later
    base_features = CORE_ASPECTS
    for col in base_features:
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

def run_5fold_cv(X, y):
    """Runs 5-Fold CV and returns average RMSE and R-squared."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse_scores = []
    r2_scores = []
    adj_r2_scores = []
    
    n = len(y)
    p = X.shape[1]
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        adj_r2 = calculate_adjusted_r2(r2, len(y_test), p)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        
    return np.mean(rmse_scores), np.mean(adj_r2_scores)

def compare_models(df, G):
    """
    Executes the comparison between Baseline and Network models.
    """
    data = prepare_modeling_data(df)
    data, interaction_features = get_network_interactions(data, G)
    
    base_features = [f"{col}_centered" for col in CORE_ASPECTS]
    
    # Sanitize names for statsmodels/formula compatibility
    name_map = {f"{col}_centered": col.replace('/', '_').replace(' ', '_') + "_c" for col in CORE_ASPECTS}
    data = data.rename(columns=name_map)
    base_features_sanitized = list(name_map.values())
    
    y = data['rating']
    
    # Model 1: Baseline
    X1 = data[base_features_sanitized]
    rmse1, adj_r2_1 = run_5fold_cv(X1, y)
    
    # Model 2: Network-Informed
    X2 = data[base_features_sanitized + interaction_features]
    rmse2, adj_r2_2 = run_5fold_cv(X2, y)
    
    # Full fit for AIC/BIC
    # Add constant for statsmodels
    X1_sm = sm.add_constant(X1)
    sm_model1 = sm.OLS(y, X1_sm).fit()
    
    X2_sm = sm.add_constant(X2)
    sm_model2 = sm.OLS(y, X2_sm).fit()
    
    results = {
        'Baseline': {'RMSE': rmse1, 'AdjR2': adj_r2_1, 'AIC': sm_model1.aic, 'BIC': sm_model1.bic},
        'Network': {'RMSE': rmse2, 'AdjR2': adj_r2_2, 'AIC': sm_model2.aic, 'BIC': sm_model2.bic}
    }
    
    # Print Comparison Table
    print("\n" + "="*70)
    print(f"{'Model':<15} | {'Avg RMSE':<12} | {'Avg Adj R2':<12} | {'AIC':<12} | {'BIC':<12}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<15} | {m['RMSE']:<12.4f} | {m['AdjR2']:<12.4f} | {m['AIC']:<12.1f} | {m['BIC']:<12.1f}")
    print("="*70)
    
    print("\nBASELINE MODEL SUMMARY (Full Dataset):")
    print(sm_model1.summary())

    print("\nNETWORK MODEL SUMMARY (Full Dataset):")
    print(sm_model2.summary())
    
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
