import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    roc_auc_score, f1_score
)
import pickle
import os

from src.modeling import prepare_raw_modeling_data, get_network_interactions, CORE_ASPECTS
from src.network_builder import construct_partial_correlation_network

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_vif_test(X):
    """
    Calculates Variance Inflation Factor (VIF) for multicollinearity check.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        
    Returns:
        pd.Series: VIF values for each feature.
    """
    logger.info("Running Multicollinearity Test (VIF)...")
    X_with_const = sm.add_constant(X)
    vifs = []
    for i in range(X_with_const.shape[1]):
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
        except:
            vif = np.nan
        vifs.append(vif)
    return pd.Series(vifs, index=X_with_const.columns)

def run_brant_test(y, X):
    """
    Manually implements a Brant test logic for the proportional odds assumption.
    Fits J-1 binary logistic regressions and compares coefficients.
    
    Args:
        y (pd.Series): Target variable (ordinal).
        X (pd.DataFrame): Feature matrix.
        
    Returns:
        pd.DataFrame: Comparison of coefficients across thresholds.
    """
    logger.info("Running Brant Test for Proportional Odds Assumption...")
    unique_vals = sorted(y.unique())
    thresholds = unique_vals[:-1]
    
    binary_coefs = {}
    for t in thresholds:
        # Binary target: 1 if rating > t, else 0
        y_bin = (y > t).astype(int)
        try:
            model = sm.Logit(y_bin, sm.add_constant(X)).fit(disp=False)
            binary_coefs[f"Y > {t}"] = model.params
        except Exception as e:
            logger.warning(f"Logit failed for threshold {t}: {e}")
            binary_coefs[f"Y > {t}"] = pd.Series(np.nan, index=sm.add_constant(X).columns)
            
    return pd.DataFrame(binary_coefs)

def calculate_rps(probs, true_labels, num_categories=5):
    """
    Calculates the Ranked Probability Score (RPS) for ordinal predictions.
    
    Args:
        probs (np.array): Predicted probabilities for each category (N, C).
        true_labels (np.array): True labels (1-based, e.g., 1 to 5).
        num_categories (int): Number of ordinal categories.
        
    Returns:
        float: Mean RPS across samples.
    """
    # Convert true labels to one-hot encoding
    y_true = np.zeros((len(true_labels), num_categories))
    for i, label in enumerate(true_labels):
        y_true[i, int(label) - 1] = 1
    
    # Cumulative distributions
    cdf_pred = np.cumsum(probs, axis=1)
    cdf_true = np.cumsum(y_true, axis=1)
    
    # RPS formula: (1/(C-1)) * sum_{c=1}^{C-1} (cdf_pred - cdf_true)^2
    rps = np.mean(np.sum((cdf_pred[:, :-1] - cdf_true[:, :-1])**2, axis=1) / (num_categories - 1))
    return rps

def run_ordinal_cv(df):
    """
    Performs 5-fold CV for Ordinal Logistic Regression (Additive & Interaction).
    """
    raw_data = prepare_raw_modeling_data(df)
    
    # Ensure rating is 1-5 integers
    raw_data['rating'] = raw_data['rating'].astype(int)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'additive': [],
        'interaction': []
    }
    
    logger.info("Starting Ordinal Logistic Regression Cross-Validation...")
    
    fold = 1
    for train_index, test_index in kf.split(raw_data):
        logger.info(f"Processing Fold {fold}/5...")
        
        train_df = raw_data.iloc[train_index].copy()
        test_df = raw_data.iloc[test_index].copy()
        
        # Centering (Leakage-Free)
        train_means = train_df[CORE_ASPECTS].mean()
        for col in CORE_ASPECTS:
            train_df[f"{col}_centered"] = train_df[col] - train_means[col]
            test_df[f"{col}_centered"] = test_df[col] - train_means[col]
            
        y_train = train_df['rating']
        y_test = test_df['rating']
        
        base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
        
        # 1. Additive Model
        X_train_base = train_df[base_centered_cols]
        X_test_base = test_df[base_centered_cols]
        
        # Fit OrderedModel (Proportional Odds)
        # using 'logit' link for proportional odds
        model_add = OrderedModel(y_train, X_train_base, distr='logit')
        res_add = model_add.fit(method='bfgs', disp=False)
        
        # Predict probabilities
        probs_add = res_add.predict(X_test_base)
        # RPS
        rps_add = calculate_rps(probs_add.values, y_test.values)
        
        results['additive'].append({
            'log_likelihood': res_add.llf,
            'aic': res_add.aic,
            'rps': rps_add
        })
        
        # 2. Interaction Model
        G_fold = construct_partial_correlation_network(train_df[CORE_ASPECTS])
        train_df_int, interaction_cols = get_network_interactions(train_df, G_fold)
        test_df_int, _ = get_network_interactions(test_df, G_fold)
        
        X_train_net = train_df_int[base_centered_cols + interaction_cols]
        X_test_net = test_df_int[base_centered_cols + interaction_cols]
        
        model_int = OrderedModel(y_train, X_train_net, distr='logit')
        res_int = model_int.fit(method='bfgs', disp=False)
        
        probs_int = res_int.predict(X_test_net)
        rps_int = calculate_rps(probs_int.values, y_test.values)
        
        results['interaction'].append({
            'log_likelihood': res_int.llf,
            'aic': res_int.aic,
            'rps': rps_int
        })
        
        fold += 1
        
    return results

def run_binary_cv(df):
    """
    Performs 5-fold CV for Binary Logistic Regression (Additive & Interaction).
    Target: 5-stars = 1, 1-4 stars = 0.
    """
    raw_data = prepare_raw_modeling_data(df)
    
    # Binarize target
    raw_data['binary_rating'] = (raw_data['rating'] == 5).astype(int)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'additive': [],
        'interaction': []
    }
    
    logger.info("Starting Binary Logistic Regression Cross-Validation...")
    
    fold = 1
    for train_index, test_index in kf.split(raw_data):
        logger.info(f"Processing Fold {fold}/5...")
        
        train_df = raw_data.iloc[train_index].copy()
        test_df = raw_data.iloc[test_index].copy()
        
        # Centering (Leakage-Free)
        train_means = train_df[CORE_ASPECTS].mean()
        for col in CORE_ASPECTS:
            train_df[f"{col}_centered"] = train_df[col] - train_means[col]
            test_df[f"{col}_centered"] = test_df[col] - train_means[col]
            
        y_train = train_df['binary_rating']
        y_test = test_df['binary_rating']
        
        base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
        
        # 1. Additive Model
        X_train_base = train_df[base_centered_cols]
        X_test_base = test_df[base_centered_cols]
        
        # Use statsmodels Logit for LL and AIC consistency if needed, 
        # but sklearn is often easier for metrics like F1/ROC-AUC.
        # The user requested log-likelihood/AIC for Ordinal, 
        # but for Binary they requested accuracy, ROC-AUC, F1.
        
        clf_add = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
        clf_add.fit(X_train_base, y_train)
        
        preds_add = clf_add.predict(X_test_base)
        probs_add = clf_add.predict_proba(X_test_base)[:, 1]
        
        results['additive'].append({
            'accuracy': accuracy_score(y_test, preds_add),
            'roc_auc': roc_auc_score(y_test, probs_add),
            'f1': f1_score(y_test, preds_add)
        })
        
        # 2. Interaction Model
        G_fold = construct_partial_correlation_network(train_df[CORE_ASPECTS])
        train_df_int, interaction_cols = get_network_interactions(train_df, G_fold)
        test_df_int, _ = get_network_interactions(test_df, G_fold)
        
        X_train_net = train_df_int[base_centered_cols + interaction_cols]
        X_test_net = test_df_int[base_centered_cols + interaction_cols]
        
        clf_int = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
        clf_int.fit(X_train_net, y_train)
        
        preds_int = clf_int.predict(X_test_net)
        probs_int = clf_int.predict_proba(X_test_net)[:, 1]
        
        results['interaction'].append({
            'accuracy': accuracy_score(y_test, preds_int),
            'roc_auc': roc_auc_score(y_test, probs_int),
            'f1': f1_score(y_test, preds_int)
        })
        
        fold += 1
        
    return results

def get_existing_results():
    """
    Returns the already-saved results for Linear models.
    Source: README.md and notebook 03 outputs.
    """
    return {
        'Linear Additive': {
            'Avg RMSE': 1.2144,
            'Avg Adj R2': 0.3351,
            'Full BIC': 1039391.3
        },
        'Linear Interaction': {
            'Avg RMSE': 1.2034,
            'Avg Adj R2': 0.3470,
            'Full BIC': 1032536.3
        }
    }

if __name__ == "__main__":
    import ast
    from src.modeling import prepare_raw_modeling_data
    from src.network_builder import build_and_analyze_network
    
    # Run full diagnostics using local pre-processed data
    logger.info("Running full dataset diagnostics from local file...")
    
    data_path = 'data/Seminar_Amazon_Results_FULL.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if isinstance(df['aspect_sentiments'].iloc[0], str):
            df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    else:
        logger.error(f"Local file {data_path} not found. Run NLP extraction first.")
        sys.exit(1)
    
    # 1. Prepare data for Ordinal Regression
    data = prepare_raw_modeling_data(df)
    for col in CORE_ASPECTS:
        data[f"{col}_centered"] = data[col] - data[col].mean()
    
    base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
    X_add = data[base_centered_cols]
    y = data['rating'].astype(int)
    
    # Multicollinearity Check (VIF) - Additive
    vif_add = run_vif_test(X_add)
    print("\n" + "="*40)
    print("VIF Results (Additive Model)")
    print("-" * 40)
    print(vif_add)
    
    # Brant Test - Additive
    brant_res = run_brant_test(y, X_add)
    print("\n" + "="*40)
    print("Brant Test Results (Parallel Lines Check)")
    print("-" * 40)
    print(brant_res)
    print("="*40)
    
    # 2. Interaction Model Diagnostics
    network_res = build_and_analyze_network(data)
    G = network_res['graph']
    data_int, interaction_cols = get_network_interactions(data, G)
    X_int = data_int[base_centered_cols + interaction_cols]
    
    vif_int = run_vif_test(X_int)
    print("\n" + "="*40)
    print("VIF Results (Interaction Model)")
    print("-" * 40)
    print(vif_int.sort_values(ascending=False).head(20))
    print("="*40)
    
    # Run CV and save
    ord_res = run_ordinal_cv(df)
    bin_res = run_binary_cv(df)
    
    with open('model_ordinal_results.pkl', 'wb') as f:
        pickle.dump(ord_res, f)
    with open('model_binary_results.pkl', 'wb') as f:
        pickle.dump(bin_res, f)
    
    logger.info("Full analysis and diagnostics complete.")
