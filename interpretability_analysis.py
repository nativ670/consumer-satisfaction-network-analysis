import pandas as pd
import numpy as np
import logging
import os
import sys
import ast
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score

# Add parent directory to path for imports
sys.path.append(os.path.abspath('.'))

from src.modeling import prepare_raw_modeling_data, get_network_interactions, CORE_ASPECTS
from src.network_builder import construct_partial_correlation_network
from src.modeling_alternative import run_brant_test

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = 'interpretability'
DATA_PATH = 'data/Seminar_Amazon_Results_FULL.csv'
RANDOM_STATE = 42

def setup_interpretability():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.rcParams.update({'font.size': 10})

def load_and_preprocess_data():
    logger.info(f"Loading full dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"File {DATA_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    if isinstance(df['aspect_sentiments'].iloc[0], str):
        df['aspect_sentiments'] = df['aspect_sentiments'].apply(ast.literal_eval)
    
    # 1. Prepare raw modeling data (pivoted scores + rating)
    data = prepare_raw_modeling_data(df)
    
    # 2. Mean-center aspect scores
    for col in CORE_ASPECTS:
        data[f"{col}_centered"] = data[col] - data[col].mean()
        
    return data

def get_statsmodels_summary_df(results):
    """Extracts coefficients and metrics from statsmodels results into a DataFrame."""
    df = pd.DataFrame({
        'feature': results.params.index,
        'coefficient': results.params.values,
        'std_error': results.bse.values,
        't_statistic': results.tvalues if hasattr(results, 'tvalues') else results.zvalues,
        'p_value': results.pvalues.values,
        'ci_lower_95': results.conf_int()[0].values,
        'ci_upper_95': results.conf_int()[1].values
    })
    if 't_statistic' in df.columns and not hasattr(results, 'tvalues'):
        df = df.rename(columns={'t_statistic': 'z_statistic'})
    
    df['significant'] = df['p_value'] < 0.05
    return df

def ax_forest(ax, df_model, title):
    """Refined forest plot logic to handle colors correctly and avoid ValueErrors."""
    df_model = df_model[df_model['feature'] != 'const'].copy()
    df_model = df_model.sort_values(by='coefficient', ascending=True)
    
    y_pos = np.arange(len(df_model))
    for i, (_, row) in enumerate(df_model.iterrows()):
        color = 'gray'
        if row['significant']:
            color = 'green' if row['coefficient'] > 0 else 'red'
        
        # Plot single point with CI
        ax.errorbar(row['coefficient'], i, 
                     xerr=[[row['coefficient'] - row['ci_lower_95']], [row['ci_upper_95'] - row['coefficient']]],
                     fmt='o', color='black', ecolor=color, capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_model['feature'])
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Coefficient Estimate (95% CI)")

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

def run_interpretability_analysis():
    setup_interpretability()
    data = load_and_preprocess_data()
    
    base_centered_cols = [f"{col}_centered" for col in CORE_ASPECTS]
    y = data['rating']
    N = len(data)
    
    # documentation list
    sig_features = {}
    bic_results = {}
    
    # --- A. Linear Models ---
    logger.info("Fitting Linear Models...")
    
    # A1. Additive
    X_add = sm.add_constant(data[base_centered_cols])
    model_lin_add = sm.OLS(y, X_add).fit()
    df_lin_add = get_statsmodels_summary_df(model_lin_add)
    df_lin_add['model'] = 'Additive'
    bic_results['Linear Additive'] = model_lin_add.bic
    
    # A2. Interaction
    # Build network on raw scores matrix
    G = construct_partial_correlation_network(data[CORE_ASPECTS])
    edges = list(G.edges())
    data_int, interaction_cols = get_network_interactions(data, G)
    X_int = sm.add_constant(data_int[base_centered_cols + interaction_cols])
    model_lin_int = sm.OLS(y, X_int).fit()
    df_lin_int = get_statsmodels_summary_df(model_lin_int)
    df_lin_int['model'] = 'Interaction'
    bic_results['Linear Interaction'] = model_lin_int.bic
    
    # Merge and add Beta Weights
    df_linear = pd.concat([df_lin_add, df_lin_int])
    y_std = y.std()
    
    def calc_std_coef(row):
        if row['feature'] == 'const': return np.nan
        feat_name = row['feature']
        if feat_name in data_int.columns:
            return row['coefficient'] * (data_int[feat_name].std() / y_std)
        return np.nan

    df_linear['std_coefficient'] = df_linear.apply(calc_std_coef, axis=1)
    df_linear.to_csv(f"{OUTPUT_DIR}/linear_coefficients.csv", index=False)
    
    sig_features['Linear Additive'] = df_lin_add[df_lin_add['significant']]['feature'].tolist()
    sig_features['Linear Interaction'] = df_lin_int[df_lin_int['significant']]['feature'].tolist()

    # A3. Partial R2
    logger.info("Computing Partial R2 for Linear Model...")
    partial_r2_results = []
    full_r2 = model_lin_int.rsquared
    features_to_test = base_centered_cols + interaction_cols
    for feat in features_to_test:
        cols_reduced = [f for f in features_to_test if f != feat]
        model_red = sm.OLS(y, sm.add_constant(data_int[cols_reduced])).fit()
        drop = full_r2 - model_red.rsquared
        partial_r2_results.append({
            'feature': feat,
            'partial_r2': drop,
            'pct_variance_explained': (drop / full_r2) * 100 if full_r2 > 0 else 0
        })
    pd.DataFrame(partial_r2_results).to_csv(f"{OUTPUT_DIR}/linear_partial_r2.csv", index=False)

    # A4. Plot linear coefficients
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_forest(axes[0], df_lin_add, "Linear Additive Coefficients")
    ax_forest(axes[1], df_lin_int, "Linear Interaction Coefficients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/linear_coefficient_plot.png")
    plt.close()

    # A5. Interaction Heatmap
    logger.info("Generating Interaction Heatmap...")
    heatmap_data = pd.DataFrame(0.0, index=CORE_ASPECTS, columns=CORE_ASPECTS)
    for u, v in edges:
        col_name = f"int_{u}_{v}".replace('/', '_').replace(' ', '_')
        if col_name in df_lin_int['feature'].values:
            val = df_lin_int[df_lin_int['feature'] == col_name]['coefficient'].values[0]
            heatmap_data.loc[u, v] = val
            heatmap_data.loc[v, u] = val
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='RdBu', center=0)
    plt.title("Interaction Coefficients (Linear Model)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/interaction_heatmap.png")
    plt.close()

    # --- B. Ordinal Models ---
    logger.info("Fitting Ordinal Models...")
    y_ord = y.astype(int)
    rating_categories = [1, 2, 3, 4, 5]
    
    # B1. Additive
    model_ord_add = OrderedModel(y_ord, data[base_centered_cols], distr='logit')
    res_ord_add = model_ord_add.fit(method='bfgs', disp=False)
    df_ord_add = get_statsmodels_summary_df(res_ord_add)
    df_ord_add['model'] = 'Additive'
    bic_results['Ordinal Additive'] = res_ord_add.bic
    
    # Confusion Matrix (Additive)
    probs_ord_add = res_ord_add.predict(data[base_centered_cols]).values
    preds_ord_add = [i + 1 for i in np.argmax(probs_ord_add, axis=1)]
    cm_ord_add = confusion_matrix(y_ord, preds_ord_add, labels=rating_categories)
    plot_confusion_matrix(cm_ord_add, rating_categories, "Ordinal Additive Confusion Matrix", "ordinal_add_cm.png")
    pd.DataFrame(cm_ord_add, index=rating_categories, columns=rating_categories).to_csv(f"{OUTPUT_DIR}/ordinal_add_cm.csv")

    # B2. Interaction
    model_ord_int = OrderedModel(y_ord, data_int[base_centered_cols + interaction_cols], distr='logit')
    res_ord_int = model_ord_int.fit(method='bfgs', disp=False)
    df_ord_int = get_statsmodels_summary_df(res_ord_int)
    df_ord_int['model'] = 'Interaction'
    bic_results['Ordinal Interaction'] = res_ord_int.bic

    # Confusion Matrix (Interaction)
    probs_ord_int = res_ord_int.predict(data_int[base_centered_cols + interaction_cols]).values
    preds_ord_int = [i + 1 for i in np.argmax(probs_ord_int, axis=1)]
    cm_ord_int = confusion_matrix(y_ord, preds_ord_int, labels=rating_categories)
    plot_confusion_matrix(cm_ord_int, rating_categories, "Ordinal Interaction Confusion Matrix", "ordinal_int_cm.png")
    pd.DataFrame(cm_ord_int, index=rating_categories, columns=rating_categories).to_csv(f"{OUTPUT_DIR}/ordinal_int_cm.csv")
    
    def label_ord_features(df):
        # A feature is a threshold if it's not in CORE_ASPECTS and doesn't start with int_ and not in base_centered_cols
        is_threshold = df['feature'].apply(lambda x: ('/' in str(x) or '|' in str(x)) and not any(a in str(x) for a in CORE_ASPECTS))
        df['feature_type'] = np.where(is_threshold, 'threshold', 'feature')
        # Rename thresholds for consistency
        t_map = {'1/2': 'threshold_1|2', '2/3': 'threshold_2|3', '3/4': 'threshold_3|4', '4/5': 'threshold_4|5',
                 'rating.1|rating.2': 'threshold_1|2', 'rating.2|rating.3': 'threshold_2|3', 
                 'rating.3|rating.4': 'threshold_3|4', 'rating.4|rating.5': 'threshold_4|5'}
        df['feature'] = df['feature'].apply(lambda x: t_map.get(str(x), x))
        return df

    df_ord_add = label_ord_features(df_ord_add)
    df_ord_int = label_ord_features(df_ord_int)
    df_ordinal = pd.concat([df_ord_add, df_ord_int])
    
    df_ordinal['odds_ratio'] = np.where(df_ordinal['feature_type'] == 'feature', np.exp(df_ordinal['coefficient']), np.nan)
    df_ordinal['or_ci_lower'] = np.where(df_ordinal['feature_type'] == 'feature', np.exp(df_ordinal['ci_lower_95']), np.nan)
    df_ordinal['or_ci_upper'] = np.where(df_ordinal['feature_type'] == 'feature', np.exp(df_ordinal['ci_upper_95']), np.nan)
    df_ordinal.to_csv(f"{OUTPUT_DIR}/ordinal_coefficients.csv", index=False)
    
    sig_features['Ordinal Additive'] = df_ord_add[df_ord_add['significant'] & (df_ord_add['feature_type'] == 'feature')]['feature'].tolist()
    sig_features['Ordinal Interaction'] = df_ord_int[df_ord_int['significant'] & (df_ord_int['feature_type'] == 'feature')]['feature'].tolist()

    # B3. Marginal Effects at Mean
    logger.info("Computing Ordinal Marginal Effects at Mean...")
    mem_results = []
    
    def compute_ord_mem(res, X_data, model_name):
        means = X_data.mean().to_frame().T
        valid_features = [f for f in res.params.index if f in X_data.columns]
        
        probs = res.predict(means).values[0] # [P(1), P(2), P(3), P(4), P(5)]
        
        delta = 0.001
        m_effects = []
        for feat in valid_features:
            means_plus = means.copy()
            means_plus[feat] += delta
            probs_plus = res.predict(means_plus).values[0]
            m_effects.append({
                'feature': feat,
                'model': model_name,
                'me_prob_rating_1': (probs_plus[0] - probs[0]) / delta,
                'me_prob_rating_5': (probs_plus[4] - probs[4]) / delta
            })
        return m_effects

    mem_add = compute_ord_mem(res_ord_add, data[base_centered_cols], 'Additive')
    mem_int = compute_ord_mem(res_ord_int, data_int[base_centered_cols + interaction_cols], 'Interaction')
    
    mem_results.extend(mem_add)
    mem_results.extend(mem_int)
    pd.DataFrame(mem_results).to_csv(f"{OUTPUT_DIR}/ordinal_marginal_effects.csv", index=False)

    # B4. Brant Test
    logger.info("Extracting Brant Test results...")
    brant_df = run_brant_test(y_ord, data[base_centered_cols])
    brant_results = []
    for feat in base_centered_cols:
        row = brant_df.loc[feat]
        max_var = row.max() - row.min()
        brant_results.append({
            'feature': feat,
            'coef_y_gt_1': row['Y > 1'],
            'coef_y_gt_2': row['Y > 2'],
            'coef_y_gt_3': row['Y > 3'],
            'coef_y_gt_4': row['Y > 4'],
            'max_variation': max_var,
            'proportional_odds_violated': max_var > 0.3
        })
    pd.DataFrame(brant_results).to_csv(f"{OUTPUT_DIR}/brant_test_results.csv", index=False)

    # B5. Odds Ratio Plot
    plt.figure(figsize=(12, 8))
    df_or_plot = df_ordinal[(df_ordinal['feature_type'] == 'feature') & (df_ordinal['feature'].isin(base_centered_cols))]
    sns.barplot(data=df_or_plot, x='odds_ratio', y='feature', hue='model')
    plt.axvline(1, color='black', linestyle='--', alpha=0.5)
    plt.title("Ordinal Odds Ratios (Main Effects)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ordinal_odds_ratio_plot.png")
    plt.close()

    # B6. Marginal Effects Plot
    plt.figure(figsize=(12, 6))
    df_mem = pd.DataFrame(mem_results)
    df_mem_plot = df_mem[df_mem['feature'].isin(base_centered_cols)]
    df_mem_melt = df_mem_plot.melt(id_vars=['feature', 'model'], value_vars=['me_prob_rating_1', 'me_prob_rating_5'])
    sns.barplot(data=df_mem_melt, x='value', y='feature', hue='variable')
    plt.title("Marginal Effects on P(Rating=1) and P(Rating=5) at Mean")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ordinal_marginal_effects_plot.png")
    plt.close()

    # --- C. Binary Models ---
    logger.info("Fitting Binary Models...")
    y_bin = (y_ord == 5).astype(int)
    binary_labels = [0, 1]
    
    # C1. Additive
    model_bin_add = sm.Logit(y_bin, sm.add_constant(data[base_centered_cols])).fit(method='bfgs', maxiter=500, disp=False)
    df_bin_add = get_statsmodels_summary_df(model_bin_add)
    df_bin_add['model'] = 'Additive'
    bic_results['Binary Additive'] = model_bin_add.bic
    
    # Confusion Matrix (Additive)
    probs_bin_add = model_bin_add.predict(sm.add_constant(data[base_centered_cols]))
    preds_bin_add = (probs_bin_add > 0.5).astype(int)
    cm_bin_add = confusion_matrix(y_bin, preds_bin_add, labels=binary_labels)
    plot_confusion_matrix(cm_bin_add, ["1-4 Stars", "5 Stars"], "Binary Additive Confusion Matrix", "binary_add_cm.png")
    pd.DataFrame(cm_bin_add, index=binary_labels, columns=binary_labels).to_csv(f"{OUTPUT_DIR}/binary_add_cm.csv")

    # C2. Interaction
    model_bin_int = sm.Logit(y_bin, sm.add_constant(data_int[base_centered_cols + interaction_cols])).fit(method='bfgs', maxiter=500, disp=False)
    df_bin_int = get_statsmodels_summary_df(model_bin_int)
    df_bin_int['model'] = 'Interaction'
    bic_results['Binary Interaction'] = model_bin_int.bic

    # Confusion Matrix (Interaction)
    probs_bin_int = model_bin_int.predict(sm.add_constant(data_int[base_centered_cols + interaction_cols]))
    preds_bin_int = (probs_bin_int > 0.5).astype(int)
    cm_bin_int = confusion_matrix(y_bin, preds_bin_int, labels=binary_labels)
    plot_confusion_matrix(cm_bin_int, ["1-4 Stars", "5 Stars"], "Binary Interaction Confusion Matrix", "binary_int_cm.png")
    pd.DataFrame(cm_bin_int, index=binary_labels, columns=binary_labels).to_csv(f"{OUTPUT_DIR}/binary_int_cm.csv")
    
    df_binary = pd.concat([df_bin_add, df_bin_int])
    df_binary['odds_ratio'] = np.exp(df_binary['coefficient'])
    df_binary['or_ci_lower'] = np.exp(df_binary['ci_lower_95'])
    df_binary['or_ci_upper'] = np.exp(df_binary['ci_upper_95'])
    df_binary.to_csv(f"{OUTPUT_DIR}/binary_coefficients.csv", index=False)
    
    sig_features['Binary Additive'] = df_bin_add[df_bin_add['significant']]['feature'].tolist()
    sig_features['Binary Interaction'] = df_bin_int[df_bin_int['significant']]['feature'].tolist()

    # C3. Average Marginal Effects (AME)
    logger.info("Computing Binary AME...")
    def get_ame_df(model, model_name):
        ame = model.get_margeff().summary_frame()
        # Rename columns safely
        col_map = {}
        for c in ame.columns:
            if 'dy/dx' in c: col_map[c] = 'ame'
            if 'Std. Err.' in c: col_map[c] = 'std_error'
            if 'P>' in c or 'Pr(>' in c: col_map[c] = 'p_value'
        ame = ame.rename(columns=col_map)
        ame['feature'] = ame.index
        ame['model'] = model_name
        return ame

    ame_add = get_ame_df(model_bin_add, 'Additive')
    ame_int = get_ame_df(model_bin_int, 'Interaction')
    
    df_ame = pd.concat([ame_add, ame_int])
    df_ame[['feature', 'ame', 'std_error', 'p_value', 'model']].to_csv(f"{OUTPUT_DIR}/binary_ame.csv", index=False)

    # C4. Odds Ratio Plot (Binary)
    plt.figure(figsize=(12, 8))
    df_bin_or = df_binary[df_binary['feature'].isin(base_centered_cols)]
    sns.barplot(data=df_bin_or, x='odds_ratio', y='feature', hue='model')
    plt.axvline(1, color='black', linestyle='--', alpha=0.5)
    plt.title("Binary Logic Odds Ratios (Main Effects on P(5-star))")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/binary_odds_ratio_plot.png")
    plt.close()

    # C5. AME Plot
    plt.figure(figsize=(12, 8))
    df_ame_plot = df_ame[df_ame['feature'].isin(base_centered_cols)]
    sns.barplot(data=df_ame_plot, x='ame', y='feature', hue='model')
    plt.title("Average Marginal Effects on P(5-star)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/binary_ame_plot.png")
    plt.close()

    # --- D. Cross-Model Comparison ---
    logger.info("Performing Cross-Model Comparison...")
    comparison_data = []
    for aspect in base_centered_cols:
        row = {'feature': aspect}
        row['linear_add_std_coef'] = df_linear[(df_linear['feature'] == aspect) & (df_linear['model'] == 'Additive')]['std_coefficient'].values[0]
        row['linear_int_std_coef'] = df_linear[(df_linear['feature'] == aspect) & (df_linear['model'] == 'Interaction')]['std_coefficient'].values[0]
        row['ordinal_add_or'] = df_ordinal[(df_ordinal['feature'] == aspect) & (df_ordinal['model'] == 'Additive')]['odds_ratio'].values[0]
        row['ordinal_int_or'] = df_ordinal[(df_ordinal['feature'] == aspect) & (df_ordinal['model'] == 'Interaction')]['odds_ratio'].values[0]
        row['binary_add_or'] = df_binary[(df_binary['feature'] == aspect) & (df_binary['model'] == 'Additive')]['odds_ratio'].values[0]
        row['binary_int_or'] = df_binary[(df_binary['feature'] == aspect) & (df_binary['model'] == 'Interaction')]['odds_ratio'].values[0]
        comparison_data.append(row)
    pd.DataFrame(comparison_data).to_csv(f"{OUTPUT_DIR}/cross_model_comparison.csv", index=False)

    # Cross-Model Heatmap
    hm_df = pd.DataFrame(comparison_data).set_index('feature')
    hm_norm = (hm_df - hm_df.mean()) / hm_df.std()
    plt.figure(figsize=(12, 8))
    sns.heatmap(hm_norm, annot=hm_df.round(2), cmap='RdBu', center=0)
    plt.title("Cross-Model Coefficient Comparison (Normalized Effects)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cross_model_heatmap.png")
    plt.close()

    # Interaction Summary
    inter_summary = []
    for feat in interaction_cols:
        l_row = df_lin_int[df_lin_int['feature'] == feat]
        o_row = df_ord_int[df_ord_int['feature'] == feat]
        b_row = df_bin_int[df_bin_int['feature'] == feat]
        l_c, l_p = l_row['coefficient'].values[0], l_row['p_value'].values[0]
        o_c, o_p = o_row['coefficient'].values[0], o_row['p_value'].values[0]
        b_c, b_p = b_row['coefficient'].values[0], b_row['p_value'].values[0]
        consistent = (np.sign(l_c) == np.sign(o_c) == np.sign(b_c))
        inter_summary.append({
            'interaction_pair': feat,
            'linear_coef': l_c, 'linear_p': l_p,
            'ordinal_coef': o_c, 'ordinal_p': o_p,
            'binary_coef': b_c, 'binary_p': b_p,
            'consistent_sign': consistent
        })
    pd.DataFrame(inter_summary).to_csv(f"{OUTPUT_DIR}/interaction_summary.csv", index=False)

    # --- E. Run Summary ---
    with open(f"{OUTPUT_DIR}/run_summary.txt", "w") as f:
        f.write(f"Interpretability Analysis Run Summary\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"N reviews used: {N}\n\n")
        
        f.write(f"MODEL COMPLEXITY (Full Dataset BIC):\n")
        f.write(f"{'Model':<25} | {'BIC':<15}\n")
        f.write(f"{'-'*40}\n")
        for model_name, bic_val in bic_results.items():
            f.write(f"{model_name:<25} | {bic_val:,.2f}\n")
        f.write("\n")
        
        f.write(f"GLASSO EDGES SELECTED:\n")
        for u, v in edges:
            f.write(f" - {u} <-> {v}\n")
        f.write(f"\nSIGNIFICANT COEFFICIENTS PER MODEL:\n")
        for model, feats in sig_features.items():
            f.write(f" - {model}: {', '.join(feats)}\n")

    logger.info("Analysis complete. All outputs saved to interpretability/ folder.")

if __name__ == "__main__":
    run_interpretability_analysis()
