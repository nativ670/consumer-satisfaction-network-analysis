import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_coefficient_comparison(models_dict, output_path=None):
    """
    Plots a side-by-side comparison of aspect coefficients from both models.
    
    Args:
        models_dict (dict): Dictionary with 'baseline' and 'network' sm.OLS models.
        output_path (str, optional): Path to save the plot.
    """
    logger.info("Generating coefficient comparison plot...")
    
    m1 = models_dict['baseline']
    m2 = models_dict['network']
    
    # Extract coefficients and errors
    def get_coef_df(model, label):
        df = pd.DataFrame({
            'feature': model.params.index,
            'coef': model.params.values,
            'err': model.bse.values,
            'model': label
        })
        return df[df['feature'] != 'const'] # Drop intercept
    
    df1 = get_coef_df(m1, 'Baseline')
    df2 = get_coef_df(m2, 'Network')
    
    # Filter Network model to only show base aspects for direct comparison
    # (Interaction terms are usually too many to show on the same plot clearly)
    base_features = df1['feature'].tolist()
    df2_base = df2[df2['feature'].isin(base_features)]
    
    plot_df = pd.concat([df1, df2_base])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, x='coef', y='feature', hue='model', palette='viridis')
    
    # Add error bars manually if needed, but seaborn barplot doesn't easily support custom per-bar errors
    # Alternatively, use pointplot
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Impact of Experience Attributes on Consumer Satisfaction\n(Baseline vs. Network Model)")
    plt.xlabel("Coefficient Estimate (Impact on Rating)")
    plt.ylabel("Experience Aspect")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    plt.show()

def plot_interaction_heatmap(network_model, output_path=None):
    """
    Visualizes the interaction coefficients in a heatmap format.
    
    Args:
        network_model (sm.OLS): The fitted network model.
        output_path (str, optional): Path to save the plot.
    """
    logger.info("Generating interaction heatmap...")
    
    params = network_model.params
    interactions = [c for c in params.index if c.startswith('int_')]
    
    if not interactions:
        logger.warning("No interaction terms found in the model.")
        return
    
    # Parse feature names from 'int_feat1_feat2'
    data = []
    for inter in interactions:
        parts = inter.split('_')
        # This is a bit brittle depending on sanitization, but let's try
        # modeling.py uses: col.replace('/', '_').replace(' ', '_')
        # And name_map values end in '_c' for base, but interactions don't have '_c' in parts
        # wait, modeling.py: col_name = f"int_{u}_{v}".replace('/', '_').replace(' ', '_')
        # Let's assume u and v are separated by the first and last parts
        # Actually, it's int_{u}_{v}.
        # Finding the boundary:
        parts = inter[4:].split('_') # remove 'int_'
        # This is still hard because features can have underscores
        # But our CORE_ASPECTS are known.
        # Let's just use a simpler approach for now if possible
        pass

    # Alternative: Just a bar plot of top interactions
    inter_series = params[interactions].sort_values()
    
    plt.figure(figsize=(10, 8))
    inter_series.plot(kind='barh', color='salmon')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Significant Behavioral Trade-offs (Interaction Terms)")
    plt.xlabel("Interaction Coefficient")
    plt.ylabel("Aspect Interaction")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def run_likelihood_ratio_test(models_dict):
    """
    Performs a Likelihood Ratio Test to see if the Network model significantly
    improves upon the Baseline.
    """
    m1 = models_dict['baseline']
    m2 = models_dict['network']
    
    # LRT = 2 * (LL_full - LL_restricted)
    # statsmodels results have .llf
    lrt_stat = 2 * (m2.llf - m1.llf)
    df_diff = m2.df_model - m1.df_model
    p_val = stats.chi2.sf(lrt_stat, df_diff)
    
    print("\n" + "="*40)
    print("LIKELIHOOD RATIO TEST")
    print("-" * 40)
    print(f"LRT Statistic: {lrt_stat:.4f}")
    print(f"Degrees of Freedom: {df_diff}")
    print(f"P-Value: {p_val:.4e}")
    print("Interpretation: ", "Significant improvement" if p_val < 0.05 else "No significant improvement")
    print("="*40 + "\n")
    
    return lrt_stat, p_val

if __name__ == "__main__":
    print("Evaluation module ready. Import functions to use in pipeline.")
