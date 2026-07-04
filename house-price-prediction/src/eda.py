import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def print_section(title: str):
    logger.info("\n" + "=" * 50 + f"\n {title}\n" + "=" * 50)

def run_basic_eda(df: pd.DataFrame, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print_section("Dataset Overview")
    logger.info(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    logger.info("Sample columns and data types:")
    for col, dtype in list(df.dtypes.items())[:10]:
        logger.info(f" - {col}: {dtype}")
    if len(df.columns) > 10:
        logger.info(f" ... and {len(df.columns) - 10} more columns.")

    print_section("Missing Values Analysis")
    missing_data = check_missing_values(df)
    
    print_section("Summary Statistics")
    summary_stats = get_summary_statistics(df)
    
    print_section("Generating Visualizations")
    
    if 'SalePrice' in df.columns:
        plot_target_distribution(df, 'SalePrice', output_dir)
        
    plot_correlation_heatmap(df, output_dir)
    plot_scatters(df, output_dir)
    plot_pairs(df, output_dir)
    
    logger.info(f"EDA successfully completed. Visualizations saved to: {output_dir}")

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percentage (%)'])
    missing_data = missing_data[missing_data['Total Missing'] > 0].sort_values(by='Total Missing', ascending=False)
    
    if len(missing_data) > 0:
        logger.info(f"Top 10 columns with missing values:\n{missing_data.head(10)}")
    else:
        logger.info("No missing values found in the dataset.")
        
    return missing_data

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    logger.info(f"Numerical summary stats (subset):\n{stats.head(10)}")
    return stats

def plot_target_distribution(df: pd.DataFrame, target_col: str, output_dir: str):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True, color='#2c3e50')
    plt.title(f'Original {target_col} Distribution')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df[target_col]), kde=True, color='#16a085')
    plt.title(f'Log-Transformed {target_col} Distribution')
    plt.xlabel(f'log({target_col} + 1)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'target_distribution.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved target distribution plot to {plot_path}")

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    numeric_df = df.select_dtypes(include=[np.number])
    
    if 'SalePrice' not in numeric_df.columns:
        logger.warning("SalePrice not found in dataframe. Skipping correlation heatmap.")
        return

    corrs = numeric_df.corr()
    top_cols = corrs['SalePrice'].abs().sort_values(ascending=False).head(top_n).index
    top_corr_matrix = numeric_df[top_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        top_corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title(f'Correlation Matrix (Top {top_n} Features with SalePrice)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved correlation heatmap to {plot_path}")

def plot_scatters(df: pd.DataFrame, output_dir: str):
    features_to_plot = ['GrLivArea', 'OverallQual', 'YearBuilt', 'GarageCars']
    present_features = [f for f in features_to_plot if f in df.columns]
    
    if 'SalePrice' not in df.columns or not present_features:
        logger.warning("Cannot plot scatters. Target or features missing.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        if feature in df.columns:
            sns.scatterplot(data=df, x=feature, y='SalePrice', ax=axes[idx], alpha=0.6, color='#2980b9')
            sns.regplot(data=df, x=feature, y='SalePrice', ax=axes[idx], scatter=False, color='#e74c3c')
            axes[idx].set_title(f'{feature} vs SalePrice')
            axes[idx].set_ylabel('SalePrice ($)')
        else:
            axes[idx].text(0.5, 0.5, f"Feature '{feature}'\nnot in dataset", 
                           ha='center', va='center', fontsize=12, color='gray')
            axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scatter_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved joint scatter plots to {plot_path}")

def plot_pairs(df: pd.DataFrame, output_dir: str):
    cols = ['SalePrice', 'GrLivArea', 'OverallQual', 'FullBath']
    present_cols = [c for c in cols if c in df.columns]
    
    if len(present_cols) < 2:
        logger.warning("Not enough features present for pair plot.")
        return

    sample_df = df[present_cols].dropna().sample(min(len(df), 500), random_state=42, replace=len(df) < 500)
    
    g = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6, 'color': '#16a085'}, diag_kws={'color': '#16a085'})
    g.fig.suptitle('Pairwise Distributions of Key Features', y=1.02, fontsize=14)
    
    plot_path = os.path.join(output_dir, 'pair_plots.png')
    g.savefig(plot_path, dpi=120)
    plt.close()
    logger.info(f"Saved pair plots to {plot_path}")
