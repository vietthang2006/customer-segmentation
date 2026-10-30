from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Default feature subsets
DEFAULT_NUMERIC_FEATURES = [
    'Recency',
    'Frequency',
    'NetSpending',
    'AvgOrderValue',
    'NumDistinctProducts',
    'CustomerTenureDays',
    'PurchaseVelocity_Monthly',
    'ReturnRate',
]

DEFAULT_SKEWED_FEATURES = [
    'Frequency',
    'NetSpending',
    'AvgOrderValue',
    'NumDistinctProducts',
    'PurchaseVelocity_Monthly',
]


def plot_feature_histograms(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    split_by_uk: bool = False,
    bins: int = 30,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """Plot distribution histograms with KDE for numerical features."""
    cols = features or [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(cols):
        ax = axes[idx]
        if split_by_uk and 'IsUK' in df.columns:
            sns.histplot(
                data=df,
                x=col,
                hue='IsUK',
                bins=bins,
                kde=True,
                ax=ax,
                palette={0: '#ff7f0e', 1: '#1f77b4', '0': '#ff7f0e', '1': '#1f77b4'},
                common_norm=False,
            )
            ax.legend(title='IsUK', labels=['UK', 'Non-UK'])
        else:
            sns.histplot(df[col], bins=bins, kde=True, ax=ax, color='#1f77b4') #type: ignore
        ax.set_title(f'Distribution: {col}')
        ax.grid(alpha=0.3)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_feature_boxplots(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    split_by_uk: bool = False,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """Plot boxplots to inspect quartiles and outliers."""
    cols = features or [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(cols):
        ax = axes[idx]
        if split_by_uk and 'IsUK' in df.columns:
            sns.boxplot(
                data=df,
                x='IsUK',
                y=col,
                hue='IsUK',
                order=[0, 1],
                ax=ax,
                palette={0: '#ff7f0e', 1: '#1f77b4', '0': '#ff7f0e', '1': '#1f77b4'},
                legend=False,
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-UK (0)', 'UK (1)'])
        else:
            sns.boxplot(y=df[col], ax=ax, color='#7293cb')
        ax.set_title(f'Boxplot: {col}')
        ax.grid(alpha=0.3)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = 'spearman',
    annot: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot a correlation matrix heatmap (Spearman or Pearson)."""
    cols = features or [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    corr = df[cols].corr(method=method) #type: ignore

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': f'{method.capitalize()} Correlation'},
    )
    ax.set_title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    return fig, ax


def plot_pca_analysis(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_components: int = 4,
    color_by: str = 'IsUK',
    figsize: Tuple[int, int] = (16, 5),
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]: #type: ignore
    """Run PCA and plot cumulative explained variance, 2D projection, and component loadings."""
    cols = features or [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols].dropna())

    pca = PCA(n_components=min(n_components, len(cols)))
    pcs = pca.fit_transform(scaled_data)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Scree Plot
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].bar(range(1, len(cum_var) + 1), pca.explained_variance_ratio_ * 100, alpha=0.6, label='Individual')
    axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='red', label='Cumulative')
    axes[0].axhline(80, color='gray', linestyle='--', label='80% Threshold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained (%)')
    axes[0].set_title('PCA Scree Plot')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. 2D Scatter Projection
    scatter_df = pd.DataFrame({'PC1': pcs[:, 0], 'PC2': pcs[:, 1]})
    if color_by in df.columns:
        scatter_df[color_by] = df[color_by].values
        sns.scatterplot(data=scatter_df, x='PC1', y='PC2', hue=color_by, alpha=0.6, ax=axes[1])
    else:
        sns.scatterplot(data=scatter_df, x='PC1', y='PC2', alpha=0.6, ax=axes[1])
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('PCA 2D Projection (PC1 vs PC2)')
    axes[1].grid(alpha=0.3)

    # 3. Loadings Heatmap
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=cols,
    )
    sns.heatmap(loadings, annot=True, fmt='.2f', cmap='vlag', center=0, ax=axes[2])
    axes[2].set_title('PCA Loadings (PC1 & PC2)')

    plt.tight_layout()
    return fig, axes, loadings


def plot_log_transform_comparison(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    ncols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """Compare raw vs log1p-transformed distributions to evaluate skewness reduction."""
    cols = features or [c for c in DEFAULT_SKEWED_FEATURES if c in df.columns]
    fig, axes = plt.subplots(len(cols), ncols, figsize=figsize or (12, len(cols) * 3))

    for idx, col in enumerate(cols):
        ax_raw = axes[idx, 0] if len(cols) > 1 else axes[0]
        ax_log = axes[idx, 1] if len(cols) > 1 else axes[1]

        # Raw distribution
        sns.histplot(df[col], bins=30, kde=True, ax=ax_raw, color='#4c72b0') #type: ignore
        ax_raw.set_title(f'Raw: {col} (Skew: {df[col].skew():.2f})')
        ax_raw.grid(alpha=0.3)

        # Log1p distribution
        log_series = np.log1p(df[col].clip(lower=0))
        sns.histplot(log_series, bins=30, kde=True, ax=ax_log, color='#55a868')
        ax_log.set_title(f'Log1p: {col} (Skew: {log_series.skew():.2f})') #type: ignore
        ax_log.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_pareto_revenue(
    df: pd.DataFrame,
    spending_col: str = 'NetSpending',
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot the Pareto curve (cumulative revenue % vs. cumulative customer %)."""
    sorted_spend = df[spending_col].sort_values(ascending=False).reset_index(drop=True)
    cum_rev_pct = (sorted_spend.cumsum() / sorted_spend.sum()) * 100
    cum_cust_pct = (np.arange(1, len(sorted_spend) + 1) / len(sorted_spend)) * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cum_cust_pct, cum_rev_pct, color='#2b5c8f', lw=2.5, label='Cumulative Revenue')
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Equality Line')

    # Find 80% revenue cutoff
    idx_80 = np.searchsorted(cum_rev_pct, 80)
    top_cust_80 = cum_cust_pct[min(idx_80, len(cum_cust_pct) - 1)]
    ax.axhline(80, color='red', linestyle=':', alpha=0.7)
    ax.axvline(top_cust_80, color='red', linestyle=':', alpha=0.7)
    ax.scatter([top_cust_80], [80], color='red', s=60, zorder=5)

    ax.set_title(f'Pareto Principle: Top {top_cust_80:.1f}% Customers Generate 80% Revenue')
    ax.set_xlabel('Cumulative % of Customers')
    ax.set_ylabel('Cumulative % of Net Revenue')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig, ax


def plot_rfm_bubble_scatter(
    df: pd.DataFrame,
    x_col: str = 'Recency',
    y_col: str = 'NetSpending',
    size_col: str = 'Frequency',
    hue_col: str = 'ReturnRate',
    figsize: Tuple[int, int] = (10, 7),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot a 4D bubble chart: Recency vs NetSpending (log), size = Frequency, color = ReturnRate."""
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize bubble size for visibility
    sizes = np.interp(df[size_col], (df[size_col].min(), df[size_col].quantile(0.99)), (20, 300))

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        s=sizes,
        c=df[hue_col],
        cmap='viridis',
        alpha=0.6,
        edgecolors='none',
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(hue_col)

    ax.set_yscale('log')
    ax.set_xlabel(f'{x_col} (days)')
    ax.set_ylabel(f'{y_col} (Log Scale)')
    ax.set_title(f'RFM Landscape: {x_col} vs {y_col} (Size={size_col}, Color={hue_col})')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    return fig, ax


def plot_tenure_vs_recency(
    df: pd.DataFrame,
    tenure_col: str = 'CustomerTenureDays',
    recency_col: str = 'Recency',
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot customer lifespan triangle (Tenure vs Recency with y = x boundary)."""
    fig, ax = plt.subplots(figsize=figsize)

    # Identify one-time buyers (Recency approx equal to Tenure)
    is_one_time = (df[recency_col] >= df[tenure_col] - 1) & (df['Frequency'] == 1) if 'Frequency' in df.columns else False

    sns.scatterplot(
        data=df,
        x=tenure_col,
        y=recency_col,
        alpha=0.5,
        ax=ax,
        color='#4c72b0',
        label='Repeat Buyers',
    )
    if isinstance(is_one_time, pd.Series) and is_one_time.any():
        ax.scatter(
            df.loc[is_one_time, tenure_col],
            df.loc[is_one_time, recency_col],
            color='#c44e52',
            s=15,
            alpha=0.6,
            label='One-time Buyers (y ≈ x)',
        )

    # Upper boundary line y = x
    max_val = max(df[tenure_col].max(), df[recency_col].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Boundary (y = x)')

    ax.set_xlabel(f'{tenure_col} (days)')
    ax.set_ylabel(f'{recency_col} (days)')
    ax.set_title('Customer Lifespan: Tenure vs Recency')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_return_behavior_analysis(
    df: pd.DataFrame,
    return_rate_col: str = 'ReturnRate',
    spending_col: str = 'GrossSpending',
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """Analyze zero-inflated return behavior: proportion of returners and rate distribution."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Non-returners vs Returners breakdown
    has_returns = df[return_rate_col] > 0
    counts = [sum(~has_returns), sum(has_returns)]
    axes[0].pie(
        counts,
        labels=['Zero Returns', 'Has Returns'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#55a868', '#c44e52'],
        explode=(0, 0.05),
    )
    axes[0].set_title('Customer Base: Zero vs Positive Returns')

    # 2. ReturnRate distribution for returners only
    returners = df[has_returns]
    if not returners.empty:
        sns.histplot(returners[return_rate_col], bins=25, kde=True, ax=axes[1], color='#c44e52') #type: ignore
        axes[1].set_title(f'ReturnRate Distribution (Returners only, N={len(returners)})')
        axes[1].set_xlabel('ReturnRate (ReturnedAmount / GrossSpending)')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No returns in dataset', ha='center', va='center')

    plt.tight_layout()
    return fig, axes


def plot_wholesale_vs_retail(
    df: pd.DataFrame,
    distinct_prod_col: str = 'NumDistinctProducts',
    items_per_order_col: str = 'AvgItemsPerOrder',
    color_by: str = 'IsUK',
    figsize: Tuple[int, int] = (9, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """Plot product variety vs order item volume to distinguish retail vs wholesale buyers."""
    fig, ax = plt.subplots(figsize=figsize)

    if color_by in df.columns:
        sns.scatterplot(
            data=df,
            x=distinct_prod_col,
            y=items_per_order_col,
            hue=color_by,
            alpha=0.6,
            ax=ax,
            palette={0: '#ff7f0e', 1: '#1f77b4', '0': '#ff7f0e', '1': '#1f77b4'},
        )
        ax.legend(title=color_by, labels=['UK (1)', 'Non-UK (0)'])
    else:
        sns.scatterplot(data=df, x=distinct_prod_col, y=items_per_order_col, alpha=0.6, ax=ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'{distinct_prod_col} (Log Scale)')
    ax.set_ylabel(f'{items_per_order_col} (Log Scale)')
    ax.set_title('Buyer Typology: Variety (Distinct Products) vs Volume (Items/Order)')
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    return fig, ax
