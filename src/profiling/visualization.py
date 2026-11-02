"""
Visualization module for Customer Profiling.
Provides visualizations for business stakeholders:
- Revenue vs. Population Share Donut charts (Pareto distribution)
- Raw Feature Distribution Boxplots by Cluster
- DBSCAN Anomaly Landscape scatter plots
- Segment Relative Profile Comparison bar charts
"""

from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CLUSTER_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10)) #type: ignore


def _get_cluster_colors(n_clusters: int) -> List[Tuple[float, float, float, float]]:
    """Return consistent tab10 colormap matching the clustering projections."""
    return [tuple(c) for c in plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))] #type: ignore


def plot_revenue_vs_population_donut(
    df_segmented: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """
    Plot side-by-side Donut Charts comparing Customer Population Share (%)
    against Total Net Revenue Share (%) across clusters.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe containing 'Cluster', 'NetSpending', and optional 'Cluster_Name'.
    figsize : tuple of (int, int), default=(14, 6)
        Figure dimensions.

    Returns
    -------
    fig, axes
    """
    if 'Cluster' not in df_segmented.columns or 'NetSpending' not in df_segmented.columns:
        raise ValueError("df_segmented must contain 'Cluster' and 'NetSpending' columns.")

    summary = (
        df_segmented.groupby('Cluster')
        .agg(
            Customers=('CustomerID', 'count'),
            NetRevenue=('NetSpending', 'sum'),
        )
        .reset_index()
    )

    if 'Cluster_Name' in df_segmented.columns:
        name_map = df_segmented.groupby('Cluster')['Cluster_Name'].first().to_dict()
        summary['Label'] = summary['Cluster'].map(
            lambda c: f"Cluster {c}\n({name_map.get(c, '')})"
        )
    else:
        summary['Label'] = summary['Cluster'].apply(lambda c: f"Cluster {c}")

    n_clusters = len(summary)
    palette = _get_cluster_colors(n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Donut Chart - Customer Population
    ax1 = axes[0]
    wedges1, texts1, autotexts1 = ax1.pie(
        summary['Customers'],
        labels=summary['Label'],
        autopct='%1.1f%%',
        startangle=140,
        colors=palette[:n_clusters],
        pctdistance=0.75,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10},
    )
    for at in autotexts1:
        at.set_fontsize(10.5)
        at.set_fontweight('bold')
    ax1.set_title(
        f"Customer Population Share\n(Total N = {len(df_segmented):,})",
        fontsize=13,
        fontweight='bold',
        pad=15,
    )

    # 2. Donut Chart - Net Revenue Contribution
    ax2 = axes[1]
    total_rev = float(summary['NetRevenue'].sum())
    wedges2, texts2, autotexts2 = ax2.pie(
        summary['NetRevenue'],
        labels=summary['Label'],
        autopct='%1.1f%%',
        startangle=140,
        colors=palette[:n_clusters],
        pctdistance=0.75,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10},
    )
    for at in autotexts2:
        at.set_fontsize(10.5)
        at.set_fontweight('bold')
    ax2.set_title(
        f"Net Revenue Share\n(Total Revenue = £{total_rev:,.0f})",
        fontsize=13,
        fontweight='bold',
        pad=15,
    )

    plt.suptitle(
        "Pareto Disproportion: Customer Share vs. Revenue Contribution by Segment",
        fontsize=15,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    return fig, axes


def plot_raw_feature_boxplots_by_cluster(
    df_segmented: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    log_scale_features: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (16, 11),
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """
    Plot boxplots of unscaled, real-world business metrics grouped by cluster.
    Applies log-scale on skewed monetary and frequency metrics to prevent outlier compression.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe containing 'Cluster' and numerical features.
    features : sequence of str, optional
        Features to display. Defaults to core 8 behavior features.
    log_scale_features : sequence of str, optional
        Features to display on logarithmic y-scale (e.g. NetSpending, Frequency, AOV).
    figsize : tuple of (int, int), default=(16, 11)

    Returns
    -------
    fig, axes
    """
    if 'Cluster' not in df_segmented.columns:
        raise ValueError("df_segmented must contain a 'Cluster' column.")

    default_features = [
        'Recency',
        'Frequency',
        'NetSpending',
        'AvgOrderValue',
        'NumDistinctProducts',
        'CustomerTenureDays',
        'PurchaseVelocity_Monthly',
        'ReturnRate',
    ]
    feats = list(features) if features is not None else [f for f in default_features if f in df_segmented.columns]
    log_feats = set(log_scale_features) if log_scale_features is not None else {
        'Frequency',
        'NetSpending',
        'AvgOrderValue',
        'NumDistinctProducts',
        'PurchaseVelocity_Monthly',
    }

    n_feats = len(feats)
    n_cols = 4 if n_feats >= 4 else n_feats
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    unique_clusters = sorted(df_segmented['Cluster'].unique())
    n_clusters = len(unique_clusters)
    palette = _get_cluster_colors(n_clusters)

    # Unit labels for clarity
    unit_map = {
        'Recency': 'Days',
        'Frequency': 'Orders',
        'NetSpending': 'GBP (£)',
        'AvgOrderValue': 'GBP (£)',
        'NumDistinctProducts': 'Unique SKUs',
        'CustomerTenureDays': 'Days',
        'PurchaseVelocity_Monthly': 'Orders / Month',
        'ReturnRate': 'Proportion (0 - 1)',
        'GrossSpending': 'GBP (£)',
        'ReturnedAmount': 'GBP (£)',
    }

    for idx, feat in enumerate(feats):
        ax = axes_flat[idx]
        data_to_plot = [
            df_segmented.loc[df_segmented['Cluster'] == cl, feat].dropna().values
            for cl in unique_clusters
        ]

        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=[f"C{cl}" for cl in unique_clusters],
            showfliers=True,
            flierprops=dict(marker='.', markersize=3, alpha=0.3, color='gray'),
            medianprops=dict(color='black', linewidth=1.8),
        )

        for patch, color in zip(bplot['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        unit = unit_map.get(feat, '')
        ax.set_title(f"{feat} ({unit})" if unit else feat, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')

        if feat in log_feats:
            # Check minimum positive value to avoid log(<=0) crash
            min_val = df_segmented[feat].min()
            if min_val > 0:
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=1.0)
            ax.set_ylabel('Log Scale', fontsize=8.5, alpha=0.7)

    # Hide extra unused subplots
    for j in range(n_feats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        "Raw Business Metric Distributions Across Segments (Unscaled Metrics)",
        fontsize=15,
        fontweight='bold',
        y=1.01,
    )
    plt.tight_layout()
    return fig, axes_flat


def plot_anomaly_scatter(
    df_segmented: pd.DataFrame,
    x_feat: str = 'Frequency',
    y_feat: str = 'NetSpending',
    figsize: Tuple[int, int] = (13, 6),
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """
    Plot scatter visualization of customer population highlighting DBSCAN anomalies (noise = -1).
    Distinguishes high-value wholesale outliers from regular accounts.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe containing 'Is_Anomaly', 'NetSpending', 'Frequency', etc.
    x_feat : str, default='Frequency'
        Horizontal axis metric.
    y_feat : str, default='NetSpending'
        Vertical axis metric.
    figsize : tuple of (int, int), default=(13, 6)

    Returns
    -------
    fig, axes
    """
    if 'Is_Anomaly' not in df_segmented.columns:
        raise ValueError("df_segmented must contain an 'Is_Anomaly' column.")

    norm = df_segmented[df_segmented['Is_Anomaly'] == 0]
    anom = df_segmented[df_segmented['Is_Anomaly'] == 1]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Linear Scale View
    ax1 = axes[0]
    ax1.scatter(
        norm[x_feat],
        norm[y_feat],
        c='steelblue',
        alpha=0.35,
        s=25,
        label=f"Core Customers (n={len(norm):,})",
    )
    ax1.scatter(
        anom[x_feat],
        anom[y_feat],
        c='crimson',
        marker='x',
        alpha=0.85,
        s=60,
        linewidths=1.8,
        label=f"DBSCAN Anomaly (n={len(anom):,})",
    )
    ax1.set_title(f"{y_feat} vs. {x_feat} (Linear Scale)", fontsize=12, fontweight='bold')
    ax1.set_xlabel(x_feat, fontsize=10.5)
    ax1.set_ylabel(y_feat, fontsize=10.5)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best', fontsize=9)

    # 2. Log-Log Scale View (Zoom into distribution structure)
    ax2 = axes[1]
    ax2.scatter(
        np.log1p(norm[x_feat].clip(lower=0)),
        np.log1p(norm[y_feat].clip(lower=0)),
        c='steelblue',
        alpha=0.35,
        s=25,
        label="Core Customers",
    )
    ax2.scatter(
        np.log1p(anom[x_feat].clip(lower=0)),
        np.log1p(anom[y_feat].clip(lower=0)),
        c='crimson',
        marker='x',
        alpha=0.85,
        s=60,
        linewidths=1.8,
        label="DBSCAN Anomaly",
    )
    ax2.set_title(f"Log1p({y_feat}) vs. Log1p({x_feat}) (Structure View)", fontsize=12, fontweight='bold')
    ax2.set_xlabel(f"Log1p({x_feat})", fontsize=10.5)
    ax2.set_ylabel(f"Log1p({y_feat})", fontsize=10.5)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best', fontsize=9)

    plt.suptitle(
        f"DBSCAN Anomaly Landscape: Core Cluster vs. Outlier Subpopulations ({len(anom)} Anomalies)",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    return fig, axes


def plot_relative_profile_bars(
    df_segmented: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """
    Plot horizontal or grouped bar charts showing each cluster's median ratio
    relative to the total population median (Cluster Median / Population Median).
    Identifies index-above / index-below behavior for executive reporting.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe.
    features : sequence of str, optional
        Features to compare.
    figsize : tuple of (int, int), default=(14, 6)

    Returns
    -------
    fig, ax
    """
    default_features = [
        'Recency',
        'Frequency',
        'NetSpending',
        'AvgOrderValue',
        'NumDistinctProducts',
        'CustomerTenureDays',
        'PurchaseVelocity_Monthly',
    ]
    feats = list(features) if features is not None else [f for f in default_features if f in df_segmented.columns]

    pop_medians = df_segmented[feats].median()
    # Avoid zero division
    pop_medians_safe = pop_medians.replace(0, 1.0)

    unique_clusters = sorted(df_segmented['Cluster'].unique())
    n_clusters = len(unique_clusters)
    palette = _get_cluster_colors(n_clusters)

    ratios_df = pd.DataFrame(index=feats)
    for cl in unique_clusters:
        cl_medians = df_segmented.loc[df_segmented['Cluster'] == cl, feats].median()
        ratios_df[f"Cluster {cl}"] = cl_medians / pop_medians_safe

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(feats))
    width = 0.8 / n_clusters

    for i, cl in enumerate(unique_clusters):
        offset = (i - n_clusters / 2 + 0.5) * width
        col_name = f"Cluster {cl}"
        name = df_segmented.loc[df_segmented['Cluster'] == cl, 'Cluster_Name'].iloc[0] if 'Cluster_Name' in df_segmented.columns else col_name
        ax.bar(
            x + offset,
            ratios_df[col_name],
            width=width,
            label=f"{col_name} ({name})",
            color=palette[i],
            alpha=0.85,
        )

    # Reference line at 1.0 (Population Median)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Population Median (= 1.0x)')
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=25, ha='right', fontsize=10.5)
    ax.set_ylabel('Ratio to Population Median (x-fold)', fontsize=11)
    ax.set_title(
        'Segment Behavior Profile: Ratio of Cluster Median to Overall Population Median',
        fontsize=13,
        fontweight='bold',
        pad=15,
    )
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend(loc='upper right', fontsize=9.5)

    plt.tight_layout()
    return fig, ax

