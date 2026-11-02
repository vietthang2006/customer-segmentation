"""
Statistical computation module for Customer Profiling.
Calculates cluster distribution, revenue contribution, descriptive statistics,
anomaly audits, and executive persona summary cards.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

CORE_PROFILING_FEATURES = [
    'Recency',
    'Frequency',
    'NetSpending',
    'AvgOrderValue',
    'NumDistinctProducts',
    'CustomerTenureDays',
    'PurchaseVelocity_Monthly',
    'ReturnRate',
]


def compute_cluster_profiling_table(
    df_segmented: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    aggs: Sequence[str] = ('median', 'mean'),
) -> pd.DataFrame:
    """
    Compute cluster-level summary statistics across unscaled business metrics.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe containing 'Cluster' and numerical behavior features.
    features : list of str, optional
        Features to summarize. Defaults to CORE_PROFILING_FEATURES.
    aggs : tuple or list of str, default=('median', 'mean')
        Aggregation functions ('median', 'mean', 'std', 'min', 'max').

    Returns
    -------
    pd.DataFrame
        Hierarchical or flattened summary table indexed by Cluster.
    """
    cols = list(features) if features is not None else [c for c in CORE_PROFILING_FEATURES if c in df_segmented.columns]
    if 'Cluster' not in df_segmented.columns:
        raise ValueError("df_segmented must contain a 'Cluster' column.")

    summary = df_segmented.groupby('Cluster')[cols].agg(list(aggs))
    return summary


def compute_segment_revenue_share(df_segmented: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate customer counts, population share (%), net spending sum,
    and revenue share (%) for each cluster.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe with 'Cluster', 'NetSpending', and optional 'Cluster_Name', 'IsUK'.

    Returns
    -------
    pd.DataFrame
        Summary table detailing customer population and revenue contribution.
    """
    total_customers = len(df_segmented)
    total_revenue = float(df_segmented['NetSpending'].sum())

    records = []
    for cl, grp in df_segmented.groupby('Cluster'):
        cnt = len(grp)
        rev = float(grp['NetSpending'].sum())
        non_uk_pct = float((grp['IsUK'] == 0).mean() * 100) if 'IsUK' in grp.columns else np.nan
        name = grp['Cluster_Name'].iloc[0] if 'Cluster_Name' in grp.columns else f'Cluster {cl}'

        records.append({
            'Cluster': cl,
            'Cluster_Name': name,
            'Customers_Count': cnt,
            'Customer_Share_Pct': round(cnt / total_customers * 100, 2),
            'Total_NetRevenue': round(rev, 2),
            'Revenue_Share_Pct': round(rev / total_revenue * 100, 2) if total_revenue > 0 else 0.0,
            'Median_NetSpending': round(float(grp['NetSpending'].median()), 2),
            'Median_AOV': round(float(grp['AvgOrderValue'].median()), 2),
            'Non_UK_Share_Pct': round(non_uk_pct, 2) if not np.isnan(non_uk_pct) else 0.0,
        })

    return pd.DataFrame(records).sort_values('Cluster').reset_index(drop=True)


def analyze_dbscan_anomalies(df_segmented: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze the subpopulation flagged as DBSCAN noise (Is_Anomaly == 1).
    Distinguishes high-value wholesale whales from chronic cancelers/returners.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        Segmented dataframe containing 'Is_Anomaly', 'NetSpending', 'ReturnRate', etc.

    Returns
    -------
    dict of {str: pd.DataFrame}
        'distribution': Breakdown of anomalies across K-Means clusters.
        'comparison': Normal (0) vs Anomaly (1) median/mean comparison.
        'top_whales': Top 10 anomaly accounts by net spending.
        'top_returners': Top 10 anomaly accounts by return rate.
    """
    if 'Is_Anomaly' not in df_segmented.columns:
        raise ValueError("df_segmented must contain an 'Is_Anomaly' column.")

    anom = df_segmented[df_segmented['Is_Anomaly'] == 1]
    norm = df_segmented[df_segmented['Is_Anomaly'] == 0]

    # Cluster distribution
    dist = (
        anom.groupby('Cluster')['CustomerID']
        .count()
        .reset_index(name='Anomaly_Count')
    )
    dist['Cluster_Share_Pct'] = (dist['Anomaly_Count'] / len(anom) * 100).round(2)
    if 'Cluster_Name' in anom.columns:
        name_map = anom.groupby('Cluster')['Cluster_Name'].first().to_dict()
        dist['Cluster_Name'] = dist['Cluster'].map(name_map)

    # Key comparison
    metrics = ['NetSpending', 'Frequency', 'AvgOrderValue', 'ReturnRate', 'NumDistinctProducts']
    comp = pd.DataFrame({
        'Normal_Median': norm[metrics].median().round(2),
        'Anomaly_Median': anom[metrics].median().round(2),
        'Normal_Mean': norm[metrics].mean().round(2),
        'Anomaly_Mean': anom[metrics].mean().round(2),
    })

    # Sub-typologies
    top_whales = anom.sort_values('NetSpending', ascending=False)[
        ['CustomerID', 'Cluster', 'NetSpending', 'Frequency', 'AvgOrderValue', 'ReturnRate']
    ].head(10)

    top_returners = anom.sort_values('ReturnRate', ascending=False)[
        ['CustomerID', 'Cluster', 'NetSpending', 'Frequency', 'AvgOrderValue', 'ReturnRate', 'ReturnedAmount']
    ].head(10)

    return {
        'distribution': dist,
        'comparison': comp,
        'top_whales': top_whales,
        'top_returners': top_returners,
    }


def generate_persona_summary_card(df_segmented: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an executive persona summary table describing each cluster's
    scale, commercial footprint, core profile, primary risk, and strategic action.
    """
    rev_df = compute_segment_revenue_share(df_segmented)

    descriptions = {
        0: {
            'Archetype': 'Churned / Low-Value Hibernating',
            'Core_Trait': 'Infrequent one-time small buyers with high inactivity (Recency ~200d).',
            'Risk_Factor': 'Low engagement, dead ad spend risk if retargeted heavily.',
            'Action': 'Low-cost automated clearance emails (Q4 promos); exclude from paid ads.',
        },
        1: {
            'Archetype': 'Core Loyal / At-Risk Frequency',
            'Core_Trait': 'VIP-level basket size (~345 GBP) with slowing order velocity (~0.4/mo).',
            'Risk_Factor': 'Cadence deceleration, transitioning into lapsed status.',
            'Action': 'Automated 45-60d reorder triggers, catalog cross-selling, loyalty milestones.',
        },
        2: {
            'Archetype': 'Newly Acquired / Promising Trials',
            'Core_Trait': 'Recent onboardings (Tenure ~43d) with brisk initial ordering velocity (~1/mo).',
            'Risk_Factor': 'High one-time drop-off after first trial order.',
            'Action': 'Structured 30-60-90d welcome journey, time-limited 2nd purchase coupon.',
        },
        3: {
            'Archetype': 'VIP / Champions',
            'Core_Trait': 'High-frequency wholesale accounts (10+ orders, ~3.5k GBP spend, 120+ SKUs).',
            'Risk_Factor': 'Competitor poaching, supplier delivery or stockout friction.',
            'Action': 'Dedicated account manager, tiered volume rebates, priority inventory allocation.',
        },
    }

    cards = []
    for _, row in rev_df.iterrows():
        cl = int(row['Cluster'])
        desc = descriptions.get(cl, {
            'Archetype': row.get('Cluster_Name', f'Cluster {cl}'),
            'Core_Trait': 'N/A',
            'Risk_Factor': 'N/A',
            'Action': 'N/A',
        })
        cards.append({
            'Cluster': cl,
            'Persona': desc['Archetype'],
            'Customer_Share': f"{row['Customer_Share_Pct']}% (n={row['Customers_Count']})",
            'Revenue_Share': f"{row['Revenue_Share_Pct']}% ({row['Total_NetRevenue']:,.0f} GBP)",
            'Core_Trait': desc['Core_Trait'],
            'Primary_Risk': desc['Risk_Factor'],
            'Strategic_Action': desc['Action'],
        })

    return pd.DataFrame(cards)

