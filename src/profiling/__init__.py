"""
Profiling package for Customer Segmentation.
Provides mapping, statistical aggregations, and business visualizations for:
- Mapping K-Means (K=4) clusters and DBSCAN anomalies to CustomerID
- Revenue contribution and population share analysis
- Raw feature distributions and anomaly audits
- Executive persona reporting
"""

from .mapping import (
    DEFAULT_CLUSTER_NAME_MAP,
    RAW_METRIC_COLUMNS,
    create_segmented_dataset,
    save_segmented_dataset,
)
from .statistics import (
    CORE_PROFILING_FEATURES,
    analyze_dbscan_anomalies,
    compute_cluster_profiling_table,
    compute_segment_revenue_share,
    generate_persona_summary_card,
)
from .visualization import (
    plot_anomaly_scatter,
    plot_raw_feature_boxplots_by_cluster,
    plot_relative_profile_bars,
    plot_revenue_vs_population_donut,
)

__all__ = [
    # Mapping
    'DEFAULT_CLUSTER_NAME_MAP',
    'RAW_METRIC_COLUMNS',
    'create_segmented_dataset',
    'save_segmented_dataset',
    # Statistics
    'CORE_PROFILING_FEATURES',
    'compute_cluster_profiling_table',
    'compute_segment_revenue_share',
    'analyze_dbscan_anomalies',
    'generate_persona_summary_card',
    # Visualization
    'plot_revenue_vs_population_donut',
    'plot_raw_feature_boxplots_by_cluster',
    'plot_anomaly_scatter',
    'plot_relative_profile_bars',
]

