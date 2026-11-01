"""
Clustering package for Customer Segmentation.
Provides implementations, evaluation routines, and visualizations for:
- K-Means
- Gaussian Mixture Models (GMM)
- Agglomerative Hierarchical Clustering
- DBSCAN
"""

from .algorithms import (
    fit_all_clustering_models,
    get_cluster_distribution,
    run_dbscan,
    run_gmm,
    run_hierarchical,
    run_kmeans,
)
from .evaluation import (
    compare_clustering_algorithms,
    evaluate_clustering_metrics,
    evaluate_dbscan_grid,
    evaluate_gmm_range,
    evaluate_hierarchical_range,
    evaluate_kmeans_range,
    find_k_distance_elbow,
)
from .visualization import (
    plot_cluster_projections,
    plot_dbscan_grid_heatmap,
    plot_gmm_aic_bic,
    plot_hierarchical_dendrogram,
    plot_k_distance_graph,
    plot_kmeans_metrics,
    plot_radar_comparison,
    plot_silhouette_analysis,
    plot_snake_comparison,
)

__all__ = [
    # Algorithms
    'run_kmeans',
    'run_gmm',
    'run_hierarchical',
    'run_dbscan',
    'get_cluster_distribution',
    'fit_all_clustering_models',
    # Evaluation
    'evaluate_clustering_metrics',
    'evaluate_kmeans_range',
    'evaluate_gmm_range',
    'evaluate_hierarchical_range',
    'find_k_distance_elbow',
    'evaluate_dbscan_grid',
    'compare_clustering_algorithms',
    # Visualization
    'plot_kmeans_metrics',
    'plot_gmm_aic_bic',
    'plot_hierarchical_dendrogram',
    'plot_k_distance_graph',
    'plot_dbscan_grid_heatmap',
    'plot_silhouette_analysis',
    'plot_cluster_projections',
    'plot_snake_comparison',
    'plot_radar_comparison',
]

