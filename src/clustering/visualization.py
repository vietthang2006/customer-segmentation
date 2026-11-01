"""
Module for clustering visualizations: Elbow curves, BIC/AIC, Dendrogram, k-distance,
Silhouette analysis diagrams, PCA 2D/3D cluster projections, Snake plots, and Radar charts.
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_kmeans_metrics(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[plt.Figure, np.ndarray]: #type:ignore
    """
    Plot Elbow curve (Inertia) and Silhouette score across k for K-Means.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output from evaluate_kmeans_range with columns: k, inertia, silhouette, davies_bouldin.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: Inertia (Elbow Method)
    ax1 = axes[0]
    ax1.plot(metrics_df['k'], metrics_df['inertia'], 'bo-', linewidth=2, markersize=7)
    ax1.set_title('K-Means: Elbow Method (Inertia / WCSS)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11)
    ax1.set_xticks(metrics_df['k'])
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Silhouette Score & Davies-Bouldin Index
    ax2 = axes[1]
    color_sil = '#2ca02c'
    color_db = '#d62728'

    ax2.plot(metrics_df['k'], metrics_df['silhouette'], 's-', color=color_sil, linewidth=2, markersize=7, label='Silhouette (Higher = Better)')
    ax2.set_title('K-Means: Silhouette Score & Davies-Bouldin', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', color=color_sil, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_sil)
    ax2.set_xticks(metrics_df['k'])
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Secondary y-axis for Davies-Bouldin Index
    ax2_db = ax2.twinx()
    ax2_db.plot(metrics_df['k'], metrics_df['davies_bouldin'], '^--', color=color_db, linewidth=2, markersize=7, label='Davies-Bouldin (Lower = Better)')
    ax2_db.set_ylabel('Davies-Bouldin Index', color=color_db, fontsize=11)
    ax2_db.tick_params(axis='y', labelcolor=color_db)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_db.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    return fig, axes


def plot_gmm_aic_bic(
    gmm_metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]: #type:ignore
    """
    Plot BIC and AIC scores across k for GMM across different covariance types.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: BIC
    ax1 = axes[0]
    sns.lineplot(data=gmm_metrics_df, x='k', y='bic', hue='covariance_type', marker='o', ax=ax1)
    ax1.set_title('GMM: Bayesian Information Criterion (BIC)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Components (k)', fontsize=11)
    ax1.set_ylabel('BIC (Lower = Better)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: AIC
    ax2 = axes[1]
    sns.lineplot(data=gmm_metrics_df, x='k', y='aic', hue='covariance_type', marker='s', ax=ax2)
    ax2.set_title('GMM: Akaike Information Criterion (AIC)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Components (k)', fontsize=11)
    ax2.set_ylabel('AIC (Lower = Better)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig, axes


def plot_hierarchical_dendrogram(
    X: Union[np.ndarray, pd.DataFrame],
    linkage_method: str = 'ward',
    metric: str = 'euclidean',
    truncate_mode: Optional[str] = 'lastp',
    p: int = 30,
    cut_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type:ignore
    """
    Compute and plot the Hierarchical Clustering Dendrogram.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    linkage_method : str, default='ward'
        Linkage algorithm.
    truncate_mode : str, default='lastp'
        Truncation mode for readability.
    p : int, default=30
        Number of clusters/leaves to show when truncated.
    cut_threshold : float, optional
        Horizontal line indicating where the dendrogram is cut into clusters.
    """
    X_mat = np.asarray(X)
    Z = linkage(X_mat, method=linkage_method, metric=metric)

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster / Leaf Node Index (Truncated)', fontsize=11)
    ax.set_ylabel('Euclidean Linkage Distance', fontsize=11)

    if cut_threshold is not None:
        ax.axhline(y=cut_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Cut Threshold = {cut_threshold}')
        ax.legend()

    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig, ax


def plot_k_distance_graph(
    sorted_distances: np.ndarray,
    knee_point: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """
    Plot the sorted k-nearest neighbor distance curve to find the optimal eps for DBSCAN.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sorted_distances, linewidth=2, color='#1f77b4')
    ax.set_title('DBSCAN: Sorted k-Distance Graph (Finding Optimal eps)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Points sorted by distance to k-th nearest neighbor', fontsize=11)
    ax.set_ylabel('k-NN Distance', fontsize=11)

    if knee_point is not None:
        ax.axhline(y=knee_point, color='r', linestyle='--', linewidth=1.5, label=f'Estimated eps = {knee_point}')
        ax.legend()

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig, ax


def plot_dbscan_grid_heatmap(
    grid_df: pd.DataFrame,
    metric: str = 'silhouette',
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """
    Plot a heatmap of DBSCAN evaluation metric (silhouette or noise_pct) across eps and min_samples.
    """
    pivot = grid_df.pivot(index='eps', columns='min_samples', values=metric)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = 'viridis' if metric == 'silhouette' else 'magma_r'
    sns.heatmap(pivot, annot=True, fmt='.3f' if metric == 'silhouette' else '.1f', cmap=cmap, ax=ax)
    title_metric = 'Silhouette Score' if metric == 'silhouette' else 'Noise Ratio (%)'
    ax.set_title(f'DBSCAN Parameter Tuning Heatmap: {title_metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('min_samples', fontsize=11)
    ax.set_ylabel('eps', fontsize=11)
    plt.tight_layout()
    return fig, ax


def plot_silhouette_analysis(
    X: Union[np.ndarray, pd.DataFrame],
    labels: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]: #type: ignore
    """
    Plot the silhouette diagram for individual clusters (showing cohesion and separation).
    """
    X_mat = np.asarray(X)
    non_noise = labels != -1
    X_valid = X_mat[non_noise]
    valid_labels = labels[non_noise]

    n_clusters = len(set(valid_labels))
    sil_avg = silhouette_score(X_valid, valid_labels)
    sample_sil_values = silhouette_samples(X_valid, valid_labels)

    fig, ax = plt.subplots(figsize=figsize)
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters)) 

    for i in range(n_clusters):
        ith_cluster_sil_values = sample_sil_values[valid_labels == i]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10, fontweight='bold')
        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Analysis Diagram (Average Score = {sil_avg:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient Values', fontsize=11)
    ax.set_ylabel('Cluster Label', fontsize=11)
    ax.axvline(x=sil_avg, color='red', linestyle='--', linewidth=1.5, label=f'Mean Silhouette = {sil_avg:.3f}')
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1.0])
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig, ax


def plot_cluster_projections(
    X: Union[np.ndarray, pd.DataFrame],
    labels_dict: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (14, 11),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot 2D PCA projections comparing cluster assignments of multiple algorithms side-by-side.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    labels_dict : dict
        Mapping of {model_name: labels_array}.
    """
    X_mat = np.asarray(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_mat)
    var1, var2 = pca.explained_variance_ratio_[:2] * 100

    n_models = len(labels_dict)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for idx, (name, labels) in enumerate(labels_dict.items()):
        ax = axes_flat[idx]
        unique_labels = np.unique(labels)
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            if lbl == -1:
                # Noise points for DBSCAN
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='gray', marker='x', alpha=0.4, s=20, label='Noise (-1)')
            else:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=palette[i % 10], alpha=0.6, s=25, label=f'Cluster {lbl}')

        ax.set_title(f'{name} Clusters (2D PCA Projection)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var1:.1f}% Variance)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var2:.1f}% Variance)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=8, markerscale=1.5)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes_flat


DEFAULT_8_FEATURE_NAMES = [
    'Recency',
    'Frequency',
    'NetSpending',
    'AvgOrderValue',
    'NumDistinctProducts',
    'CustomerTenureDays',
    'PurchaseVelocity',
    'ReturnRate',
]


def _prepare_profile_dataframe(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert input feature matrix to a standardized Z-score DataFrame with clean column names.
    """
    if isinstance(X, pd.DataFrame):
        df_num = X.drop(columns=['CustomerID'], errors='ignore').copy()
        if feature_names is None:
            clean_names = [
                str(c).replace('_scaled', '').replace('_log1p', '').replace('_Monthly', '')
                for c in df_num.columns
            ]
        else:
            clean_names = list(feature_names)
        mat = df_num.to_numpy(dtype=float)
    else:
        mat = np.asarray(X, dtype=float)
        if feature_names is not None:
            clean_names = list(feature_names)
        elif mat.shape[1] == len(DEFAULT_8_FEATURE_NAMES):
            clean_names = list(DEFAULT_8_FEATURE_NAMES)
        else:
            clean_names = [f'Feature {i}' for i in range(mat.shape[1])]

    # Standardize to Z-scores if not already zero-mean unit-variance
    means = np.nanmean(mat, axis=0)
    stds = np.nanstd(mat, axis=0)
    if np.any(np.abs(means) > 1e-2) or np.any(np.abs(stds - 1.0) > 1e-2):
        stds_safe = np.where(stds == 0, 1.0, stds)
        mat = (mat - means) / stds_safe

    return pd.DataFrame(mat, columns=clean_names), clean_names


def plot_snake_comparison(
    X: Union[np.ndarray, pd.DataFrame],
    labels_dict: Union[Dict[str, np.ndarray], np.ndarray],
    feature_names: Optional[List[str]] = None,
    agg: str = 'mean',
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """
    Plot side-by-side Snake Plots (Z-score profiles across features) to visually compare
    different cluster configurations (e.g., k=3 vs k=4).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled (Z-score) or numerical feature matrix.
    labels_dict : dict of {title: labels_array} or single labels_array
        Cluster assignments to compare (e.g., {'K-Means (k=3)': km3, 'K-Means (k=4)': km4}).
    feature_names : list of str, optional
        Display names for features on the x-axis.
    agg : {'mean', 'median'}, default='mean'
        Aggregation statistic for cluster Z-score profiles.
    figsize : tuple of (int, int), optional
        Figure dimensions.
    """
    if not isinstance(labels_dict, dict):
        labels_dict = {'Cluster Profile': np.asarray(labels_dict)}

    df_z, cols = _prepare_profile_dataframe(X, feature_names=feature_names)
    n_plots = len(labels_dict)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=figsize or (7 * n_plots, 5.5),
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    x_idx = np.arange(len(cols))

    for idx, (title, labels) in enumerate(labels_dict.items()):
        ax = axes_flat[idx]
        lbl_arr = np.asarray(labels)
        valid_mask = lbl_arr != -1
        df_sub = df_z.loc[valid_mask].copy()
        df_sub['_cluster'] = lbl_arr[valid_mask]

        grouped = (
            df_sub.groupby('_cluster')[cols].median()
            if agg == 'median'
            else df_sub.groupby('_cluster')[cols].mean()
        )

        unique_labels = grouped.index.to_numpy()
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1))) #type: ignore
        total_n = len(lbl_arr)

        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Population Avg (Z=0)')

        for i, cl in enumerate(unique_labels):
            cnt = int(np.sum(lbl_arr == cl))
            pct = cnt / total_n * 100
            y_vals = grouped.loc[cl].to_numpy(dtype=float)
            ax.plot(
                x_idx,
                y_vals,
                marker='o',
                linewidth=2.2,
                markersize=6,
                color=palette[i % 10],
                label=f'Cluster {cl} (n={cnt}, {pct:.1f}%)',
            )

        ax.set_title(f'Snake Plot: {title}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_idx)
        ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=10)
        if idx == 0:
            ax.set_ylabel(f'Standardized Z-Score ({agg.capitalize()})', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=8.5)

    plt.tight_layout()
    return fig, axes_flat


def plot_radar_comparison(
    X: Union[np.ndarray, pd.DataFrame],
    labels_dict: Union[Dict[str, np.ndarray], np.ndarray],
    feature_names: Optional[List[str]] = None,
    agg: str = 'mean',
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]: #type: ignore
    """
    Plot side-by-side Polar Radar Charts of cluster Z-score profiles to compare
    behavioral shapes across different cluster counts (e.g., k=3 vs k=4).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled (Z-score) or numerical feature matrix.
    labels_dict : dict of {title: labels_array} or single labels_array
        Cluster assignments to compare (e.g., {'K-Means (k=3)': km3, 'K-Means (k=4)': km4}).
    feature_names : list of str, optional
        Display names for radial spokes.
    agg : {'mean', 'median'}, default='mean'
        Aggregation statistic for cluster Z-score profiles.
    figsize : tuple of (int, int), optional
        Figure dimensions.
    """
    if not isinstance(labels_dict, dict):
        labels_dict = {'Cluster Profile': np.asarray(labels_dict)}

    df_z, cols = _prepare_profile_dataframe(X, feature_names=feature_names)
    n_features = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Precompute profiles to synchronize radial axis limits across subplots
    profiles_per_model = {}
    global_min, global_max = 0.0, 0.0
    for title, labels in labels_dict.items():
        lbl_arr = np.asarray(labels)
        valid_mask = lbl_arr != -1
        df_sub = df_z.loc[valid_mask].copy()
        df_sub['_cluster'] = lbl_arr[valid_mask]
        grouped = (
            df_sub.groupby('_cluster')[cols].median()
            if agg == 'median'
            else df_sub.groupby('_cluster')[cols].mean()
        )
        profiles_per_model[title] = (grouped, lbl_arr)
        global_min = min(global_min, float(grouped.min().min()))
        global_max = max(global_max, float(grouped.max().max()))

    r_min = np.floor(global_min * 2) / 2 - 0.25
    r_max = np.ceil(global_max * 2) / 2 + 0.25

    n_plots = len(labels_dict)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=figsize or (7 * n_plots, 6.5),
        subplot_kw={'polar': True},
    )
    axes_flat = np.atleast_1d(axes).flatten()

    theta_grid = np.linspace(0, 2 * np.pi, 200)

    for idx, (title, (grouped, lbl_arr)) in enumerate(profiles_per_model.items()):
        ax = axes_flat[idx]
        ax.set_theta_offset(np.pi / 2) #type: ignore
        ax.set_theta_direction(-1) #type: ignore

        unique_labels = grouped.index.to_numpy()
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1))) #type: ignore
        total_n = len(lbl_arr)

        # Reference circle at Z = 0
        ax.plot(theta_grid, np.zeros_like(theta_grid), 'k--', linewidth=1.1, alpha=0.65, label='Z = 0')

        for i, cl in enumerate(unique_labels):
            cnt = int(np.sum(lbl_arr == cl))
            pct = cnt / total_n * 100
            vals = grouped.loc[cl].to_numpy(dtype=float).tolist()
            vals_closed = vals + vals[:1]

            color = palette[i % 10]
            ax.plot(
                angles_closed,
                vals_closed,
                marker='o',
                linewidth=2,
                markersize=5,
                color=color,
                label=f'Cluster {cl} ({pct:.1f}%)',
            )
            ax.fill(angles_closed, vals_closed, color=color, alpha=0.12)

        ax.set_xticks(angles)
        ax.set_xticklabels(cols, fontsize=9.5)
        ax.set_ylim(r_min, r_max)
        ax.set_title(f'Radar Chart: {title}', fontsize=12, fontweight='bold', pad=22)
        ax.grid(True, linestyle='--', alpha=0.45)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), fontsize=8.5)

    plt.tight_layout()
    return fig, axes_flat


