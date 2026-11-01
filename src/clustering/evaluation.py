from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def evaluate_clustering_metrics(
    X: Union[np.ndarray, pd.DataFrame],
    labels: np.ndarray,
    metric: str = 'euclidean',
) -> Dict[str, Union[int, float]]:
    """
    Compute common internal validation metrics for a set of cluster labels.
    Properly handles DBSCAN noise (-1) points.

    Returns
    -------
    metrics : dict
        Contains n_clusters, n_noise, noise_pct, silhouette, davies_bouldin, calinski_harabasz.
    """
    X_mat = np.asarray(X)
    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    noise_pct = float(n_noise / len(labels) * 100)

    # Exclude noise points when calculating cluster cohesion/separation
    non_noise_mask = labels != -1
    valid_labels = labels[non_noise_mask]
    n_clusters = len(set(valid_labels))

    results: Dict[str, Union[int, float]] = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_pct': round(noise_pct, 2),
        'silhouette': np.nan,
        'davies_bouldin': np.nan,
        'calinski_harabasz': np.nan,
    }

    if n_clusters >= 2 and len(valid_labels) > n_clusters:
        X_valid = X_mat[non_noise_mask]
        results['silhouette'] = float(silhouette_score(X_valid, valid_labels, metric=metric))
        results['davies_bouldin'] = float(davies_bouldin_score(X_valid, valid_labels))
        results['calinski_harabasz'] = float(calinski_harabasz_score(X_valid, valid_labels))

    return results


def evaluate_kmeans_range(
    X: Union[np.ndarray, pd.DataFrame],
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
    n_init: Union[int, str] = 'auto',
) -> pd.DataFrame:
    """
    Evaluate K-Means across a range of k values.
    Calculates Inertia (Elbow), Silhouette, Davies-Bouldin, and Calinski-Harabasz.

    Returns
    -------
    metrics_df : pd.DataFrame
    """
    X_mat = np.asarray(X)
    records = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init) #type: ignore
        labels = model.fit_predict(X_mat)

        sil = silhouette_score(X_mat, labels)
        db = davies_bouldin_score(X_mat, labels)
        ch = calinski_harabasz_score(X_mat, labels)

        records.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
        })

    return pd.DataFrame(records)


def evaluate_gmm_range(
    X: Union[np.ndarray, pd.DataFrame],
    k_range: Iterable[int] = range(2, 11),
    covariance_types: Tuple[str, ...] = ('full', 'diag'),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate Gaussian Mixture Models across k range and covariance types.
    Calculates AIC, BIC, and Silhouette Score.

    Returns
    -------
    metrics_df : pd.DataFrame
    """
    X_mat = np.asarray(X)
    records = []

    for cov in covariance_types:
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov, #type: ignore
                random_state=random_state,
            )
            labels = gmm.fit_predict(X_mat)
            sil = silhouette_score(X_mat, labels) if len(set(labels)) > 1 else np.nan

            records.append({
                'k': k,
                'covariance_type': cov,
                'bic': gmm.bic(X_mat),
                'aic': gmm.aic(X_mat),
                'silhouette': sil,
            })

    return pd.DataFrame(records)


def evaluate_hierarchical_range(
    X: Union[np.ndarray, pd.DataFrame],
    k_range: Iterable[int] = range(2, 11),
    linkage: str = 'ward',
    metric: str = 'euclidean',
) -> pd.DataFrame:
    """
    Evaluate Agglomerative Hierarchical Clustering across a range of k values.

    Returns
    -------
    metrics_df : pd.DataFrame
    """
    X_mat = np.asarray(X)
    if linkage == 'ward':
        metric = 'euclidean'

    records = []
    for k in k_range:
        hclust = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric) #type: ignore
        labels = hclust.fit_predict(X_mat)

        sil = silhouette_score(X_mat, labels, metric=metric)
        db = davies_bouldin_score(X_mat, labels)
        ch = calinski_harabasz_score(X_mat, labels)

        records.append({
            'k': k,
            'linkage': linkage,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
        })

    return pd.DataFrame(records)


def find_k_distance_elbow(
    X: Union[np.ndarray, pd.DataFrame],
    k: Optional[int] = None,
) -> np.ndarray:
    """
    Compute sorted k-nearest neighbor distances for DBSCAN eps tuning.
    Usually k = 2 * n_features - 1 or k = min_samples.

    Returns
    -------
    sorted_distances : np.ndarray
        Array of k-NN distances sorted in ascending order.
    """
    X_mat = np.asarray(X)
    if k is None:
        k = max(4, 2 * X_mat.shape[1] - 1)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_mat)
    distances, _ = nn.kneighbors(X_mat)

    # Distances to the k-th nearest neighbor (0-indexed: index k - 1)
    k_distances = distances[:, k - 1]
    return np.sort(k_distances)


def evaluate_dbscan_grid(
    X: Union[np.ndarray, pd.DataFrame],
    eps_values: Optional[List[float]] = None,
    min_samples_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Grid search evaluation for DBSCAN across combinations of eps and min_samples.

    Returns
    -------
    grid_df : pd.DataFrame
        Contains eps, min_samples, n_clusters, noise_pct, and silhouette.
    """
    X_mat = np.asarray(X)
    if eps_values is None:
        eps_values = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    if min_samples_values is None:
        min_samples_values = [5, 10, 15, 20]

    records = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_mat)
            metrics = evaluate_clustering_metrics(X_mat, labels)

            records.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': metrics['n_clusters'],
                'noise_pct': metrics['noise_pct'],
                'silhouette': metrics['silhouette'],
                'davies_bouldin': metrics['davies_bouldin'],
            })

    return pd.DataFrame(records)


def compare_clustering_algorithms(
    X: Union[np.ndarray, pd.DataFrame],
    models_dict: Dict[str, Union[np.ndarray, Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Generate a side-by-side comparison table across multiple clustering models.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    models_dict : dict
        Mapping of model_name -> labels array OR model_name -> {'labels': labels}.

    Returns
    -------
    comparison_df : pd.DataFrame
        Summary table ranking models by Silhouette, Davies-Bouldin, cluster balance.
    """
    X_mat = np.asarray(X)
    records = []

    for name, item in models_dict.items():
        if isinstance(item, dict) and 'labels' in item:
            labels = np.asarray(item['labels'])
        else:
            labels = np.asarray(item)

        eval_res = evaluate_clustering_metrics(X_mat, labels)

        # Cluster size balance
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 0:
            _, counts = np.unique(valid_labels, return_counts=True)
            min_size_pct = round((counts.min() / len(labels)) * 100, 2)
            max_size_pct = round((counts.max() / len(labels)) * 100, 2)
        else:
            min_size_pct, max_size_pct = 0.0, 0.0

        records.append({
            'Algorithm': name,
            'Clusters (k)': eval_res['n_clusters'],
            'Noise (%)': eval_res['noise_pct'],
            'Silhouette': round(eval_res['silhouette'], 4) if not np.isnan(eval_res['silhouette']) else None,
            'Davies-Bouldin': round(eval_res['davies_bouldin'], 4) if not np.isnan(eval_res['davies_bouldin']) else None,
            'Calinski-Harabasz': round(eval_res['calinski_harabasz'], 1) if not np.isnan(eval_res['calinski_harabasz']) else None,
            'Min Cluster Size (%)': min_size_pct,
            'Max Cluster Size (%)': max_size_pct,
        })

    df_comp = pd.DataFrame(records)
    # Sort primarily by Silhouette score descending
    if 'Silhouette' in df_comp.columns and df_comp['Silhouette'].notnull().any():
        df_comp = df_comp.sort_values(by='Silhouette', ascending=False).reset_index(drop=True)

    return df_comp