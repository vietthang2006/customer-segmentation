from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


def run_kmeans(
    X: Union[np.ndarray, pd.DataFrame],
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: Union[int, str] = 'auto',
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-Means clustering algorithm.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    n_clusters : int, default=4
        The number of clusters to form.
    random_state : int, default=42
        Determines random number generation for centroid initialization.
    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm is run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    Returns
    -------
    model : KMeans
        Fitted KMeans estimator.
    labels : np.ndarray
        Cluster labels for each point.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init, #type: ignore
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(
    X: Union[np.ndarray, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = 'full',
    random_state: int = 42,
    max_iter: int = 100,
) -> Tuple[GaussianMixture, np.ndarray]:
    """
    Fit Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    n_components : int, default=4
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
    random_state : int, default=42
        Controls the random seed given to the method at each run.
    max_iter : int, default=100
        The number of EM iterations to perform.

    Returns
    -------
    model : GaussianMixture
        Fitted GaussianMixture estimator.
    labels : np.ndarray
        Cluster labels for each point.
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type, # type: ignore
        random_state=random_state,
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    return model, labels


def run_hierarchical(
    X: Union[np.ndarray, pd.DataFrame],
    n_clusters: int = 4,
    linkage: str = 'ward',
    metric: str = 'euclidean',
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fit Agglomerative Hierarchical Clustering.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    n_clusters : int, default=4
        The number of clusters to find.
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. Note: 'ward' only works with 'euclidean'.
    metric : str, default='euclidean'
        Metric used to compute the linkage.

    Returns
    -------
    model : AgglomerativeClustering
        Fitted AgglomerativeClustering estimator.
    labels : np.ndarray
        Cluster labels for each point.
    """
    # If linkage is ward, metric must be euclidean
    if linkage == 'ward':
        metric = 'euclidean'

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage, # type: ignore
        metric=metric,
    )
    labels = model.fit_predict(X)
    return model, labels


def run_dbscan(
    X: Union[np.ndarray, pd.DataFrame],
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean',
) -> Tuple[DBSCAN, np.ndarray]:
    """
    Fit Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Scaled feature matrix.
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered as in
        the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point.
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.

    Returns
    -------
    model : DBSCAN
        Fitted DBSCAN estimator.
    labels : np.ndarray
        Cluster labels for each point. Noisy samples are given the label -1.
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )
    labels = model.fit_predict(X)
    return model, labels


def get_cluster_distribution(labels: np.ndarray) -> pd.DataFrame:
    """
    Calculate cluster counts, proportions, and presence of noise (-1).

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels.

    Returns
    -------
    distribution_df : pd.DataFrame
        Summary table with Cluster label, Count, and Percentage.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    df_dist = pd.DataFrame({
        'Cluster': unique,
        'Count': counts,
        'Percentage': (counts / total) * 100,
    })
    df_dist['IsNoise'] = df_dist['Cluster'] == -1
    return df_dist.sort_values(by='Count', ascending=False).reset_index(drop=True)


def fit_all_clustering_models(
    X: Union[np.ndarray, pd.DataFrame],
    k_kmeans: int = 4,
    k_gmm: int = 4,
    k_hierarchical: int = 4,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 10,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to fit all 4 clustering models with chosen parameters.

    Returns
    -------
    results : dict
        Dictionary of {model_name: {'model': estimator, 'labels': labels, 'distribution': df}}
    """
    results = {}

    # 1. K-Means
    km_model, km_labels = run_kmeans(X, n_clusters=k_kmeans, random_state=random_state)
    results['KMeans'] = {
        'model': km_model,
        'labels': km_labels,
        'distribution': get_cluster_distribution(km_labels),
    }

    # 2. GMM
    gmm_model, gmm_labels = run_gmm(X, n_components=k_gmm, random_state=random_state)
    results['GMM'] = {
        'model': gmm_model,
        'labels': gmm_labels,
        'distribution': get_cluster_distribution(gmm_labels),
    }

    # 3. Hierarchical
    h_model, h_labels = run_hierarchical(X, n_clusters=k_hierarchical, linkage='ward')
    results['Hierarchical'] = {
        'model': h_model,
        'labels': h_labels,
        'distribution': get_cluster_distribution(h_labels),
    }

    # 4. DBSCAN
    db_model, db_labels = run_dbscan(X, eps=dbscan_eps, min_samples=dbscan_min_samples)
    results['DBSCAN'] = {
        'model': db_model,
        'labels': db_labels,
        'distribution': get_cluster_distribution(db_labels),
    }

    return results

