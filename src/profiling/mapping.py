"""
Mapping module for Customer Profiling.
Combines raw customer behavior metrics with K-Means cluster assignments
and DBSCAN anomaly indicators into a production-ready segmented dataset.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

DEFAULT_CLUSTER_NAME_MAP: Dict[int, str] = {
    0: 'Churned / Low-Value',
    1: 'Core Loyal / At-Risk Frequency',
    2: 'Newly Acquired / Promising Trials',
    3: 'VIP / Champions',
}

RAW_METRIC_COLUMNS = [
    'CustomerID',
    'Recency',
    'Frequency',
    'NetSpending',
    'AvgOrderValue',
    'NumDistinctProducts',
    'CustomerTenureDays',
    'PurchaseVelocity_Monthly',
    'ReturnRate',
    'GrossSpending',
    'ReturnedAmount',
    'CancelFrequency',
    'IsUK',
]


def create_segmented_dataset(
    df_raw: pd.DataFrame,
    km_labels: Union[np.ndarray, list, pd.Series],
    db_labels: Optional[Union[np.ndarray, list, pd.Series]] = None,
    df_scaled: Optional[pd.DataFrame] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Merge K-Means cluster assignments and DBSCAN anomaly flags into the raw
    profiling dataset based on CustomerID.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Unscaled behavioral dataset (from customer_raw_profiling.parquet).
    km_labels : np.ndarray or list or pd.Series
        K-Means cluster assignments (0, 1, 2, 3) aligned with df_raw rows.
    db_labels : np.ndarray or list or pd.Series, optional
        DBSCAN cluster assignments where -1 indicates noise/outlier.
    df_scaled : pd.DataFrame, optional
        Scaled feature dataset (used for consistency checks if provided).
    cluster_name_map : dict, optional
        Mapping from cluster integer to business persona name.

    Returns
    -------
    pd.DataFrame
        Segmented dataframe containing raw metrics, 'Cluster', 'Cluster_Name',
        and 'Is_Anomaly'.
    """
    df = df_raw.copy()
    if 'CustomerID' not in df.columns:
        raise ValueError("df_raw must contain a 'CustomerID' column.")

    if len(df) != len(km_labels):
        raise ValueError(
            f"Row count mismatch: df_raw has {len(df)} rows, but km_labels has {len(km_labels)}."
        )

    # Optional cross-check with scaled dataset
    if df_scaled is not None and 'CustomerID' in df_scaled.columns:
        if not (df['CustomerID'].astype(str).values == df_scaled['CustomerID'].astype(str).values).all():
            raise ValueError("CustomerID ordering mismatch between df_raw and df_scaled.")

    # 1. Assign K-Means cluster
    df['Cluster'] = np.asarray(km_labels, dtype=int)

    # 2. Assign Persona Name
    name_map = cluster_name_map or DEFAULT_CLUSTER_NAME_MAP
    df['Cluster_Name'] = df['Cluster'].map(name_map).fillna(df['Cluster'].apply(lambda c: f'Cluster {c}'))

    # 3. Assign DBSCAN anomaly flag
    if db_labels is not None:
        if len(df) != len(db_labels):
            raise ValueError(
                f"Row count mismatch: df_raw has {len(df)} rows, but db_labels has {len(db_labels)}."
            )
        db_arr = np.asarray(db_labels, dtype=int)
        df['Is_Anomaly'] = (db_arr == -1).astype(int)
    else:
        df['Is_Anomaly'] = 0

    return df


def save_segmented_dataset(
    df_segmented: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the segmented customer dataframe to a Parquet file.

    Parameters
    ----------
    df_segmented : pd.DataFrame
        The segmented customer dataframe.
    output_path : str or Path, optional
        Target output file path. Defaults to data/clustered/customer_segmented.parquet
        under the project root.

    Returns
    -------
    Path
        Resolved path of the saved Parquet file.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parents[2]
        resolved = project_root / 'data' / 'clustered' / 'customer_segmented.parquet'
    else:
        resolved = Path(output_path).resolve()

    resolved.parent.mkdir(parents=True, exist_ok=True)
    df_segmented.to_parquet(resolved, index=False)
    return resolved
