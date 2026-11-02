from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import shap


def plot_surrogate_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Vẽ ma trận nhầm lẫn thể hiện độ ăn khớp giữa K-Means và Decision Tree."""
    fig, ax = plt.subplots(figsize=figsize)
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
        )

    ax.set_title("Decision Tree Surrogate Fidelity: Confusion Matrix (K-Means vs. Tree)", fontsize=13, pad=15)
    ax.set_xlabel("Predicted Persona (Decision Tree)", fontsize=11, labelpad=10)
    ax.set_ylabel("True Cluster Assignment (K-Means)", fontsize=11, labelpad=10)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_depth_tuning_curve(
    depth_df: pd.DataFrame,
    figsize: Tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Biểu đồ so sánh Độ sâu cây (max_depth) vs Độ trung thực (Fidelity %) vs Độ phức tạp (Số node lá)."""
    fig, ax1 = plt.subplots(figsize=figsize)

    color_acc = "#1f77b4"
    ax1.set_xlabel("Tree Maximum Depth (max_depth)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Fidelity Accuracy (%)", color=color_acc, fontsize=11)
    line1 = ax1.plot(
        depth_df["Max_Depth"],
        depth_df["Fidelity_Accuracy_Mean_Pct"],
        marker="o",
        linewidth=2,
        color=color_acc,
        label="Fidelity Accuracy (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color_leaf = "#ff7f0e"
    ax2.set_ylabel("Total Leaf Nodes (Rule Count)", color=color_leaf, fontsize=11)
    line2 = ax2.plot(
        depth_df["Max_Depth"],
        depth_df["Avg_Leaf_Nodes"],
        marker="s",
        linestyle="--",
        linewidth=2,
        color=color_leaf,
        label="Leaf Nodes (Rules)",
    )
    ax2.tick_params(axis="y", labelcolor=color_leaf)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)
    plt.title("Surrogate Optimization: Tree Depth vs. Fidelity vs. Rule Complexity", fontsize=13, pad=15)
    plt.tight_layout()
    return fig


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Vẽ biểu đồ cột ngang độ quan trọng của đặc trưng theo tỷ lệ giảm Impurity."""
    df_plot = importance_df.head(top_n).sort_values(by="Importance_Share_Pct", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4" if i < len(df_plot) - 2 else "#ff7f0e" for i in range(len(df_plot))]
    bars = ax.barh(df_plot["Feature"], df_plot["Importance_Share_Pct"], color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=10)

    ax.set_xlim(0, max(df_plot["Importance_Share_Pct"]) * 1.15)
    ax.set_title("Decision Tree Surrogate Feature Importance (Impurity Reduction Share)", fontsize=13, pad=12)
    ax.set_xlabel("Contribution to Impurity Split (%)", fontsize=11, labelpad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_decision_tree_diagram(
    tree_model: Any,
    feature_names: List[str],
    class_names: List[str],
    max_depth: int = 3,
    figsize: Tuple[int, int] = (20, 10),
) -> plt.Figure:
    """Vẽ cấu trúc phân nhánh thực tế của cây quyết định."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        proportion=True,
        precision=1,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(f"Decision Tree Surrogate Architecture (Display Depth = {max_depth})", fontsize=15, pad=15)
    plt.tight_layout()
    return fig


def plot_shap_summary_multiclass(
    shap_values: Any,
    X: pd.DataFrame,
    class_names: List[str],
    max_display: int = 8,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Biểu đồ SHAP stacked bar tổng quan toàn cục cho 4 nhóm khách hàng.
    Chuyển đổi ma trận 3D thành danh sách các mảng 2D cho từng class và đặt plot_type='bar'.
    """

    plt.figure(figsize=figsize)

    # Chuyển đổi an toàn: SHAP multiclass yêu cầu danh sách mảng 2D cho từng class
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        num_classes = shap_values.values.shape[2]
        shap_data = [shap_values.values[:, :, i] for i in range(num_classes)]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        num_classes = shap_values.shape[2]
        shap_data = [shap_values[:, :, i] for i in range(num_classes)]
    elif isinstance(shap_values, list):
        shap_data = shap_values
    else:
        shap_data = shap_values

    # Bắt buộc khai báo plot_type="bar" cho bài toán multiclass
    shap.summary_plot(
        shap_data,
        X,
        plot_type="bar",
        class_names=class_names,
        max_display=max_display,
        show=False,
    )
    plt.title("Global SHAP Feature Importance across Customer Personas", fontsize=13, pad=15)
    plt.tight_layout()
    return plt.gcf()

def plot_shap_cohort_beeswarm(
    shap_values: Any,
    X: pd.DataFrame,
    class_idx: int,
    class_name: str,
    max_display: int = 8,
    figsize: Tuple[int, int] = (9, 6),
) -> Optional[plt.Figure]:
    """Biểu đồ SHAP Beeswarm thể hiện giá trị đặc trưng cao/thấp đẩy khách hàng vào nhóm class_idx."""
    plt.figure(figsize=figsize)
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        target_explanation = shap_values[:, :, class_idx]
        shap.plots.beeswarm(target_explanation, max_display=max_display, show=False)
    elif isinstance(shap_values, list):
        shap.summary_plot(shap_values[class_idx], X, max_display=max_display, show=False)
    else:
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)

    plt.title(f"SHAP Beeswarm: Driving Forces for '{class_name}'", fontsize=13, pad=15)
    plt.tight_layout()
    return plt.gcf()