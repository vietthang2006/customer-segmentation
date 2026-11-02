"""
Package: src/surrogate
Mô hình Decision Tree Surrogate và Explainable AI (XAI) cho phân khúc khách hàng.
"""

from .trainer import DecisionTreeSurrogateModel
from .explainability import SurrogateSHAPExplainer
from .rule_extractor import DecisionTreeRuleExtractor
from .visualization import (
    plot_surrogate_confusion_matrix,
    plot_depth_tuning_curve,
    plot_feature_importance_bar,
    plot_decision_tree_diagram,
    plot_shap_summary_multiclass,
    plot_shap_cohort_beeswarm,
)

__all__ = [
    "DecisionTreeSurrogateModel",
    "SurrogateSHAPExplainer",
    "DecisionTreeRuleExtractor",
    "plot_surrogate_confusion_matrix",
    "plot_depth_tuning_curve",
    "plot_feature_importance_bar",
    "plot_decision_tree_diagram",
    "plot_shap_summary_multiclass",
    "plot_shap_cohort_beeswarm",
]