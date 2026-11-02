from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import shap


class SurrogateSHAPExplainer:

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[Any] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        self.explainer = shap.TreeExplainer(self.model)

    def fit_transform(self, X: pd.DataFrame) -> Any:
        """Tính toán ma trận SHAP values (n_samples, n_features, n_classes)."""
        if self.explainer is None:
            self._init_explainer()
        self.shap_values = self.explainer(X) # type: ignore
        return self.shap_values

    def get_global_importance(self, class_idx: Optional[int] = None) -> pd.DataFrame:
        if self.shap_values is None:
            raise ValueError("Chưa tính SHAP values. Hãy gọi fit_transform(X) trước.")

        if hasattr(self.shap_values, "values"):
            vals = self.shap_values.values
            if vals.ndim == 3:
                if class_idx is not None:
                    mean_abs = np.mean(np.abs(vals[:, :, class_idx]), axis=0)
                else:
                    mean_abs = np.mean(np.mean(np.abs(vals), axis=2), axis=0)
            else:
                mean_abs = np.mean(np.abs(vals), axis=0)
        elif isinstance(self.shap_values, list):
            if class_idx is not None:
                mean_abs = np.mean(np.abs(self.shap_values[class_idx]), axis=0)
            else:
                mean_abs = np.mean([np.mean(np.abs(cv), axis=0) for cv in self.shap_values], axis=0)
        else:
            mean_abs = np.mean(np.abs(self.shap_values), axis=0)

        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs,
        })
        df_imp["Importance_Share_Pct"] = (df_imp["Mean_Abs_SHAP"] / df_imp["Mean_Abs_SHAP"].sum()) * 100.0
        return df_imp.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    def explain_individual_customer(
        self,
        X: pd.DataFrame,
        sample_idx: int,
        target_class: int,
    ) -> pd.DataFrame:
        if self.shap_values is None:
            raise ValueError("Chưa tính SHAP values. Hãy gọi fit_transform(X) trước.")

        row_features = X.iloc[sample_idx]

        if hasattr(self.shap_values, "values") and self.shap_values.values.ndim == 3:
            row_shap = self.shap_values.values[sample_idx, :, target_class]
            base_val = (
                self.shap_values.base_values[sample_idx, target_class]
                if self.shap_values.base_values.ndim == 2
                else self.shap_values.base_values[target_class]
            )
        elif isinstance(self.shap_values, list):
            row_shap = self.shap_values[target_class][sample_idx]
            base_val = getattr(self.explainer, "expected_value", [0])[target_class]
        else:
            row_shap = self.shap_values.values[sample_idx]
            base_val = getattr(self.shap_values, "base_values", [0])[sample_idx]

        df_local = pd.DataFrame({
            "Feature": self.feature_names,
            "Feature_Value": row_features.values,
            "SHAP_Contribution": row_shap,
            "Abs_Contribution": np.abs(row_shap),
            "Impact_Direction": np.where(row_shap > 0, "Ủng hộ phân khúc (+)", "Kéo lùi phân khúc (-)"),
        })
        df_local["Base_Value_Probability"] = base_val
        return df_local.sort_values(by="Abs_Contribution", ascending=False).reset_index(drop=True)