from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

class DecisionTreeSurrogateModel:
    def __init__(
            self, 
            max_depth: int = 5,
            min_samples_leaf: int = 20,
            criterion: str = "entropy",
            random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model: Optional[DecisionTreeClassifier] = None 
        self.feature_names_: List[str] = []
        self.class_names_: Dict[int, str] = {}
        self.cv_results_: Dict[str, Any] = {}

    def evaluate_depth_candidates(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            depth_candidates: List[int] = [3,4,5,6],
            n_splits: int = 5,
    ) -> pd.DataFrame:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        records = []

        for d in depth_candidates:
            acc_list, f1_list, leaf_counts = [], [], []

            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                tree = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion=self.criterion, # type: ignore
                    random_state=self.random_state
                )

                tree.fit(X_train, y_train)
                preds = tree.predict(X_val)

                acc_list.append(accuracy_score(y_val, preds))
                f1_list.append(f1_score(y_val, preds, average="macro"))
                leaf_counts.append(tree.get_n_leaves())

            records.append({
                "Max_Depth": d,
                "Fidelity_Accuracy_Mean_Pct": np.mean(acc_list) * 100.0,
                "Fidelity_Accuracy_Std_Pct": np.std(acc_list) * 100.0,
                "Macro_F1": np.mean(f1_list),
                "Avg_Leaf_Nodes": int(np.round(np.mean(leaf_counts)))
            })

        return pd.DataFrame(records)

    def cross_validation(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            n_splits: int = 5,
            class_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        self.feature_names_ = list(X.columns)
        self.class_names_ = class_names or {c: f"Cluster {c}" for c in np.unique(y)}
        num_classes = len(np.unique(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros((len(y), num_classes))
        X_arr = X.values
        y_arr = y.values
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
            fold_tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion, # type: ignore
                random_state=self.random_state,
            )
            fold_tree.fit(X_train, y_train)
            oof_preds[val_idx] = fold_tree.predict(X_val)
            oof_probs[val_idx] = fold_tree.predict_proba(X_val)
        overall_fidelity = accuracy_score(y_arr, oof_preds)
        macro_f1 = f1_score(y_arr, oof_preds, average="macro")
        balanced_acc = balanced_accuracy_score(y_arr, oof_preds)
        cm = confusion_matrix(y_arr, oof_preds)
        target_names = [self.class_names_.get(c, str(c)) for c in sorted(self.class_names_.keys())]
        report_dict = classification_report(y_arr, oof_preds, target_names=target_names, output_dict=True)
        self.cv_results_ = {
            "overall_fidelity": overall_fidelity,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "confusion_matrix": cm,
            "oof_preds": oof_preds,
            "oof_probs": oof_probs,
            "classification_report": report_dict,
        }
        return self.cv_results_
    
    def fit(self, X: pd.DataFrame, y: pd.Series, class_names: Optional[Dict[int, str]] = None):
        self.feature_names_ = list(X.columns)
        self.class_names_ = class_names or {c: f"Cluster {c}" for c in np.unique(y)}
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion, # type: ignore
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": importances,
            "Importance_Share_Pct": (importances / importances.sum()) * 100.0 if importances.sum() > 0 else 0.0,
        })
        return df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)