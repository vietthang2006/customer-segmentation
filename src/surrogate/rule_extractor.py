"""
Module: src/surrogate/rule_extractor.py
Mục đích: Rút trích tập luật quyết định trực tiếp từ DecisionTreeClassifier và xuất SQL.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeRuleExtractor:
    """
    Rút trích các quy tắc nghiệp vụ dạng IF-THEN rõ ràng trực tiếp từ cấu trúc cây.
    """

    def __init__(
        self,
        tree_model: DecisionTreeClassifier,
        feature_names: List[str],
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.class_names = class_names or {}
        self.rules_: List[Dict[str, Any]] = []
        self._extract_all_paths()

    def _extract_all_paths(self) -> None:
        """Duyệt đệ quy cây từ node gốc đến các node lá để gom điều kiện."""
        tree = self.tree_model.tree_
        feature_names = self.feature_names
        children_left = tree.children_left # type: ignore
        children_right = tree.children_right # type: ignore
        feature = tree.feature # type: ignore
        threshold = tree.threshold # type: ignore
        value = tree.value

        extracted_rules = []

        def recurse(node: int, current_path: List[Dict[str, Any]]) -> None:
            if children_left[node] != children_right[node]:
                # Node rẽ nhánh
                feat_name = feature_names[feature[node]]
                thresh = float(threshold[node])

                # Nhánh trái (<=)
                recurse(children_left[node], current_path + [{"feature": feat_name, "op": "<=", "thresh": thresh}])
                # Nhánh phải (>)
                recurse(children_right[node], current_path + [{"feature": feat_name, "op": ">", "thresh": thresh}])
            else:
                # Node lá
                leaf_samples = int(tree.n_node_samples[node]) # type: ignore
                raw_dist = value[node][0]
                pred_class_idx = int(np.argmax(raw_dist))

                # Tương thích với cả scikit-learn dạng xác suất hay số lượng mẫu thô
                if raw_dist.sum() <= 1.05:
                    confidence = float(raw_dist[pred_class_idx])
                    class_counts = (raw_dist * leaf_samples).round().astype(int)
                else:
                    confidence = float(raw_dist[pred_class_idx] / raw_dist.sum())
                    class_counts = raw_dist.astype(int)

                extracted_rules.append({
                    "conditions": current_path,
                    "predicted_class": pred_class_idx,
                    "persona": self.class_names.get(pred_class_idx, f"Cluster {pred_class_idx}"),
                    "leaf_samples": leaf_samples,
                    "confidence": confidence,
                    "class_distribution": {
                        self.class_names.get(c, str(c)): int(class_counts[c]) for c in range(len(class_counts))
                    },
                })

        recurse(0, [])
        # Sắp xếp luật theo phân khúc và độ tin cậy giảm dần
        self.rules_ = sorted(extracted_rules, key=lambda r: (r["predicted_class"], -r["confidence"], -r["leaf_samples"]))

    def get_rules_summary_dataframe(self) -> pd.DataFrame:
        """Bảng tổng hợp các luật quyết định cùng độ tin cậy và số lượng tài khoản áp dụng."""
        rows = []
        for idx, rule in enumerate(self.rules_, start=1):
            cond_str = " AND ".join([f"{c['feature']} {c['op']} {c['thresh']:.2f}" for c in rule["conditions"]])
            rows.append({
                "Rule_ID": f"Rule_{idx:02d}",
                "Cluster": rule["predicted_class"],
                "Persona": rule["persona"],
                "Confidence_Pct": round(rule["confidence"] * 100.0, 2),
                "Sample_Support": rule["leaf_samples"],
                "Conditions_Count": len(rule["conditions"]),
                "Logic_Conditions": cond_str if cond_str else "TRUE (Default Leaf)",
            })
        return pd.DataFrame(rows)

    def to_sql_case_when(self, table_name: str = "customer_raw_features") -> str:
        """Sinh câu truy vấn SQL CASE WHEN sẵn sàng thực thi trên BigQuery / Snowflake / Postgres."""
        lines = [
            f"-- ====================================================================",
            f"-- Auto-generated Decision Tree Rules from Customer Segmentation",
            f"-- Source Model: DecisionTreeClassifier (Depth={self.tree_model.max_depth})", # type: ignore
            f"-- ====================================================================",
            f"SELECT",
            f"    CustomerID,",
            f"    CASE",
        ]

        sorted_rules = sorted(self.rules_, key=lambda r: (-r["confidence"], -r["leaf_samples"]))

        for rule in sorted_rules:
            if not rule["conditions"]:
                continue
            cond_sql = " AND ".join([f"{c['feature']} {c['op']} {c['thresh']:.2f}" for c in rule["conditions"]])
            lines.append(
                f"        -- Confidence: {rule['confidence']*100:.1f}%, Support: {rule['leaf_samples']} accounts\n"
                f"        WHEN {cond_sql} THEN '{rule['persona']}'"
            )

        lines.append(f"        ELSE 'Unassigned / Review Required'")
        lines.append(f"    END AS Segment_Persona")
        lines.append(f"FROM `{table_name}`;")
        return "\n".join(lines)

    def evaluate_rule_fidelity(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Đo lường độ chính xác của tập luật so với nhãn cụm K-Means gốc."""
        preds = self.tree_model.predict(X)
        fidelity = float(np.mean(preds == y))
        avg_path_len = float(np.mean([len(r["conditions"]) for r in self.rules_]))
        return {
            "rule_fidelity": fidelity,
            "total_rules": len(self.rules_),
            "avg_conditions_per_rule": avg_path_len,
        }