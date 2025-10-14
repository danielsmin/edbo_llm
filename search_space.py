import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class SearchSpace:
    features: List[str]
    feature_values: Dict[str, List]
    labels: List[str]
    df: pd.DataFrame

    @staticmethod
    def from_csv(csv_path: str, label_cols: Optional[List[str]] = None) -> "SearchSpace":
        assert os.path.exists(csv_path), f"Scope CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        df.attrs["source"] = os.path.abspath(csv_path)
        label_cols = [c.strip() for c in (label_cols or []) if c.strip() in df.columns]
        features = [c for c in df.columns if c not in label_cols]
        feature_values = {c: sorted(pd.unique(df[c]).tolist(), key=lambda x: str(x)) for c in features}
        return SearchSpace(features=features, feature_values=feature_values, labels=label_cols, df=df)


def summarize_search_space(space: SearchSpace, max_vals: int = 12) -> str:
    lines = []
    lines.append(f"Features ({len(space.features)}): {', '.join(space.features)}")
    if space.labels:
        lines.append(f"Experimental labels ({len(space.labels)}): {', '.join(space.labels)}")
    lines.append(f"Search space size: {len(space.df):,} rows")
    lines.append("Feature domains:")
    for f in space.features:
        vals = space.feature_values.get(f, [])
        shown = vals[:max_vals]
        more = f" … (+{len(vals)-max_vals} more)" if len(vals) > max_vals else ""
        lines.append(f"- {f}: {shown}{more}")
    return "\n".join(lines)


def preview_df(df: pd.DataFrame, n: int = 5) -> str:
    return df.head(n).to_string(index=False)
