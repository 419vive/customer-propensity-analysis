from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Paths and I/O
# -----------------------------
def get_paths() -> Tuple[Path, Path, Path]:
    data_dir = Path("/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset")
    input_csv = data_dir / "training_sample.csv"
    out_dir = data_dir / "analysis_mvp"
    return input_csv, out_dir, data_dir


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


# -----------------------------
# Column classification
# -----------------------------
def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    binary_cols: List[str] = []
    numeric_non_binary: List[str] = []
    for col in numeric_cols:
        values = set(pd.unique(df[col].dropna()))
        if values.issubset({0, 1}):
            binary_cols.append(col)
        else:
            numeric_non_binary.append(col)
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    return {
        "binary": binary_cols,
        "numeric_non_binary": numeric_non_binary,
        "non_numeric": non_numeric,
    }


# -----------------------------
# Target distribution
# -----------------------------
def save_target_distribution(df: pd.DataFrame, target: str, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' is not in dataframe")
    vc = df[target].value_counts(dropna=False)
    vcp = df[target].value_counts(normalize=True, dropna=False) * 100
    out = pd.DataFrame({"count": vc.astype(int), "rate_percent": vcp.round(2)})
    out.to_csv(out_dir / "target_distribution.csv")
    return out


# -----------------------------
# Bivariate (feature vs target)
# -----------------------------
def bivariate_binary_vs_target(df: pd.DataFrame, binary_cols: List[str], target: str, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' missing")
    global_rate = float(df[target].mean())
    rows = []
    for col in binary_cols:
        if col == target:
            continue
        s = df[col]
        one_mask = s == 1
        zero_mask = s == 0
        n1 = int(one_mask.sum())
        n0 = int(zero_mask.sum())
        r1 = float(df.loc[one_mask, target].mean()) if n1 > 0 else np.nan
        r0 = float(df.loc[zero_mask, target].mean()) if n0 > 0 else np.nan
        uplift_pp = (r1 - global_rate) if pd.notna(r1) else np.nan
        rows.append(
            {
                "feature": col,
                "support_1": n1,
                "support_0": n0,
                "rate_when_1": r1,
                "rate_when_0": r0,
                "global_rate": global_rate,
                "uplift_pp": uplift_pp,
            }
        )
    out = pd.DataFrame(rows).sort_values(["uplift_pp", "support_1"], ascending=[False, False])
    out.to_csv(out_dir / "bivariate_binary_vs_target.csv", index=False)
    return out


def bivariate_numeric_vs_target_corr(
    df: pd.DataFrame, numeric_cols: List[str], target: str, out_dir: Path
) -> pd.DataFrame:
    """MVP: Pearson correlation of each numeric feature with binary target (approx point-biserial)."""
    ensure_dir(out_dir)
    cols = [c for c in numeric_cols if c != target]
    if not cols:
        return pd.DataFrame()
    corr = df[cols + [target]].corr(method="pearson")[target].drop(labels=[target]).sort_values(ascending=False)
    out = corr.reset_index().rename(columns={"index": "feature", target: "pearson_corr_with_target"})
    out.to_csv(out_dir / "bivariate_numeric_corr_with_target.csv", index=False)
    return out


# -----------------------------
# Correlation matrix & multicollinearity
# -----------------------------
def save_binary_correlation_and_flags(
    df: pd.DataFrame, binary_cols: List[str], out_dir: Path, threshold: float = 0.8
) -> pd.DataFrame:
    ensure_dir(out_dir)
    if not binary_cols:
        return pd.DataFrame()
    corr = df[binary_cols].corr().round(3)
    corr.to_csv(out_dir / "binary_correlation_matrix.csv")
    flags = []
    for i, a in enumerate(binary_cols):
        for j in range(i + 1, len(binary_cols)):
            b = binary_cols[j]
            val = float(corr.loc[a, b])
            if abs(val) >= threshold and a != b:
                flags.append({"feature_a": a, "feature_b": b, "pearson_corr": val})
    flagged = pd.DataFrame(flags).sort_values(by="pearson_corr", ascending=False)
    flagged.to_csv(out_dir / "high_correlation_pairs.csv", index=False)
    return flagged


# -----------------------------
# Behavior patterns (engagement)
# -----------------------------
def save_engagement_score(df: pd.DataFrame, binary_cols: List[str], out_dir: Path, exclude: List[str]) -> None:
    """MVP engagement score = sum of selected binary actions per user."""
    ensure_dir(out_dir)
    action_cols = [c for c in binary_cols if c not in set(exclude)]
    if not action_cols:
        return
    score = df[action_cols].sum(axis=1)
    stats = score.describe(percentiles=[0.5, 0.9, 0.99]).rename("engagement_score")
    stats.to_csv(out_dir / "engagement_score_summary.csv", header=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    score.plot(kind="hist", bins=30, color="#1971c2", alpha=0.9, ax=ax)
    ax.set_title("Engagement Score — Distribution")
    ax.set_xlabel("Sum of actions (selected binary features)")
    ax.set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(out_dir / "engagement_score_hist.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Segment analysis (returning vs new)
# -----------------------------
def save_segment_analysis(
    df: pd.DataFrame, binary_cols: List[str], target: str, segment_col: str, out_dir: Path
) -> None:
    """Compare conversion and engagement by segment (e.g., returning_user 0 vs 1)."""
    ensure_dir(out_dir)
    if segment_col not in df.columns:
        return
    seg_dir = out_dir / f"segment_{segment_col}"
    ensure_dir(seg_dir)

    conv = df.groupby(segment_col)[target].agg(["mean", "count"]).rename(columns={"mean": "conversion_rate"})
    conv["conversion_rate"] = (conv["conversion_rate"] * 100).round(2)
    conv.to_csv(seg_dir / "conversion_by_segment.csv")

    features = [c for c in binary_cols if c not in {target}]
    rates = df.groupby(segment_col)[features].mean().T
    rates = (rates * 100).round(2)
    rates.to_csv(seg_dir / "engagement_rates_by_segment.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    conv["conversion_rate"].plot(kind="bar", color=["#adb5bd", "#0ca678"], ax=ax)
    ax.set_title(f"Conversion by {segment_col} (%)")
    ax.set_xlabel(segment_col)
    ax.set_ylabel("Conversion (%)")
    for i, v in enumerate(conv["conversion_rate"].values):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(seg_dir / "conversion_by_segment.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    input_csv, out_dir, _ = get_paths()
    ensure_dir(out_dir)

    df = load_dataset(input_csv)
    cols = classify_columns(df)
    binary_cols = cols["binary"]
    numeric_non_binary = cols["numeric_non_binary"]

    target = "ordered"

    # 1) Target distribution
    save_target_distribution(df, target, out_dir)

    # 2) Bivariate (feature vs target)
    bivariate_binary_vs_target(df, binary_cols, target, out_dir)
    bivariate_numeric_vs_target_corr(df, numeric_non_binary, target, out_dir)

    # 3) Correlation matrix & multicollinearity flags
    save_binary_correlation_and_flags(df, binary_cols, out_dir, threshold=0.8)

    # 4) Behavior patterns (MVP engagement score)
    exclude_from_score = ["ordered", "loc_uk", "returning_user"] + [c for c in binary_cols if c.startswith("device_")]
    save_engagement_score(df, binary_cols, out_dir, exclude=exclude_from_score)

    # 5) Segment analysis (returning vs new)
    save_segment_analysis(df, binary_cols, target=target, segment_col="returning_user", out_dir=out_dir)


if __name__ == "__main__":
    # Do not execute per instructions.
    pass


