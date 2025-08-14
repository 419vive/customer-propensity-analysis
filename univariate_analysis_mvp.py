from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def get_paths() -> Tuple[Path, Path, Path]:
    """Return input CSV path, output directory, and base data directory."""
    data_dir = Path(
        "/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset"
    )
    input_csv = data_dir / "training_sample.csv"
    out_dir = data_dir / "univariate_mvp"
    return input_csv, out_dir, data_dir


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load dataset with safe defaults and explicit failure if missing."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify columns into numeric_non_binary and binary.

    A binary column is numeric/bool whose non-null unique values are a subset of {0, 1}.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    binary_cols: List[str] = []
    numeric_non_binary: List[str] = []
    for col in numeric_cols:
        values = set(pd.unique(df[col].dropna()))
        if values.issubset({0, 1}):
            binary_cols.append(col)
        else:
            numeric_non_binary.append(col)
    return {"binary": binary_cols, "numeric_non_binary": numeric_non_binary}


def compute_numeric_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Compute numeric summary with skewness and kurtosis for given columns."""
    if not columns:
        return pd.DataFrame()
    summary = df[columns].describe(percentiles=[0.5, 0.95, 0.99]).T
    summary["skewness"] = df[columns].skew()
    summary["kurtosis"] = df[columns].kurtosis()
    return summary


def compute_binary_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Compute 0/1 counts and engagement rate for binary columns."""
    rows: List[Dict[str, object]] = []
    for col in columns:
        series = df[col]
        zero_count = int((series == 0).sum())
        one_count = int((series == 1).sum())
        total_non_null = int(series.notna().sum())
        engagement_rate = float(series.mean()) if total_non_null > 0 else float("nan")
        rows.append(
            {
                "column": col,
                "zero": zero_count,
                "one": one_count,
                "total": total_non_null,
                "engagement_rate": engagement_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("engagement_rate", ascending=False)


def save_numeric_histograms(df: pd.DataFrame, columns: List[str], out_dir: Path, show: bool = False) -> None:
    """Save histogram PNGs for numeric non-binary columns.

    If show=True, keep figures open for interactive viewing; otherwise close after saving.
    """
    if not columns:
        return
    ensure_directory(out_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].plot(kind="hist", bins=30, alpha=0.85, color="#2b8a3e", ax=ax)
        ax.set_title(f"{col} — Histogram")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{col}.png", dpi=140)
        if not show:
            plt.close(fig)


def save_engagement_overview(binary_summary: pd.DataFrame, out_dir: Path, show: bool = False) -> None:
    """Save a horizontal bar chart of engagement rates for binary features (excluding 'ordered').

    If show=True, keep the figure open for interactive viewing; otherwise close after saving.
    """
    if binary_summary.empty:
        return
    df_plot = binary_summary.copy()
    if "ordered" in df_plot["column"].values:
        df_plot = df_plot[df_plot["column"] != "ordered"]
    df_plot = df_plot.sort_values("engagement_rate", ascending=True)

    ensure_directory(out_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, max(4, 0 if df_plot.empty else len(df_plot) * 0.25)))
    if not df_plot.empty:
        ax.barh(df_plot["column"], df_plot["engagement_rate"] * 100.0, color="#0ca678")
        ax.set_title("Engagement Rates by Feature (%)")
        ax.set_xlabel("Rate (%)")
        ax.set_ylabel("Feature")
        for y, x in enumerate(df_plot["engagement_rate"] * 100.0):
            ax.text(x + 0.2, y, f"{x:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "engagement_rates.png", dpi=140)
    if not show:
        plt.close(fig)


def save_binary_count_bars(df: pd.DataFrame, columns: List[str], out_dir: Path, show: bool = False) -> None:
    """Save 0/1 count bar plots for each binary column into a subfolder.

    If show=True, keep figures open for interactive viewing; otherwise close after saving.
    """
    if not columns:
        return
    figs_dir = out_dir / "binary_bars"
    ensure_directory(figs_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    for col in columns:
        series = df[col]
        counts = series.value_counts(dropna=False).reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.bar(["0", "1"], counts.values, color=["#adb5bd", "#1c7ed6"])
        ax.set_title(f"{col} — 0/1 Counts")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(figs_dir / f"bar_{col}.png", dpi=140)
        if not show:
            plt.close(fig)

def main(show: bool = False) -> None:
    """Run MVP univariate analysis and save outputs under the output directory.

    Steps:
    - Numeric summary for non-binary numeric columns
    - Binary 0/1 counts and engagement rates
    - Histograms for numeric non-binary columns
    - Engagement-rate overview for binary columns (excluding target)
    """
    input_csv, out_dir_root, _ = get_paths()
    out_dir = out_dir_root
    ensure_directory(out_dir)

    df = load_dataset(input_csv)
    groups = classify_columns(df)
    numeric_non_binary = groups["numeric_non_binary"]
    binary_cols = groups["binary"]

    numeric_summary = compute_numeric_summary(df, numeric_non_binary)
    if not numeric_summary.empty:
        numeric_summary.to_csv(out_dir / "numeric_summary.csv")

    binary_summary = compute_binary_summary(df, binary_cols)
    binary_summary.to_csv(out_dir / "binary_summary.csv", index=False)

    save_numeric_histograms(df, numeric_non_binary, out_dir, show=show)
    save_engagement_overview(binary_summary, out_dir, show=show)
    save_binary_count_bars(df, binary_cols, out_dir, show=show)

    if show:
        # Block until all open figures are closed by the user
        plt.show(block=True)


if __name__ == "__main__":
    # Default to non-interactive run; set show=True to display charts.
    main(show=True)
