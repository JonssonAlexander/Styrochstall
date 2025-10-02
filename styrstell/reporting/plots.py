"""Plotting helpers for policy comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_policy_comparison(kpis: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    """Generate a bar chart comparing completed trips across policies."""

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    order = kpis.sort_values("completed_trips", ascending=False)
    sns.barplot(data=order, x="policy", y="completed_trips", hue="feasible", palette="viridis", ax=ax)
    ax.set_ylabel("Completed trips")
    ax.set_xlabel("Policy")
    ax.set_title("Policy comparison: completed trips and feasibility")
    ax.legend(title="Feasible")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig
