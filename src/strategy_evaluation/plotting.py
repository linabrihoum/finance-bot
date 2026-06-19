from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Resolve the outputs directory relative to this file so charts are always
# written to <repo_root>/outputs/ regardless of the launch directory.
_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def save_chart(
    df: pd.DataFrame,
    fig_name: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot a DataFrame and save it to outputs/, then clear the figure.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot. Each column becomes a separate line in the legend.

    fig_name : str
        Output filename (e.g. "Experiment1.png"). Saved under outputs/.

    title : str
        Chart title.

    x_label : str
        Label for the horizontal axis.

    y_label : str
        Label for the vertical axis.

    """
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    df.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(_OUTPUTS_DIR / fig_name)
    plt.clf()
