from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def save_chart(
    df: pd.DataFrame,
    fig_name: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    
    Plot a DataFrame and save it, then clear the figure

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot. Each column becomes a separate line in the legend

    fig_name : str
        Output filename (e.g. "Experiment1.png"). Extension is optional; matplotlib infers format from the suffix

    title : str
        Chart title

    x_label : str
        Label for the horizontal axis

    y_label : str
        Label for the vertical axis

    """
    
    df.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fig_name)
    plt.clf()
