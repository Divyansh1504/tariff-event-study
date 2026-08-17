"""Shared plotting helpers so chart style is consistent across both notebooks."""

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (9, 5)


def plot_car_curve(ar_series, title, ylabel="Cumulative AR", marker="o"):
    fig, ax = plt.subplots()
    ax.plot(ar_series.cumsum(), marker=marker)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_cumulative_return(returns_by_label: dict, title):
    fig, ax = plt.subplots()
    for label, x in returns_by_label.items():
        ax.plot((1 + x).cumprod() - 1, label=label)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.legend()
    return fig, ax


def plot_industry_heatmap(df, title, fmt=".2%", cmap="RdYlGn", center=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, center=center, ax=ax)
    ax.set_title(title)
    return fig, ax
