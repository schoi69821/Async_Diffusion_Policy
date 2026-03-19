"""Visualization utilities."""
import numpy as np
from typing import Optional, List


def plot_phase_timeline(
    phases: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot phase labels over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 2))
    phase_names = ["reach", "align", "close", "lift", "place", "return"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

    for i in range(len(phases) - 1):
        ax.axvspan(i, i + 1, color=colors[phases[i] % len(colors)], alpha=0.7)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Phase")
    ax.set_yticks(range(len(phase_names)))
    ax.set_yticklabels(phase_names)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_contact_signal(
    contact_soft: np.ndarray,
    contact_hard: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot contact signals over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(contact_soft, label="contact_soft", alpha=0.8)
    ax.fill_between(range(len(contact_hard)), contact_hard, alpha=0.3, label="contact_hard")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Contact")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
