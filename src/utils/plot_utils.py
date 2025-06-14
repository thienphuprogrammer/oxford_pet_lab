from __future__ import annotations

"""plot_utils.py
Light-weight Matplotlib helpers for visualising training progress & results.
These helpers are intentionally framework-agnostic and require only a history
python dictionary as produced by :class:`src.training.trainer.TrainingLogger`.
"""

from typing import Dict, List
import matplotlib.pyplot as plt
import pathlib

__all__ = ["plot_training_history"]


_DEF_METRIC_COLORS = {
    "loss": "tab:blue",
    "val_loss": "tab:orange",
}


def _ensure_parent(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_history(history: Dict[str, List[float]], save_path: str | pathlib.Path) -> None:
    """Plot loss & metric curves stored in *history* dict.

    Parameters
    ----------
    history
        Dictionary where each key maps to a list of values (one per epoch).
    save_path
        Destination filepath. The parent directories will be created.
    """
    if not history:
        raise ValueError("Empty history dictionary – nothing to plot.")

    metrics = [k for k in history.keys() if k not in {"learning_rate", "epoch_time"}]
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    if not epochs:
        # Fallback – determine epochs from length of first metric list
        first_key = metrics[0]
        epochs = list(range(1, len(history[first_key]) + 1))

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in metrics:
        y = history[m]
        ax.plot(epochs, y, label=m.replace("_", " ").title(), color=_DEF_METRIC_COLORS.get(m))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    save_path = _ensure_parent(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
