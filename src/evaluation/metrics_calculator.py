from __future__ import annotations

"""metrics_calculator.py
Provide convenient aliases for metric classes used during evaluation. These
classes are implemented in *src.training.metrics* – this module simply re-
exports them so that the evaluation package can be used without importing from
training directly.
"""

from src.training.metrics import SOTAMetrics

__all__ = ["SOTAMetrics"]
