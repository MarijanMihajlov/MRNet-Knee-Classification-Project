
"""
Metrics.py

A lightweight class for computing clinical classification metrics from
confusion-matrix counts. Optionally computes AUC-ROC and AUC-PR if
probability scores are provided.
"""

from typing import Optional, Dict
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

class Metrics:
    """
    Compute classification metrics from confusion-matrix counts.

    Parameters
    ----------
    tp : int
        True positives (correctly identified injuries).
    fp : int
        False positives (normal knees called injured).
    tn : int
        True negatives (correctly identified normal knees).
    fn : int
        False negatives (missed injuries).

    Optional (for AUC metrics)
    --------------------------
    y_true : array-like, optional
        Ground-truth binary labels (0 = normal, 1 = injury).
    y_scores : array-like, optional
        Predicted probabilities or scores for the positive class.
        Must be provided alongside y_true to compute AUC-ROC / AUC-PR.
    """

    def __init__(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        y_true: Optional[np.ndarray] = None,
        y_scores: Optional[np.ndarray] = None,
    ):
        # Validate counts
        if any(v < 0 for v in (tp, fp, tn, fn)):
            raise ValueError("Confusion-matrix counts must be non-negative.")
        if y_true is not None and y_scores is not None:
            if len(y_true) != len(y_scores):
                raise ValueError("y_true and y_scores must have the same length.")

        self.tp = int(tp)
        self.fp = int(fp)
        self.tn = int(tn)
        self.fn = int(fn)

        self.y_true = np.asarray(y_true) if y_true is not None else None
        self.y_scores = np.asarray(y_scores) if y_scores is not None else None

        # Pre-compute denominators to avoid repeated checks
        self._pos_actual = self.tp + self.fn          # all actual injuries
        self._neg_actual = self.tn + self.fp          # all actual normal
        self._pos_pred = self.tp + self.fp           # all predicted injuries
        self._neg_pred = self.tn + self.fn            # all predicted normal
        self._total = self.tp + self.fp + self.tn + self.fn

    # ------------------------------------------------------------------ #
    # Core metrics derived from the confusion matrix
    # ------------------------------------------------------------------ #

    @property
    def sensitivity(self) -> Optional[float]:
        """Recall / Sensitivity: TP / (TP + FN)."""
        return self._safe_div(self.tp, self._pos_actual)

    @property
    def recall(self) -> Optional[float]:
        """Alias for sensitivity."""
        return self.sensitivity

    @property
    def specificity(self) -> Optional[float]:
        """Specificity: TN / (TN + FP)."""
        return self._safe_div(self.tn, self._neg_actual)

    @property
    def precision(self) -> Optional[float]:
        """Precision: TP / (TP + FP)."""
        return self._safe_div(self.tp, self._pos_pred)

    @property
    def npv(self) -> Optional[float]:
        """Negative Predictive Value: TN / (TN + FN)."""
        return self._safe_div(self.tn, self._neg_pred)

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy: (TP + TN) / Total."""
        return self._safe_div(self.tp + self.tn, self._total)

    @property
    def fpr(self) -> Optional[float]:
        """False Positive Rate: FP / (FP + TN)."""
        return self._safe_div(self.fp, self._neg_actual)

    @property
    def fnr(self) -> Optional[float]:
        """False Negative Rate: FN / (FN + TP)."""
        return self._safe_div(self.fn, self._pos_actual)

    # ------------------------------------------------------------------ #
    # F-beta scores
    # ------------------------------------------------------------------ #

    def f_beta(self, beta: float = 1.0) -> Optional[float]:
        """
        Compute F-beta score.

        F1  (beta=1) balances precision and recall equally.
        F2  (beta=2) weights recall higher.
        """
        precision = self.precision
        recall = self.recall
        if precision is None or recall is None or (precision == 0 and recall == 0):
            return 0.0

        beta_sq = beta ** 2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    @property
    def f1(self) -> Optional[float]:
        """F1 score (beta = 1)."""
        return self.f_beta(beta=1.0)

    @property
    def f2(self) -> Optional[float]:
        """F2 score (beta = 2), weights recall higher."""
        return self.f_beta(beta=2.0)

    # ------------------------------------------------------------------ #
    # Threshold-independent metrics (require y_true + y_scores)
    # ------------------------------------------------------------------ #

    @property
    def auc_roc(self) -> Optional[float]:
        """Area Under the ROC Curve. Requires y_scores."""
        if self.y_true is None or self.y_scores is None:
            return None
        return self._compute_auc_roc()

    @property
    def auc_pr(self) -> Optional[float]:
        """
        Area Under the Precision-Recall Curve.
        More informative than AUC-ROC when the positive class is rare.
        Requires y_scores.
        """
        if self.y_true is None or self.y_scores is None:
            return None
        return self._compute_auc_pr()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Optional[float]]:
        """
        Return all metrics in a dictionary.

        Keys
        ----
        sensitivity, recall, specificity, precision, npv,
        accuracy, fpr, fnr,
        f1, f2,
        auc_roc, auc_pr
        """
        return {
            "sensitivity": self.sensitivity,
            "recall": self.recall,
            "specificity": self.specificity,
            "precision": self.precision,
            "npv": self.npv,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "f1": self.f1,
            "f2": self.f2,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
        }

    def __repr__(self) -> str:
        lines = [f"Metrics(TP={self.tp}, FP={self.fp}, TN={self.tn}, FN={self.fn})"]
        for key, val in self.to_dict().items():
            if val is not None:
                lines.append(f"  {key:12s}: {val:.4f}")
            else:
                lines.append(f"  {key:12s}: N/A (needs scores)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_div(numerator, denominator):
        """Return numerator/denominator or None if denominator is 0."""
        return numerator / denominator if denominator > 0 else None

    def _compute_auc_roc(self) -> float:
        """Manually compute AUC-ROC via the trapezoidal rule."""
        # Sort by descending score
        order = np.argsort(-self.y_scores)
        y_true_sorted = self.y_true[order]

        # Cumulative counts
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        # Total positives / negatives
        P = tps[-1]
        N = fps[-1]
        if P == 0 or N == 0:
            return float('nan')

        # TPR and FPR at each threshold
        tpr = np.concatenate(([0.0], tps / P))
        fpr = np.concatenate(([0.0], fps / N))

        # Trapezoidal integration
        return float(_trapz(tpr, fpr))

    def _compute_auc_pr(self) -> float:
        """Manually compute AUC-PR via the trapezoidal rule on PR curve."""
        # Sort by descending score
        order = np.argsort(-self.y_scores)
        y_true_sorted = self.y_true[order]

        # Cumulative counts
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        P_total = tps[-1]
        if P_total == 0:
            return float('nan')

        precision = np.concatenate(([1.0], tps / (tps + fps)))
        recall = np.concatenate(([0.0], tps / P_total))

        # Trapezoidal integration (standard approximation for AUC-PR)
        return float(_trapz(precision, recall))