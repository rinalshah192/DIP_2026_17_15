"""
Learned one-class baselines.
ECE501 - Digital Image Processing (Group 15)

One-Class SVM (Schölkopf et al., 2001) and Isolation Forest (Liu et al., 2008),
trained on the same defect-free descriptors as the Mahalanobis reference and
thresholded the same way (tau = mu + k sigma on held-out defect-free images),
so that only the scoring function differs between them.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from fabric_defect.core.config import ModelConfig


class OneClassBaseline:
    """sklearn one-class model behind the reference-model interface; higher score = more anomalous."""

    def __init__(self, fabric_type: str, cfg: ModelConfig = ModelConfig(), seed: int = 0):
        self.fabric_type = fabric_type
        self.adaptive_k = cfg.k
        if cfg.scoring == "ocsvm":
            self.estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=cfg.ocsvm_nu)
        elif cfg.scoring == "iforest":
            self.estimator = IsolationForest(n_estimators=cfg.iforest_trees, random_state=seed)
        else:
            raise ValueError(f"{cfg.scoring!r} is not a one-class baseline")
        self.adaptive_threshold = np.inf
        self.calibrated_out_of_sample = False

    def fit(self, X_normal: np.ndarray, X_calib: Optional[np.ndarray] = None) -> "OneClassBaseline":
        if len(X_normal) == 0:
            raise ValueError("Cannot fit baseline with empty feature array.")
        # Standardise on the fit set: the RBF kernel and tree splits see every feature on one scale
        self.mean = X_normal.mean(axis=0)
        self.std = X_normal.std(axis=0) + 1e-6
        self.estimator.fit(self._scale(X_normal))
        self.calibrated_out_of_sample = X_calib is not None and len(X_calib) > 0
        s = self.score_batch(X_calib if self.calibrated_out_of_sample else X_normal)
        self.adaptive_threshold = float(s.mean() + self.adaptive_k * (s.std() + 1e-6))
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        # decision_function is positive for inliers, so negate it
        return (-self.estimator.decision_function(self._scale(X))).astype(np.float32)

    def score_strip(self, X_strip: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.score_batch(X_strip), self.adaptive_threshold
