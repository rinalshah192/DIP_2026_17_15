"""
Fused patch descriptor.
ECE501 - Digital Image Processing (Group 15)
"""

from typing import List

import numpy as np

from fabric_defect.core.config import GLCMConfig, LBPConfig
from fabric_defect.features.glcm import GLCMFeatureExtractor
from fabric_defect.features.lbp import LBPFeatureExtractor


class TextureDescriptor:
    """Fused GLCM + LBP patch descriptor (D = 30 + 28 = 58 with the default config)."""

    def __init__(self, glcm: GLCMConfig = GLCMConfig(), lbp: LBPConfig = LBPConfig()):
        self.glcm = GLCMFeatureExtractor(glcm)
        self.lbp = LBPFeatureExtractor(lbp)

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        return np.hstack([self.glcm.extract_batch(patches), self.lbp.extract_batch(patches)])

    def get_feature_names(self) -> List[str]:
        return self.glcm.get_feature_names() + self.lbp.get_feature_names()
