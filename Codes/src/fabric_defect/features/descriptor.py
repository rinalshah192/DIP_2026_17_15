"""
Fused patch descriptor.
ECE501 - Digital Image Processing (Group 15)
"""

from typing import List

import numpy as np

from fabric_defect.core.config import FeatureConfig, GaborConfig, GLCMConfig, LBPConfig
from fabric_defect.features.gabor import GaborFeatureExtractor
from fabric_defect.features.glcm import GLCMFeatureExtractor
from fabric_defect.features.lbp import LBPFeatureExtractor


class TextureDescriptor:
    """Concatenation of the feature blocks named in FeatureConfig.use.

    Default (glcm, lbp): D = 30 + 28 = 58. Adding gabor: + 24 = 82.
    """

    def __init__(
        self,
        glcm: GLCMConfig = GLCMConfig(),
        lbp: LBPConfig = LBPConfig(),
        gabor: GaborConfig = GaborConfig(),
        features: FeatureConfig = FeatureConfig()
    ):
        self.use = tuple(features.use)
        self.extractors = {
            "glcm": GLCMFeatureExtractor(glcm),
            "lbp": LBPFeatureExtractor(lbp),
            "gabor": GaborFeatureExtractor(gabor),
        }

    def extract_batch(self, patches: np.ndarray) -> np.ndarray:
        return np.hstack([self.extractors[b].extract_batch(patches) for b in self.use])

    def get_feature_names(self) -> List[str]:
        return [n for b in self.use for n in self.extractors[b].get_feature_names()]
