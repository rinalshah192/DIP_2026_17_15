import numpy as np

from fabric_defect.features.descriptor import TextureDescriptor
from fabric_defect.features.glcm import GLCMFeatureExtractor
from fabric_defect.features.lbp import LBPFeatureExtractor


def _patches(n=3, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 64, 64)).astype(np.uint8)


def test_descriptor_dimensions_match_names():
    desc = TextureDescriptor()
    X = desc.extract_batch(_patches())
    assert X.shape == (3, 58)  # 5 props x 3 distances x (mean, std) + (10 + 18) LBP bins
    assert len(desc.get_feature_names()) == 58
    assert np.isfinite(X).all()


def test_lbp_histograms_are_normalised_per_scale():
    h = LBPFeatureExtractor().extract_batch(_patches())
    np.testing.assert_allclose(h[:, :10].sum(1), 1.0, atol=1e-5)
    np.testing.assert_allclose(h[:, 10:].sum(1), 1.0, atol=1e-5)


def test_glcm_names_follow_feature_order():
    names = GLCMFeatureExtractor().get_feature_names()
    assert names[:6] == [
        "glcm_contrast_d1_mean", "glcm_contrast_d2_mean", "glcm_contrast_d3_mean",
        "glcm_contrast_d1_std", "glcm_contrast_d2_std", "glcm_contrast_d3_std",
    ]


def test_flat_patch_does_not_crash():
    X = TextureDescriptor().extract_batch(np.full((1, 64, 64), 128, np.uint8))
    assert X.shape == (1, 58)


def test_features_are_deterministic():
    p = _patches(seed=3)
    np.testing.assert_array_equal(TextureDescriptor().extract_batch(p), TextureDescriptor().extract_batch(p))
