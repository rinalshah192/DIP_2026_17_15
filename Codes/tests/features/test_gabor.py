import numpy as np
import pytest

from fabric_defect.core.config import FeatureConfig
from fabric_defect.features.descriptor import TextureDescriptor
from fabric_defect.features.gabor import GaborFeatureExtractor, sigma_for


def _stripes(period, vertical=True, n=256):
    wave = (127 + 100 * np.cos(2 * np.pi * np.arange(n) / period)).astype(np.uint8)
    img = np.tile(wave, (n, 1))
    return img if vertical else img.T


def test_one_octave_bandwidth_sigma():
    assert sigma_for(8.0, 1.0) / 8.0 == pytest.approx(0.5622, abs=1e-4)


@pytest.mark.parametrize("period, vertical, expected", [
    (8, True, "gabor_l8_o0_mean"),     # wave along x
    (8, False, "gabor_l8_o90_mean"),   # wave along y
    (16, True, "gabor_l16_o0_mean"),
])
def test_strongest_filter_matches_texture(period, vertical, expected):
    g = GaborFeatureExtractor()
    f = g.extract_windows(_stripes(period, vertical), [(0, 0)], 256)[0]
    names = g.get_feature_names()
    assert names[int(np.argmax(f[:12]))] == expected


def test_flat_image_has_no_response():
    f = GaborFeatureExtractor().extract_windows(np.full((128, 128), 90, np.uint8), [(0, 0)], 128)[0]
    np.testing.assert_allclose(f, 0, atol=1e-4)


def test_windows_match_names_and_batch():
    g = GaborFeatureExtractor()
    img = np.random.default_rng(0).integers(0, 255, (128, 256)).astype(np.uint8)
    X = g.extract_windows(img, [(0, 0), (0, 128)], 128)
    assert X.shape == (2, 24) == (2, len(g.get_feature_names()))
    assert g.extract_batch(np.empty((0, 64, 64), np.uint8)).shape == (0, 24)


def test_descriptor_selects_blocks_in_order():
    p = np.random.default_rng(1).integers(0, 255, (2, 64, 64)).astype(np.uint8)
    full = TextureDescriptor(features=FeatureConfig(use=("glcm", "lbp", "gabor")))
    gab = TextureDescriptor(features=FeatureConfig(use=("gabor",)))
    assert full.extract_batch(p).shape == (2, 82)
    np.testing.assert_allclose(full.extract_batch(p)[:, 58:], gab.extract_batch(p))
    assert full.get_feature_names()[58].startswith("gabor_")


@pytest.mark.parametrize("use", [(), ("hog",)])
def test_invalid_feature_selection_rejected(use):
    with pytest.raises(ValueError):
        FeatureConfig(use=use)
