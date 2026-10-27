import numpy as np
import pytest

from fabric_defect.core.config import ModelConfig
from fabric_defect.modeling.reference import TextureReferenceModel, robust_z


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    X = rng.normal(size=(400, 5)) @ A  # correlated features
    return X, rng


def test_vectorised_score_matches_explicit_formula(data):
    X, rng = data
    m = TextureReferenceModel("f").fit(X)
    Q = rng.normal(size=(50, 5))
    expected = [np.sqrt((q - m.mean_vector) @ m.inv_covariance @ (q - m.mean_vector)) for q in Q]
    np.testing.assert_allclose(m.score_batch(Q), expected, rtol=1e-5)
    assert m.score_patch(Q[0]) == pytest.approx(expected[0], rel=1e-5)


def test_mahalanobis_is_scale_invariant(data):
    X, rng = data
    Q = rng.normal(size=(20, 5))
    s1 = TextureReferenceModel("f", ModelConfig(regularization=0.0)).fit(X).score_batch(Q)
    s2 = TextureReferenceModel("f", ModelConfig(regularization=0.0)).fit(X * 1000).score_batch(Q * 1000)
    np.testing.assert_allclose(s1, s2, rtol=1e-4)


def test_scoring_before_fit_raises():
    with pytest.raises(RuntimeError):
        TextureReferenceModel("f").score_batch(np.zeros((1, 3)))


def test_empty_fit_raises():
    with pytest.raises(ValueError):
        TextureReferenceModel("f").fit(np.empty((0, 3)))


def test_held_out_calibration_is_used_and_flagged(data):
    X, _ = data
    in_sample = TextureReferenceModel("f").fit(X[:40])
    held_out = TextureReferenceModel("f").fit(X[:40], X[40:])
    assert not in_sample.calibrated_out_of_sample and held_out.calibrated_out_of_sample
    # With few fit samples the in-sample spread is optimistic, so tau comes out lower
    assert held_out.adaptive_threshold > in_sample.adaptive_threshold


def test_hybrid_scoring_flags_an_outlier_patch(data):
    X, _ = data
    m = TextureReferenceModel("f", ModelConfig(scoring="hybrid", k=2.5)).fit(X)
    strip = X[:30].copy()
    strip[7] += 25.0
    scores, tau = m.score_strip(strip)
    assert tau == 2.5
    assert np.argmax(scores) == 7 and scores[7] >= tau


def test_deviation_map_averages_overlaps():
    m = TextureReferenceModel("f")
    hm = m.generate_deviation_map(np.array([1.0, 3.0]), [(0, 0), (0, 2)], strip_shape=(4, 6), patch_size=4)
    np.testing.assert_allclose(hm[0], [1, 1, 2, 2, 3, 3])


def test_invalid_model_config_rejected():
    with pytest.raises(ValueError):
        ModelConfig(scoring="nope")
    with pytest.raises(ValueError):
        ModelConfig(calibration_fraction=1.0)


def test_strip_mahalanobis_removes_a_strip_wide_offset(data):
    X, _ = data
    m = TextureReferenceModel("f", ModelConfig(scoring="strip_mahalanobis", k=2.5)).fit(X)
    strip = X[:40].copy()
    strip[5] += 25.0
    plain, _ = m.score_strip(strip)
    shifted, tau = m.score_strip(strip + 3.0)  # same strip, globally shifted capture
    assert tau == 2.5 and np.argmax(plain) == np.argmax(shifted) == 5
    assert (shifted == 0).mean() >= 0.5  # patches at or below the strip median score 0


def test_robust_z_is_one_sided():
    z = robust_z(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert z[:3].tolist() == [0.0, 0.0, 0.0] and z[-1] > 50
