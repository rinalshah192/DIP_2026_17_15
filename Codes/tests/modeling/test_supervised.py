import numpy as np
import pytest

from fabric_defect.core.config import SupervisedConfig
from fabric_defect.features.cache import ImageFeatures
from fabric_defect.features.pixel import PixelFeatureExtractor
from fabric_defect.modeling.supervised import WindowClassifier, sample_pixels, window_matrix


def _feats(n32=6, seed=0):
    rng = np.random.default_rng(seed)
    blocks = lambda n: {"glcm": rng.normal(size=(n, 30)), "lbp": rng.normal(size=(n, 28)),  # noqa: E731
                        "gabor": rng.normal(size=(n, 24))}
    coords = {32: np.array([(0, 16 * i) for i in range(n32)]), 64: np.array([(0, 0), (0, 32)]),
              256: np.array([(0, 0)])}
    return ImageFeatures((256, 128), (0, 128), coords, {32: blocks(n32), 64: blocks(2), 256: blocks(1)})


def test_window_matrix_layout():
    X = window_matrix(_feats(), "fabric_03", SupervisedConfig(window=32, ensemble=(), context=(64, 256)))
    # raw + strip-normalised 32 blocks, strip-normalised 64 and 256 context, one-hot fabric
    assert X.shape == (6, 82 * 2 + 82 * 2 + 7)
    assert X[:, -7:].tolist() == [[0, 0, 0, 1, 0, 0, 0]] * 6
    z = X[:, 82:164]
    np.testing.assert_allclose(np.median(z, axis=0), 0, atol=1e-5)  # strip-normalised columns


def test_window_classifier_learns_and_needs_positives():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] > 1.8).astype(int)
    p = WindowClassifier(SupervisedConfig(max_iter=50, early_stopping=False)).fit(X, y).predict(X)
    assert p[y == 1].mean() > 0.8 and p[y == 0].mean() < 0.2
    with pytest.raises(ValueError):
        WindowClassifier().fit(X, np.zeros(len(X), int))


def test_sample_pixels_balances_defect_border_and_random():
    F = np.random.default_rng(0).normal(size=(64, 64, 3)).astype(np.float32)
    gt = np.zeros((64, 64), bool)
    gt[30:32, :] = True  # a 2-row broken pick: 128 defect pixels
    X, y = sample_pixels(F, gt, 50, np.random.default_rng(0))
    assert X.shape[1] == 3 and y[:50].sum() == 50  # positives first, capped at n
    assert len(y) == 150 and y[50:100].sum() == 0  # border negatives are defect-free


def test_pixel_features_shape_and_row_profile_response():
    img = np.full((64, 256), 120, np.uint8)
    img += np.random.default_rng(0).integers(0, 20, img.shape).astype(np.uint8)
    img[30, :] = 30  # dark full-width line: a missing weft yarn
    px = PixelFeatureExtractor()
    F = px.extract(img)
    assert F.shape == (64, 256, 30) and len(px.get_feature_names()) == 30
    names = px.get_feature_names()
    weft = F[..., names.index("weft_line_smooth33")]
    assert weft[30].mean() < -3 and abs(np.median(weft[10])) < 1  # the dark row, not the texture
    warp = F[..., names.index("warp_line")]
    assert abs(warp[30].mean()) < 1  # a full-width row is invisible to the warp detector


@pytest.mark.parametrize("kw", [{"window": 33}, {"window": 64, "context": (64,)}, {"folds": 1},
                                {"validation_fraction": 0.0}, {"ensemble": (16,)}, {"ensemble": (31,)}])
def test_supervised_config_validation(kw):
    with pytest.raises(ValueError):
        SupervisedConfig(**kw)


def test_hough_link_joins_dashed_yarn_lines_inside_region_only():
    from fabric_defect.modeling.supervised import hough_link
    mask = np.zeros((64, 512), np.uint8)
    for x in range(0, 400, 40):
        mask[30, x:x + 30] = 255  # a dashed broken pick: 30 px on, 10 px off
    for t in range(60):
        mask[t, 450 + t // 2] = 255  # a steep diagonal: not a yarn direction
    region = np.zeros_like(mask, bool)
    region[:, :300] = True
    out = hough_link(mask, region, min_length=64, max_gap=16, thickness=1)
    assert out[30, :290].all()  # gaps filled inside the region
    assert not out[30, 320:330].any() or (mask[30, 320:330] > 0).all()  # nothing added outside
    np.testing.assert_array_equal(out[:, 440:] > 0, mask[:, 440:] > 0)  # diagonal untouched


def test_nearest_context_matches_brute_force():
    from fabric_defect.features.preprocessing import window_coords
    from fabric_defect.modeling.supervised import nearest_on_grid
    fine = np.array(window_coords((256, 1000), 16, 8)) + [0, 40]  # offset like a cropped ROI
    grid = np.array(window_coords((256, 1000), 64, 32)) + [0, 40]
    idx = nearest_on_grid(fine, 16, grid, 64)
    d = (((fine + 8)[:, None, :] - (grid + 32)[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(d[np.arange(len(fine)), idx], d.min(1))
    with pytest.raises(ValueError):
        nearest_on_grid(fine, 16, grid[:-1], 64)  # not a full grid


def test_sample_windows_keeps_positives_and_caps_negatives():
    from fabric_defect.modeling.supervised import sample_windows
    X = np.arange(1000, dtype=float)[:, None]
    y = np.zeros(1000, int)
    y[[3, 500]] = 1
    cfg = SupervisedConfig(negatives_per_positive=20, min_negatives=300)
    Xs, ys = sample_windows(X, y, cfg, np.random.default_rng(0))
    assert ys.sum() == 2 and len(ys) == 2 + 300  # max(300, 20 * 2) negatives
    Xn, yn = sample_windows(X, np.zeros(1000, int), cfg, np.random.default_rng(0))
    assert len(yn) == 300 and yn.sum() == 0  # defect-free strips still contribute negatives
    np.testing.assert_array_equal(sample_windows(X, y, cfg, np.random.default_rng(0))[0], Xs)


def test_levels_are_finest_first():
    assert SupervisedConfig().levels() == (16, 32)
