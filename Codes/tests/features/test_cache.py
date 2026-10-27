from dataclasses import replace

import cv2
import numpy as np

from fabric_defect.core.config import GLCMConfig, PipelineConfig
from fabric_defect.features.cache import compute_features, load_features


def _strips(tmp_path, n=3):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        img = np.full((256, 700), 255, np.uint8)
        img[:, 50:650] = rng.integers(40, 200, (256, 600))  # fabric between white margins
        p = tmp_path / f"{i:04d}_000_00.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def test_features_cover_roi_in_strip_coordinates(tmp_path):
    f = compute_features(_strips(tmp_path, 1)[0], PipelineConfig(), [(256, 128)])
    assert f.shape == (256, 700) and f.roi == (90, 610)  # 40 px selvage margin inside the fabric
    xs = f.coords[256][:, 1]
    assert xs.min() == 90 and xs.max() == 610 - 256
    assert {k: v.shape[1] for k, v in f.blocks[256].items()} == {"glcm": 30, "lbp": 28, "gabor": 24}


def test_cache_round_trip_and_parallel_match(tmp_path):
    paths, cache = _strips(tmp_path), tmp_path / "cache"
    first = load_features(paths, PipelineConfig(), [(256, 128), (64, 32)], cache, workers=2)
    files = list(cache.glob("features-*/*.npz"))
    assert len(files) == 3
    again = load_features(paths, PipelineConfig(), [(256, 128), (64, 32)], cache, workers=1)
    for p in paths:
        for s in (256, 64):
            np.testing.assert_array_equal(first[p].coords[s], again[p].coords[s])
            for b in ("glcm", "lbp", "gabor"):
                np.testing.assert_array_equal(first[p].blocks[s][b], again[p].blocks[s][b])


def test_feature_settings_get_their_own_cache(tmp_path):
    paths, cache = _strips(tmp_path, 1), tmp_path / "cache"
    load_features(paths, PipelineConfig(), [(256, 128)], cache)
    load_features(paths, replace(PipelineConfig(), glcm=GLCMConfig(levels=32)), [(256, 128)], cache)
    model_only = replace(PipelineConfig(), seed=7)  # modelling settings must reuse the cache
    load_features(paths, model_only, [(256, 128)], cache)
    assert len(list(cache.glob("features-*"))) == 2
