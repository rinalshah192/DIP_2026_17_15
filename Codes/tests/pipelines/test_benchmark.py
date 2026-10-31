import numpy as np
import pandas as pd

from fabric_defect.core.config import PipelineConfig
from fabric_defect.pipelines.benchmark import assign_folds, split_images


def test_split_images_is_image_level_seeded_and_three_way():
    paths = [f"img{i}" for i in range(20)]
    fit, calib, test = split_images(paths, 0.2, np.random.default_rng(0), test_fraction=0.2)
    assert (len(fit), len(calib), len(test)) == (12, 4, 4)
    assert sorted(fit + calib + test) == sorted(paths) and not (set(fit) & set(calib) | set(fit) & set(test))
    assert split_images(paths, 0.2, np.random.default_rng(0), test_fraction=0.2) == (fit, calib, test)


def test_split_images_keeps_a_fit_image_when_data_are_scarce():
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    assert split_images(["only"], 0.2, rng(), 0.2) == (["only"], [], [])
    fit, calib, test = split_images(["a", "b"], 0.2, rng(), 0.2)
    assert len(fit) == 1 and len(calib) == 1 and test == []  # no negative before 3 images
    fit, calib, test = split_images(["a", "b", "c"], 0.2, rng(), 0.2)
    assert (len(fit), len(calib), len(test)) == (1, 1, 1)
    assert split_images(list("abcde"), 0.0, rng())[1:] == ([], [])


def test_folds_never_mix_and_cover_every_strip():
    cfg = PipelineConfig()
    test = pd.DataFrame({"image_path": [f"d{i}" for i in range(12)] + ["solo"],
                         "fabric_type": ["f"] * 12 + ["g"]})
    neg = pd.DataFrame({"image_path": ["n0", "n1", "m0"], "fabric_type": ["f", "f", "g"]})
    folds = assign_folds(test, neg, cfg)
    assert set(folds) == set(test["image_path"]) | set(neg["image_path"])
    assert sorted({folds[f"d{i}"] for i in range(12)}) == [0, 1, 2, 3, 4]
    assert folds["solo"] == folds["m0"] == -1  # one defective strip: nothing to learn from
    assert assign_folds(test, neg, cfg) == folds  # seeded


def test_background_rows_exclude_defect_windows_and_other_strips():
    from fabric_defect.features.cache import ImageFeatures
    from fabric_defect.pipelines.benchmark import background
    feats = {p: ImageFeatures((64, 128), (0, 128), {64: np.array([[0, 0], [0, 64]])},
                              {64: {"glcm": np.array([[1.0], [2.0]]) + i * 10}})
             for i, p in enumerate(["a", "b"])}
    counts = {"a": {64: np.array([0, 7])}, "b": {64: np.array([0, 0])}}
    rows = background(feats, counts, ["a"], 64, ("glcm",))
    assert rows.tolist() == [[1.0]]  # only a's defect-free window; b is not a training strip
    assert background(feats, counts, [], 64, ("glcm",)) is None
