import numpy as np

from fabric_defect.core.config import PreprocessConfig
from fabric_defect.features.preprocessing import Preprocessor, detect_fabric_roi, extract_patches


def test_patches_cover_the_right_edge():
    img = np.zeros((256, 1000), np.uint8)
    out = extract_patches(img, patch_size=256, stride=128)
    xs = [x for _, x in out["coordinates"]]
    assert xs[0] == 0 and xs[-1] == 1000 - 256
    assert out["patches"].shape == (len(xs), 256, 256)


def test_patch_labels_offsets_and_threshold():
    img = np.zeros((256, 768), np.uint8)
    mask = np.zeros_like(img)
    mask[0:5, 600:606] = 255  # 30 px, only in the last window
    out = extract_patches(img, mask=mask, patch_size=256, stride=256, offset_x=40, defect_min_pixels=25)
    assert [x for _, x in out["coordinates"]] == [40, 296, 552]
    assert out["labels"].tolist() == [0, 0, 1]
    out = extract_patches(img, mask=mask, patch_size=256, stride=256, defect_min_pixels=30)
    assert out["labels"].tolist() == [0, 0, 0]  # strictly more than the threshold


def test_strip_smaller_than_patch_gives_no_patches():
    out = extract_patches(np.zeros((100, 100), np.uint8), patch_size=256, stride=128)
    assert out["patches"].shape == (0, 256, 256)


def test_roi_drops_white_margins_and_selvage():
    rng = np.random.default_rng(0)
    img = np.full((256, 2000), 255, np.uint8)
    img[:, 300:1700] = rng.integers(60, 160, (256, 1400))
    assert detect_fabric_roi(img, margin=40, min_width=256) == (340, 1660)


def test_roi_falls_back_when_fabric_is_too_narrow():
    img = np.full((256, 2000), 255, np.uint8)
    assert detect_fabric_roi(img, margin=40, min_width=256) == (0, 2000)


def test_preprocessor_keeps_shape_and_dtype():
    img = np.random.default_rng(1).integers(0, 255, (256, 512)).astype(np.uint8)
    stages = Preprocessor(PreprocessConfig()).process(img)
    for key in ("raw", "flattened", "filtered", "normalized"):
        assert stages[key].shape == img.shape and stages[key].dtype == np.uint8
