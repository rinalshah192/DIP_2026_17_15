import numpy as np
import pytest

from fabric_defect.core.config import PatchConfig, PipelineConfig, RefineConfig
from fabric_defect.features.preprocessing import window_coords
from fabric_defect.modeling.refinement import coarse_region, refined_mask, windows_inside


def test_coarse_region_is_union_of_flagged_windows():
    coords = np.array([(0, 0), (0, 128), (0, 256)])
    region = coarse_region(np.array([1, 0, 1]), coords, (256, 512), 256)
    assert region[:, :256].all() and region[:, 256:].all()
    region = coarse_region(np.array([0, 0, 0]), coords, (256, 512), 256)
    assert not region.any()


def test_windows_inside_matches_brute_force():
    rng = np.random.default_rng(0)
    region = np.zeros((128, 256), bool)
    region[10:100, 30:200] = True
    coords = np.array(window_coords(region.shape, 32, 16))
    fast = windows_inside(region, coords, 32)
    slow = np.array([region[y:y + 32, x:x + 32].all() for y, x in coords])
    np.testing.assert_array_equal(fast, slow)
    assert fast.any() and not fast.all()


def test_refined_mask_keeps_only_flagged_fine_windows_inside_region():
    region = np.zeros((128, 256), bool)
    region[:, :128] = True
    coords = np.array(window_coords(region.shape, 64, 64))  # 2 rows x 4 cols
    scores = np.zeros(len(coords))
    scores[[0, 3]] = 9.0  # (0,0) is inside the region; (0,192) is outside
    mask = refined_mask(region, coords, scores, tau=5.0, size=64)
    assert np.count_nonzero(mask) == 64 * 64 and mask[:64, :64].all()


def test_refine_must_be_finer_than_patch():
    with pytest.raises(ValueError):
        PipelineConfig(patch=PatchConfig(size=64, stride=64), refine=RefineConfig(sizes=(64,)))


@pytest.mark.parametrize("sizes", [(), (32, 64), (64, 64), (63,)])
def test_refine_levels_must_be_even_and_coarse_to_fine(sizes):
    with pytest.raises(ValueError):
        RefineConfig(sizes=sizes)


def test_refine_cascade_scales_use_half_stride():
    assert RefineConfig(sizes=(64, 32)).scales() == ((64, 32), (32, 16))


def test_cascade_keeps_only_fine_windows_inside_the_previous_level():
    # Coarse region covers x < 128; level 64 flags only (0, 0); level 32 may only fire inside it
    region = np.zeros((64, 256), bool)
    region[:, :128] = True
    c64 = np.array(window_coords(region.shape, 64, 64))
    level1 = refined_mask(region, c64, np.where((c64 == [0, 0]).all(1), 9.0, 0.0), 5.0, 64) > 0
    c32 = np.array(window_coords(region.shape, 32, 32))
    hit = np.full(len(c32), 9.0)  # every 32x32 window looks anomalous
    level2 = refined_mask(level1, c32, hit, 5.0, 32)
    assert np.count_nonzero(level2) == 64 * 64 and level2[:, 64:].max() == 0
