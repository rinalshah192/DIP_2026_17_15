import cv2
import numpy as np
import pytest

from fabric_defect.afid import AITEXDataset, find_masks, load_mask, parse_afid_filename


@pytest.mark.parametrize("name, expected", [
    ("0001_002_00.png", ("fabric_00", "broken_end")),
    ("0010_006_02.png", ("fabric_02", "broken_yarn")),
    ("0018_010_03.png", ("fabric_03", "broken_pick")),
    ("0044_019_04.png", ("fabric_04", "fuzzy_ball")),
    ("0094_027_05.png", ("fabric_05", "knot")),
    ("0001_000_06.png", ("fabric_06", "none")),
    ("0001_099_01.png", ("fabric_01", "unknown_099")),
])
def test_parse_afid_filename_uses_official_codes(name, expected):
    assert parse_afid_filename(name) == expected


@pytest.mark.parametrize("name", ["Fabric01_Plain_broken_end.png", "0001_002.png", "abcd_002_00.png"])
def test_parse_rejects_non_afid_names(name):
    assert parse_afid_filename(name) is None


def test_multi_part_masks_are_merged_and_resized(tmp_path):
    a = np.zeros((128, 2048), np.uint8)
    a[0:10, 0:10] = 255
    b = np.zeros((128, 2048), np.uint8)
    b[50:60, 1000:1010] = 255
    cv2.imwrite(str(tmp_path / "0044_019_04_mask1.png"), a)
    cv2.imwrite(str(tmp_path / "0044_019_04_mask2.png"), b)
    cv2.imwrite(str(tmp_path / "0045_019_04_mask.png"), a)  # other image, must not match

    paths = find_masks(tmp_path, "0044_019_04")
    assert len(paths) == 2

    mask = load_mask(paths, (256, 4096))  # nearest-neighbour 2x upscale
    assert mask.shape == (256, 4096)
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 2 * (20 * 20)


def test_load_mask_needs_paths():
    with pytest.raises(ValueError):
        load_mask([], (4, 4))


def test_catalog_fails_loudly_without_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="--synthetic"):
        AITEXDataset(tmp_path).scan_catalog()


def test_catalog_scans_published_layout(tmp_path):
    img = np.full((256, 4096), 120, np.uint8)
    (tmp_path / "NODefect_images" / "2306881-210020u").mkdir(parents=True)
    (tmp_path / "Defect_images").mkdir()
    (tmp_path / "Mask_images").mkdir()
    cv2.imwrite(str(tmp_path / "NODefect_images" / "2306881-210020u" / "0001_000_05.png"), img)
    cv2.imwrite(str(tmp_path / "Defect_images" / "0094_027_05.png"), img)
    cv2.imwrite(str(tmp_path / "Defect_images" / "0095_027_05.png"), img)  # no mask on purpose
    cv2.imwrite(str(tmp_path / "Mask_images" / "0094_027_05_mask.png"), img)

    df = AITEXDataset(tmp_path).scan_catalog().set_index("filename")
    assert df.loc["0001_000_05.png", "is_defective"] == False  # noqa: E712 (numpy bool)
    assert df.loc["0094_027_05.png", "defect_type"] == "knot"
    assert df.loc["0094_027_05.png", "fabric_type"] == "fabric_05"
    assert len(df.loc["0094_027_05.png", "mask_paths"]) == 1
    assert df.loc["0095_027_05.png", "mask_paths"] == []
