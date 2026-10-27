import numpy as np

from fabric_defect.pipelines.benchmark import split_images


def test_split_images_is_image_level_and_seeded():
    paths = [f"img{i}" for i in range(20)]
    fit, calib = split_images(paths, 0.25, np.random.default_rng(0))
    assert len(calib) == 5 and not set(fit) & set(calib) and sorted(fit + calib) == sorted(paths)
    assert split_images(paths, 0.25, np.random.default_rng(0)) == (fit, calib)
    assert split_images(["only"], 0.25, np.random.default_rng(0)) == (["only"], [])
    assert split_images(["a", "b"], 0.25, np.random.default_rng(0))[1] != []
