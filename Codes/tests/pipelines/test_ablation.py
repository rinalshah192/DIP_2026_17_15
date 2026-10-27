import pandas as pd
import pytest

from fabric_defect.cli.main import main
from fabric_defect.core.config import PipelineConfig
from fabric_defect.pipelines.ablation import load_grid, slug
from fabric_defect.pipelines.benchmark import Split, fit_models


def _grid(tmp_path, body):
    (tmp_path / "base.toml").write_text("[patch]\nstride = 256\n")
    g = tmp_path / "grid.toml"
    g.write_text('base = "base.toml"\n' + body)
    return g


def test_grid_applies_overrides_on_top_of_base(tmp_path):
    g = _grid(tmp_path, '''
[[variant]]
name = "default"
[[variant]]
name = "gabor only"
overrides = { features = { use = ["gabor"] }, model = { k = 3.0 } }
''')
    base, variants = load_grid(g)
    assert base.patch.stride == 256
    (n1, c1), (n2, c2) = variants
    assert c1 == base and c2.features.use == ("gabor",) and c2.model.k == 3.0 and c2.patch.stride == 256
    assert slug(n2) == "gabor-only"


@pytest.mark.parametrize("body, error", [
    ('[[variant]]\nname = "a"\n[[variant]]\nname = "a"\n', ValueError),
    ('[[variant]]\nname = "a"\noverrides = { model = { bogus = 1 } }\n', TypeError),
    ('', ValueError),
])
def test_bad_grids_are_rejected(tmp_path, body, error):
    with pytest.raises(error):
        load_grid(_grid(tmp_path, body))


def test_global_reference_shares_one_model(tmp_path):
    import numpy as np
    from fabric_defect.features.cache import ImageFeatures
    rng = np.random.default_rng(0)
    feats = {f"{f}{i}": ImageFeatures((256, 512), (0, 512), {256: np.zeros((3, 2), int)},
                                      {256: {"glcm": rng.normal(size=(3, 4)), "lbp": rng.normal(size=(3, 2))}})
             for f in "ab" for i in range(4)}
    split = Split(fit={"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"]}, calib={"a": ["a3"], "b": ["b3"]})
    cfg = PipelineConfig().with_overrides({"model": {"reference": "global", "scoring": "mahalanobis"}})
    models = fit_models(feats, split, cfg)
    assert models["a"] is models["b"] and len(models["a"].mean_vector) == 6


def test_ablate_cli_end_to_end_on_synthetic_data(tmp_path):
    g = _grid(tmp_path, '''
[[variant]]
name = "default"
[[variant]]
name = "mahalanobis + refine"
overrides = { model = { scoring = "mahalanobis" }, refine = { enabled = true } }
''')
    out = tmp_path / "out"
    assert main(["ablate", "--synthetic", "--grid", str(g), "--out", str(out), "--max-test", "7"]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["variant"]) == ["default", "mahalanobis + refine"]
    assert (summary["n_images"] == 7).all() and summary["refine"].tolist() == [False, True]
    assert (out / "mahalanobis-refine.csv").exists() and (out / "run_config.json").exists()


def test_ablate_cli_rejects_bad_grid(tmp_path):
    assert main(["ablate", "--synthetic", "--grid", str(_grid(tmp_path, "")), "--out", str(tmp_path / "o")]) == 2
