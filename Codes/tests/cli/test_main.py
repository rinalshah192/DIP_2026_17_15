import json

import pandas as pd
import pytest

from fabric_defect.cli.main import main
from fabric_defect.core.config import PipelineConfig


def test_default_toml_matches_dataclass_defaults():
    from pathlib import Path
    assert PipelineConfig.load(Path(__file__).parents[2] / "configs" / "default.toml") == PipelineConfig()


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[model]\nbogus = 1\n")
    with pytest.raises(TypeError):
        PipelineConfig.load(bad)
    bad.write_text("not_a_section = 1\n")
    with pytest.raises(ValueError):
        PipelineConfig.load(bad)


def test_cli_missing_dataset_exits_2(tmp_path):
    assert main(["run", "--data-dir", str(tmp_path / "none"), "--out", str(tmp_path / "out")]) == 2


def test_synthetic_end_to_end(tmp_path):
    cfg = tmp_path / "fast.toml"
    cfg.write_text("[patch]\nstride = 256\n")  # non-overlapping windows keep the test quick
    out = tmp_path / "out"
    assert main(["run", "--synthetic", "--out", str(out), "--config", str(cfg), "--max-test", "7"]) == 0

    metrics = pd.read_csv(out / "metrics_summary.csv")
    assert len(metrics) == 7 and metrics["patch_accuracy"].between(0, 1).all()
    assert (out / "fabric_summary.csv").exists()
    assert (out / "figures" / "benchmark_performance_charts.png").exists()

    info = json.loads((out / "run_config.json").read_text())
    assert info["synthetic"] is True and info["config"]["patch"]["stride"] == 256
