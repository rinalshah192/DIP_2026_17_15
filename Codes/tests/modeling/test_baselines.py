import numpy as np
import pytest

from fabric_defect.core.config import ModelConfig
from fabric_defect.modeling.baselines import OneClassBaseline


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(300, 6)) * [1, 10, 100, 1, 1, 1], rng


@pytest.mark.parametrize("scoring", ["ocsvm", "iforest"])
def test_outliers_score_higher_and_cross_tau(data, scoring):
    X, rng = data
    m = OneClassBaseline("f", ModelConfig(scoring=scoring), seed=0).fit(X[:200], X[200:])
    assert m.calibrated_out_of_sample
    normal = m.score_batch(rng.normal(size=(50, 6)) * [1, 10, 100, 1, 1, 1])
    X_out = np.full((5, 6), 8.0) * [1, 10, 100, 1, 1, 1]
    outlier = m.score_batch(X_out)
    assert outlier.min() > np.median(normal)
    assert (outlier >= m.adaptive_threshold).all()
    scores, tau = m.score_strip(X_out)
    assert tau == m.adaptive_threshold and scores.shape == (5,)


def test_iforest_is_seeded(data):
    X, _ = data
    a = OneClassBaseline("f", ModelConfig(scoring="iforest"), seed=3).fit(X[:200], X[200:])
    b = OneClassBaseline("f", ModelConfig(scoring="iforest"), seed=3).fit(X[:200], X[200:])
    assert a.adaptive_threshold == b.adaptive_threshold


def test_rejects_non_baseline_scoring():
    with pytest.raises(ValueError):
        OneClassBaseline("f", ModelConfig(scoring="hybrid"))
