import numpy as np
import pytest

from fabric_defect.modeling.confidence import ConformalConfidence, image_score


def test_image_score_is_max_over_tau():
    assert image_score(np.array([1.0, 6.0, 3.0]), 3.0) == 2.0
    assert np.isnan(image_score(np.array([]), 3.0))


def test_conformal_p_values():
    c = ConformalConfidence([1, 2, 3, 4])
    assert [c.p_value(s) for s in (0.5, 2, 4, 5)] == [1.0, 0.8, 0.4, 0.2]
    assert c.confidence(5) == pytest.approx(0.8)


def test_false_alarm_rate_is_bounded_by_alpha():
    # Exchangeable defect-free scores: the share flagged at p <= alpha stays near or below alpha
    rng = np.random.default_rng(0)
    c = ConformalConfidence(rng.normal(size=199))
    flagged = np.mean([c.p_value(s) <= 0.05 for s in rng.normal(size=5000)])
    assert flagged <= 0.06


def test_needs_calibration_data():
    with pytest.raises(ValueError):
        ConformalConfidence([float("nan")])
