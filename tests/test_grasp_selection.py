import numpy as np

from rars01_graspnet.grasp_selection import (
    joint_limit_cost,
    motion_cost,
    normalized_grasp_costs,
)


def test_grasp_scores_are_normalized_before_costing():
    assert np.allclose(normalized_grasp_costs(np.array([4.0, 2.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(normalized_grasp_costs(np.array([4.0, 4.0])), [0.5, 0.5])


def test_joint_and_motion_costs_use_joint_ranges():
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    chain = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])

    assert joint_limit_cost(chain, lower, upper) == 1.0
    assert np.isclose(motion_cost(chain, lower, upper, normalization=2.0), np.sqrt(0.125))


def test_cost_order_matches_exhaustive_validation_with_ties_and_rejections():
    from types import SimpleNamespace
    from rars01_graspnet.grasp_selection import best_validated_candidate
    rng = np.random.default_rng(20261024)
    for _ in range(500):
        candidates = [SimpleNamespace(cost=float(rng.integers(0, 6)), valid=bool(rng.integers(0, 2)))
                      for _ in range(int(rng.integers(0, 30)))]
        expected = min((c for c in candidates if c.valid), key=lambda c: c.cost, default=None)
        seen = []; metrics = {}
        def validate(candidate):
            seen.append(candidate)
            return candidate.valid
        actual = best_validated_candidate(candidates, validate, metrics=metrics)
        assert actual is expected
        assert metrics['checked_count'] == len(seen)
        assert [c.cost for c in seen] == sorted(c.cost for c in seen)
        if actual is not None:
            assert seen[-1] is actual
            assert all(not c.valid for c in seen[:-1])


def test_validation_errors_and_cancel_are_not_candidate_rejection():
    import pytest
    from types import SimpleNamespace
    from rars01_graspnet.grasp_selection import best_validated_candidate
    candidates = [SimpleNamespace(cost=0.), SimpleNamespace(cost=1.)]
    def fail(*args):
        raise RuntimeError('COLLISION_CANCELLED:SHUTDOWN')
    with pytest.raises(RuntimeError, match='CANCELLED'):
        best_validated_candidate(candidates, fail)
    with pytest.raises(RuntimeError, match='CANCELLED'):
        best_validated_candidate(candidates, lambda c: True, cancel_check=fail)
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            best_validated_candidate([SimpleNamespace(cost=value)], lambda c: True)
