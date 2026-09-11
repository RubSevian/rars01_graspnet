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
