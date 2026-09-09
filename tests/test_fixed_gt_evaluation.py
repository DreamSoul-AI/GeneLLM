"""The cost objective never rewards missing necessary information."""

import pytest

from run_experiment.evaluate_group_lasso_fixed_gt import score_plan


@pytest.mark.parametrize("selected,exact,sufficient,missing,extra,excess", [
    ([0, 1], True, True, 0, 0, 0.),
    ([0, 1, 2], False, True, 0, 1, 35.),
    ([0], False, False, 1, 0, None),
    ([], False, False, 2, 0, None),
    ([0, 2], False, False, 1, 1, None),
])
def test_fixed_ground_truth_outcomes(selected, exact, sufficient, missing, extra, excess):
    row = score_plan(selected, [0, 1], [10., 20., 35.])
    assert row["exact_gt"] is exact
    assert row["information_sufficient"] is sufficient
    assert row["n_missing"] == missing and row["n_extra"] == extra
    assert row["excess_cost_if_sufficient"] == excess
    assert row["gt_cost"] == 30.


def test_zero_price_alternatives_require_a_generalized_optimal_solution_set():
    with pytest.raises(ValueError):
        score_plan([0, 1, 2], [0, 1], [10., 20., 0.])
