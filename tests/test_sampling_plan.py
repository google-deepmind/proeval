# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the target-free score-feature sampling plan API."""

import numpy as np
import pandas as pd
import pytest

from proeval import BQPriorSampler, SamplingPlan


@pytest.fixture
def score_data():
    source_scores = pd.DataFrame(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        index=[f"case-{i}" for i in range(8)],
        columns=["label_source_a", "label_source_b", "label_source_c"],
        dtype=float,
    )
    target_scores = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=float)
    return source_scores, target_scores


def test_plan_uses_dataframe_index_and_needs_no_target_scores(score_data):
    source_scores, _ = score_data

    plan = BQPriorSampler(n_init=2).plan(source_scores, budget=5, seed=7)

    assert isinstance(plan, SamplingPlan)
    assert isinstance(plan.indices, tuple)
    assert isinstance(plan.item_ids, tuple)
    assert len(plan.indices) == 5
    assert len(set(plan.indices)) == 5
    assert plan.item_ids == tuple(source_scores.index[index] for index in plan.indices)


def test_plan_alignment_is_immutable(score_data):
    source_scores, _ = score_data
    plan = BQPriorSampler().plan(source_scores, budget=3)

    with pytest.raises(AttributeError):
        plan.item_ids = tuple(reversed(plan.item_ids))
    with pytest.raises(AttributeError):
        plan.indices = tuple(reversed(plan.indices))


def test_plan_and_estimate_match_legacy_sample(score_data):
    source_scores, target_scores = score_data
    predictions = source_scores.assign(label_target=target_scores)
    sampler = BQPriorSampler(noise_variance=0.3, n_init=2)

    legacy = sampler.sample(
        predictions,
        target_model="target",
        budget=5,
        pretrain_mode="all",
        seed=7,
    )
    plan = sampler.plan(source_scores, budget=5, seed=7)
    planned = plan.estimate(target_scores[list(plan.indices)])

    assert list(plan.indices) == legacy.selected_indices
    assert planned.selected_indices == legacy.selected_indices
    np.testing.assert_allclose(planned.estimates, legacy.estimates)
    np.testing.assert_allclose(planned.rounded_estimates, legacy.rounded_estimates)
    np.testing.assert_allclose(planned.posterior_mean, legacy.posterior_mean)
    np.testing.assert_allclose(planned.posterior_var, legacy.posterior_var)
    np.testing.assert_allclose(planned.prior_mean, legacy.prior_mean)
    np.testing.assert_allclose(planned.integral_variance, legacy.integral_variance)


def test_estimate_aligns_mapping_scores_by_item_id(score_data):
    source_scores, target_scores = score_data
    plan = BQPriorSampler().plan(source_scores, budget=4)
    score_by_id = {
        source_scores.index[index]: target_scores[index]
        for index in reversed(plan.indices)
    }

    from_sequence = plan.estimate(target_scores[list(plan.indices)])
    from_mapping = plan.estimate(score_by_id)
    from_series = plan.estimate(pd.Series(score_by_id))

    np.testing.assert_allclose(from_mapping.estimates, from_sequence.estimates)
    np.testing.assert_allclose(from_mapping.posterior_mean, from_sequence.posterior_mean)
    np.testing.assert_allclose(from_series.estimates, from_sequence.estimates)


def test_plan_seed_does_not_mutate_global_numpy_random_state(score_data):
    source_scores, _ = score_data
    np.random.seed(11)
    expected_next_values = np.random.random(3)

    np.random.seed(11)
    BQPriorSampler(n_init=2).plan(source_scores, budget=5, seed=7)

    np.testing.assert_allclose(np.random.random(3), expected_next_values)


@pytest.mark.parametrize(
    ("source_scores", "message"),
    [
        (np.ones(4), "two-dimensional"),
        (np.ones((4, 1)), "at least two source models"),
        (np.array([[0.0, 1.0], [np.nan, 0.0]]), "finite"),
        (np.empty((0, 2)), "at least one item"),
    ],
)
def test_plan_validates_source_scores(source_scores, message):
    with pytest.raises(ValueError, match=message):
        BQPriorSampler().plan(source_scores, budget=1)


@pytest.mark.parametrize(
    ("item_ids", "message"),
    [
        (["a"], "one ID per"),
        (["a", "a"], "unique"),
        ([[], []], "hashable"),
        ([None, "b"], "missing"),
        ([np.nan, "b"], "missing"),
    ],
)
def test_plan_validates_item_ids(item_ids, message):
    with pytest.raises(ValueError, match=message):
        BQPriorSampler().plan(np.ones((2, 2)), budget=1, item_ids=item_ids)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([1.0], "expected 2"),
        ([[1.0], [0.0]], "one-dimensional"),
        ([1.0, np.inf], "finite"),
        ({"item-a": 1.0}, "missing item IDs"),
        ({"item-a": 1.0, "item-b": 0.0, "item-c": 1.0}, "unexpected item IDs"),
    ],
)
def test_estimate_validates_score_alignment(scores, message):
    plan = BQPriorSampler().plan(
        np.array([[0, 1], [1, 0], [0, 0]], dtype=float),
        budget=2,
        item_ids=["item-a", "item-b", "item-c"],
    )

    with pytest.raises(ValueError, match=message):
        plan.estimate(scores)


@pytest.mark.parametrize("budget", [0, 1.5, True])
def test_plan_requires_a_positive_integer_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        BQPriorSampler().plan(np.ones((3, 2)), budget=budget)


@pytest.mark.parametrize("n_init", [1.5, True])
def test_plan_requires_an_integer_initial_budget(n_init):
    with pytest.raises(ValueError, match="must be integers"):
        BQPriorSampler(n_init=n_init).plan(np.ones((3, 2)), budget=2)


def test_plan_accepts_full_budget(score_data):
    source_scores, _ = score_data

    plan = BQPriorSampler().plan(source_scores, budget=len(source_scores))

    assert len(plan.indices) == len(source_scores)
