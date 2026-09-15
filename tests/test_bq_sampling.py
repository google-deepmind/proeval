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

"""Regression tests for Bayesian Quadrature sampling-budget semantics."""

import numpy as np
import pytest

from proeval.sampler.bq import (
    _bq_active_sampling,
    _bq_encoder_random_sampling,
    _bq_encoder_sampling,
    _bq_matern_active_sampling,
    _bq_matern_random_sampling,
    _bq_random_sampling,
    _get_posterior,
)


def _synthetic_case():
    rng = np.random.default_rng(0)
    n_samples = 12
    features = rng.normal(size=(4, n_samples))
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=float)
    prior_mean = np.linspace(0.25, 0.55, n_samples)
    prior_cov = np.eye(n_samples)
    return features, labels, prior_mean, prior_cov


class _DummyLinearEncoder:
    kernel_type = "linear"

    def __init__(self, torch):
        self._anchor = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self._anchor


def _run_active(kind, budget, n_init=0):
    features, labels, prior_mean, prior_cov = _synthetic_case()
    np.random.seed(7)

    if kind == "sf":
        return _bq_active_sampling(
            features,
            labels,
            prior_mean,
            prior_cov,
            budget=budget,
            n_init=n_init,
        )
    if kind == "rpf":
        return _bq_matern_active_sampling(
            features.T,
            labels,
            prior_mean,
            budget=budget,
            n_init=n_init,
        )

    torch = pytest.importorskip("torch")
    return _bq_encoder_sampling(
        features.T,
        labels,
        prior_mean,
        0.3,
        _DummyLinearEncoder(torch),
        budget=budget,
        n_init=n_init,
    )


def _run_random(kind, budget):
    features, labels, prior_mean, prior_cov = _synthetic_case()
    np.random.seed(7)

    if kind == "sf":
        return _bq_random_sampling(
            features,
            labels,
            prior_mean,
            prior_cov,
            budget=budget,
        )
    if kind == "rpf":
        return _bq_matern_random_sampling(
            features.T,
            labels,
            prior_mean,
            budget=budget,
        )

    torch = pytest.importorskip("torch")
    return _bq_encoder_random_sampling(
        features.T,
        labels,
        prior_mean,
        0.3,
        _DummyLinearEncoder(torch),
        budget=budget,
    )


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
@pytest.mark.parametrize("budget", [1, 5])
def test_active_sampling_honors_budget(kind, budget):
    result = _run_active(kind, budget)

    assert len(result.selected_indices) == budget
    assert len(set(result.selected_indices)) == budget
    assert len(result.estimates) == budget
    assert np.all(np.isfinite(result.estimates))
    assert np.all(np.isfinite(result.integral_variance))


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
def test_active_sampling_honors_budget_with_initial_samples(kind):
    result = _run_active(kind, budget=5, n_init=2)

    assert len(result.selected_indices) == 5
    assert len(set(result.selected_indices)) == 5
    assert len(result.estimates) == 5
    assert np.all(np.isfinite(result.estimates))


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
def test_active_and_random_sampling_use_same_number_of_evaluations(kind):
    active = _run_active(kind, budget=5)
    random = _run_random(kind, budget=5)

    assert len(active.selected_indices) == 5
    assert len(random.selected_indices) == 5


def test_active_estimates_are_post_sample_posteriors():
    features, labels, prior_mean, prior_cov = _synthetic_case()
    np.random.seed(7)
    result = _bq_active_sampling(
        features,
        labels,
        prior_mean,
        prior_cov,
        budget=5,
    )

    first_idx = result.selected_indices[:1]
    first_posterior, _ = _get_posterior(
        features[:, first_idx],
        labels[first_idx],
        features,
        0.3,
        first_idx,
        prior_mean,
    )
    final_posterior, _ = _get_posterior(
        features[:, result.selected_indices],
        labels[result.selected_indices],
        features,
        0.3,
        result.selected_indices,
        prior_mean,
    )

    assert result.estimates[0] == pytest.approx(np.mean(first_posterior))
    assert result.estimates[-1] == pytest.approx(np.mean(final_posterior))
    np.testing.assert_allclose(result.posterior_mean, final_posterior)


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
def test_active_sampling_rejects_n_init_above_budget(kind):
    with pytest.raises(ValueError, match="0 <= n_init <= budget <= n_samples"):
        _run_active(kind, budget=2, n_init=3)


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
def test_active_sampling_rejects_budget_above_n_samples(kind):
    n_samples = len(_synthetic_case()[1])
    with pytest.raises(ValueError, match="0 <= n_init <= budget <= n_samples"):
        _run_active(kind, budget=n_samples + 1)


@pytest.mark.parametrize("kind", ["sf", "rpf", "tpf"])
@pytest.mark.parametrize(("budget", "n_init"), [(0, 0), (12, 12)])
def test_active_sampling_accepts_budget_boundaries(kind, budget, n_init):
    result = _run_active(kind, budget=budget, n_init=n_init)

    assert len(result.selected_indices) == budget
    assert len(set(result.selected_indices)) == budget
    assert len(result.estimates) == budget
