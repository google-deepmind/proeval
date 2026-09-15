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

"""Check linear BQ acquisition against conditioning the full GP covariance."""

import numpy as np
import pandas as pd
import pytest

from proeval.sampler.baselines import variance_improvement
from proeval.sampler.bq import BQPriorSampler, _variance_improvement
from proeval.sampler.data import setup_train_test_split


@pytest.mark.parametrize("acquire", [_variance_improvement, variance_improvement])
@pytest.mark.parametrize("noise_variance", [0.1, 0.3, 1.0, 3.0])
def test_linear_acquisition_matches_full_gp(acquire, noise_variance):
    features = np.random.default_rng(0).normal(size=(3, 8))
    prior_covariance = features.T @ features
    labeled = []
    unlabeled = list(range(features.shape[1]))
    cached_inverse = None

    for _ in range(5):
        # Compute the posterior in sample space, independently of the cached
        # feature-space inverse used by the fast acquisition implementation.
        cross_covariance = prior_covariance[:, labeled]
        observation_covariance = prior_covariance[np.ix_(labeled, labeled)]
        observation_covariance += noise_variance * np.eye(len(labeled))
        posterior_covariance = prior_covariance - cross_covariance @ np.linalg.solve(
            observation_covariance, cross_covariance.T
        )
        conditional_variances = []
        for candidate in unlabeled:
            column = posterior_covariance[:, candidate]
            conditioned = posterior_covariance - np.outer(column, column) / (
                posterior_covariance[candidate, candidate] + noise_variance
            )
            conditional_variances.append(np.mean(conditioned))
        expected = unlabeled[int(np.argmin(conditional_variances))]

        selected, cached_inverse = acquire(
            features[:, labeled], cached_inverse, noise_variance,
            unlabeled, features,
        )
        assert selected == expected
        labeled.append(selected)
        unlabeled.remove(selected)

        # The returned state must describe conditioning on every selected point
        # with the same observation noise as the GP posterior.
        selected_features = features[:, labeled]
        expected_inverse = np.linalg.inv(
            np.eye(features.shape[0])
            + selected_features @ selected_features.T / noise_variance
        )
        np.testing.assert_allclose(cached_inverse, expected_inverse, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("noise_variance", [0.3, 3.0])
def test_prior_sampler_minimizes_integral_variance(noise_variance):
    predictions = np.random.default_rng(0).integers(0, 2, size=(12, 5))
    frame = pd.DataFrame(predictions, columns=[f"label_model_{i}" for i in range(5)])
    result = BQPriorSampler(noise_variance=noise_variance).sample(
        frame, target_model=0, pretrain_mode="all", budget=5,
    )
    _, features, _, _, _ = setup_train_test_split(predictions, target_model=0)
    prior_covariance = features.T @ features
    labeled = []
    unlabeled = list(range(len(frame)))

    for selected in result.selected_indices:
        cross_covariance = prior_covariance[:, labeled]
        observation_covariance = prior_covariance[np.ix_(labeled, labeled)]
        observation_covariance += noise_variance * np.eye(len(labeled))
        posterior_covariance = prior_covariance - cross_covariance @ np.linalg.solve(
            observation_covariance, cross_covariance.T
        )
        columns = posterior_covariance[:, unlabeled]
        reductions = np.sum(columns, axis=0) ** 2 / (
            np.diag(posterior_covariance)[unlabeled] + noise_variance
        )
        # Score features can tie, so compare objective values instead of indices.
        assert reductions[unlabeled.index(selected)] == pytest.approx(max(reductions))
        labeled.append(selected)
        unlabeled.remove(selected)
