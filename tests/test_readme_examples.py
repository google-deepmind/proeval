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

"""No-network smoke tests for the public examples in the README files."""

import numpy as np
import pandas as pd


def test_documented_public_imports():
    """The core and optional-class import paths shown in the guide stay valid."""
    from proeval import BQPriorSampler, Dataset, LLMPredictor, TopicAwareGenerator
    from proeval.sampler import BQEncoderSampler

    assert BQPriorSampler is not None
    assert BQEncoderSampler is not None
    assert Dataset is not None
    assert LLMPredictor is not None
    assert TopicAwareGenerator is not None


def test_in_memory_quickstart():
    """The root README quickstart runs without data files or network access."""
    from proeval import BQPriorSampler

    predictions = pd.DataFrame(
        {
            "label_reference_a": [0, 0, 1, 0, 1, 0],
            "label_reference_b": [0, 1, 1, 0, 0, 0],
            "label_candidate": [0, 0, 1, 0, 1, 1],
        }
    )

    result = BQPriorSampler(noise_variance=0.3).sample(
        predictions=predictions,
        target_model="candidate",
        budget=3,
        pretrain_mode="all",
        seed=42,
    )

    assert len(result.selected_indices) == 3
    assert result.estimates.shape == (3,)
    assert np.isfinite(result.estimates).all()

