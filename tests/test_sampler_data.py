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

"""Tests for sampler prediction-data normalization."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proeval.sampler.baselines import extract_model_predictions as extract_legacy
from proeval.sampler.data import extract_model_predictions

_ENCODER_DATA_PATH = Path(__file__).parents[1] / "proeval" / "encoder" / "data.py"
_ENCODER_DATA_SPEC = importlib.util.spec_from_file_location(
    "proeval_encoder_data", _ENCODER_DATA_PATH
)
assert _ENCODER_DATA_SPEC is not None
assert _ENCODER_DATA_SPEC.loader is not None
_ENCODER_DATA = importlib.util.module_from_spec(_ENCODER_DATA_SPEC)
_ENCODER_DATA_SPEC.loader.exec_module(_ENCODER_DATA)
load_benchmark_data = _ENCODER_DATA.load_benchmark_data


@pytest.mark.parametrize("extractor", [extract_model_predictions, extract_legacy])
@pytest.mark.parametrize("dataset_name", ["dices", "dices_t2i"])
def test_dices_error_scores_are_binarized_as_failures(extractor, dataset_name):
    frame = pd.DataFrame({"label_model": [0.0, 0.25, 0.5, 1.0]})

    predictions, model_names = extractor(frame, dataset_name)

    assert model_names == ["model"]
    np.testing.assert_array_equal(predictions[:, 0], [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("extractor", [extract_model_predictions, extract_legacy])
def test_non_dices_error_scores_are_unchanged(extractor):
    raw_scores = np.array([0.0, 0.25, 0.5, 1.0])
    frame = pd.DataFrame({"label_model": raw_scores})

    predictions, _ = extractor(frame, "custom")

    np.testing.assert_array_equal(predictions[:, 0], raw_scores)


@pytest.mark.parametrize("dataset_name", ["dices", "dices_t2i"])
def test_encoder_dices_error_scores_are_binarized_as_failures(tmp_path, dataset_name):
    frame = pd.DataFrame({"label_model": [0.0, 0.25, 0.5, 1.0]})
    frame.to_csv(tmp_path / f"{dataset_name}_predictions.csv", index=False)
    np.save(tmp_path / f"{dataset_name}_embeddings.npy", np.zeros((4, 2)))

    _, scores, model_names = load_benchmark_data(dataset_name, str(tmp_path))

    assert model_names == ["model"]
    np.testing.assert_array_equal(scores[:, 0], [0.0, 0.0, 1.0, 1.0])


def test_encoder_non_dices_error_scores_are_unchanged(tmp_path):
    raw_scores = np.array([0.0, 0.25, 0.5, 1.0])
    frame = pd.DataFrame({"label_model": raw_scores})
    frame.to_csv(tmp_path / "custom_predictions.csv", index=False)
    np.save(tmp_path / "custom_embeddings.npy", np.zeros((4, 2)))

    _, scores, _ = load_benchmark_data("custom", str(tmp_path))

    np.testing.assert_array_equal(scores[:, 0], raw_scores)
