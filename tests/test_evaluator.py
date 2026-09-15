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

"""Tests for backend-neutral LLM evaluation."""

import math

import pytest

from proeval import Dataset, LLMPredictor, PredictionClient
from proeval.evaluator import DATASET_CONFIGS


class _StubClient:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def predict(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


def test_custom_client_does_not_require_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = _StubClient(['{"reasoning": "clear day", "answer": "yes"}'])

    predictor = LLMPredictor(model="local-agent", client=client)
    raw, prediction, score = predictor.evaluate(
        "Is the sky blue?", True, DATASET_CONFIGS["strategyqa"]
    )

    assert isinstance(client, PredictionClient)
    assert raw == '{"reasoning": "clear day", "answer": "yes"}'
    assert prediction == "yes"
    assert score == 0.0
    assert client.calls[0][1]["model"] == "local-agent"


def test_custom_client_and_openrouter_key_are_mutually_exclusive():
    with pytest.raises(ValueError, match="api_key cannot be used"):
        LLMPredictor(client=_StubClient(), api_key="unused")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("skip_error", [False, True])
def test_batch_parse_failure_contract(parallel, skip_error):
    client = _StubClient(["not json"])
    predictor = LLMPredictor(model="local-agent", client=client)
    kwargs = {
        "questions": ["question"],
        "ground_truths": [True],
        "dataset_config": DATASET_CONFIGS["strategyqa"],
        "max_parse_retries": 1,
        "show_progress": False,
        "skip_error": skip_error,
    }

    if parallel:
        result = predictor.predict_batch_parallel(max_workers=1, **kwargs)[0]
    else:
        result = predictor.predict_batch(**kwargs)[0]

    assert result[3] == "PARSE_ERROR"
    if skip_error:
        assert math.isnan(result[4])
    else:
        assert result[4] == 1.0


def test_dataset_forwards_sequential_retry_and_error_options():
    client = _StubClient(["not json"])
    predictor = LLMPredictor(model="local-agent", client=client)
    dataset = Dataset.from_lists(
        name="yes_no",
        questions=["question"],
        ground_truths=[True],
        config=DATASET_CONFIGS["strategyqa"],
    )

    result = dataset.predict(
        predictor,
        parallel=False,
        max_parse_retries=1,
        show_progress=False,
        skip_error=True,
    )[0]

    assert len(client.calls) == 1
    assert result[3] == "PARSE_ERROR"
    assert math.isnan(result[4])


@pytest.mark.parametrize("method_name", ["predict_batch", "predict_batch_parallel"])
def test_batch_rejects_mismatched_input_lengths(method_name):
    predictor = LLMPredictor(model="local-agent", client=_StubClient())
    method = getattr(predictor, method_name)

    with pytest.raises(ValueError, match="same length"):
        method(
            ["one", "two"],
            [True],
            DATASET_CONFIGS["strategyqa"],
            show_progress=False,
        )


def test_inference_client_errors_propagate_from_single_evaluation():
    predictor = LLMPredictor(
        model="local-agent",
        client=_StubClient(error=RuntimeError("backend unavailable")),
    )

    with pytest.raises(RuntimeError, match="backend unavailable"):
        predictor.evaluate(
            "question",
            True,
            DATASET_CONFIGS["strategyqa"],
            max_parse_retries=1,
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_normalizes_inference_client_errors(parallel):
    client = _StubClient(error=RuntimeError("429 rate limit"))
    predictor = LLMPredictor(model="local-agent", client=client)
    kwargs = {
        "questions": ["question"],
        "ground_truths": [True],
        "dataset_config": DATASET_CONFIGS["strategyqa"],
        "max_parse_retries": 1,
        "show_progress": False,
    }

    if parallel:
        result = predictor.predict_batch_parallel(max_workers=1, **kwargs)[0]
    else:
        result = predictor.predict_batch(**kwargs)[0]

    assert result[2] == "429 rate limit"
    assert result[3] == "RATE_LIMITED"
    assert result[4] == 1.0
    assert len(client.calls) == 1
