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

from proeval import Dataset, LLMPredictor, PredictionClientError
from proeval.evaluator import DATASET_CONFIGS, DatasetConfig, UnifiedCSVManager


class _StubClient:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def predict(self, prompt, *, model, max_tokens, response_format):
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
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

    assert raw == '{"reasoning": "clear day", "answer": "yes"}'
    assert prediction == "yes"
    assert score == 0.0
    assert client.calls[0]["model"] == "local-agent"
    assert client.calls[0]["max_tokens"] == 8192
    assert client.calls[0]["response_format"] == DATASET_CONFIGS["strategyqa"].json_schema


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

    with pytest.raises(PredictionClientError, match="backend unavailable") as exc_info:
        predictor.evaluate(
            "question",
            True,
            DATASET_CONFIGS["strategyqa"],
            max_parse_retries=1,
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)


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


@pytest.mark.parametrize("method_name", ["predict_batch", "predict_batch_parallel"])
def test_batch_rejects_invalid_parse_retry_count(method_name):
    predictor = LLMPredictor(model="local-agent", client=_StubClient())
    method = getattr(predictor, method_name)

    with pytest.raises(ValueError, match="positive integer"):
        method(
            ["question"],
            [True],
            DATASET_CONFIGS["strategyqa"],
            max_parse_retries=0,
            show_progress=False,
        )


@pytest.mark.parametrize("max_workers", [0, 1.5, True])
def test_parallel_batch_rejects_invalid_worker_count(max_workers):
    predictor = LLMPredictor(model="local-agent", client=_StubClient())

    with pytest.raises(ValueError, match="positive integer"):
        predictor.predict_batch_parallel(
            ["question"],
            [True],
            DATASET_CONFIGS["strategyqa"],
            max_workers=max_workers,
            show_progress=False,
        )


def test_batch_does_not_normalize_configuration_errors():
    def broken_prompt(_question):
        raise ValueError("bad prompt config")

    config = DatasetConfig(
        name="broken",
        prompt_template=broken_prompt,
        json_schema={"type": "json_object"},
        extract_prediction=lambda data: data["answer"],
        extract_ground_truth=lambda value: value,
        compare_predictions=lambda prediction, truth: 0.0,
    )
    predictor = LLMPredictor(model="local-agent", client=_StubClient())

    with pytest.raises(ValueError, match="bad prompt config"):
        predictor.predict_batch(
            ["question"], [True], config, show_progress=False
        )


def test_prediction_client_must_return_text():
    predictor = LLMPredictor(
        model="local-agent",
        client=_StubClient(responses=[{"answer": "yes"}]),
    )

    with pytest.raises(PredictionClientError, match="must return a string"):
        predictor.evaluate(
            "question",
            True,
            DATASET_CONFIGS["strategyqa"],
            max_parse_retries=1,
        )


def test_parallel_batch_preserves_input_order_with_shared_client():
    class _ThreadSafeClient:
        def predict(self, prompt, *, model, max_tokens, response_format):
            return '{"reasoning": "ok", "answer": "yes"}'

    questions = [f"question-{index}" for index in range(8)]
    predictor = LLMPredictor(model="local-agent", client=_ThreadSafeClient())

    results = predictor.predict_batch_parallel(
        questions,
        [True] * len(questions),
        DATASET_CONFIGS["strategyqa"],
        max_workers=3,
        show_progress=False,
    )

    assert [result[0] for result in results] == questions


@pytest.mark.parametrize("parallel", [False, True])
def test_csv_manager_normalizes_backend_errors_in_both_modes(tmp_path, parallel):
    manager = UnifiedCSVManager("custom", output_dir=str(tmp_path))
    manager.load_or_create(["question"], [True])
    predictor = LLMPredictor(
        model="local-agent",
        client=_StubClient(error=RuntimeError("backend unavailable")),
    )

    manager.run_evaluation(
        predictor,
        model_name="candidate",
        dataset_config=DATASET_CONFIGS["strategyqa"],
        questions=["question"],
        ground_truths=[True],
        parallel=parallel,
        workers=1,
        max_parse_retries=1,
    )

    assert manager.df.loc[0, "prediction_candidate"] == "ERROR"
    assert manager.df.loc[0, "label_candidate"] == 1.0


def test_csv_manager_sequential_fix_uses_normalized_evaluation(tmp_path):
    manager = UnifiedCSVManager("custom", output_dir=str(tmp_path))
    manager.load_or_create(["question"], [True])
    failing = LLMPredictor(model="local-agent", client=_StubClient(["not json"]))
    manager.run_evaluation(
        failing,
        model_name="candidate",
        dataset_config=DATASET_CONFIGS["strategyqa"],
        questions=["question"],
        ground_truths=[True],
        parallel=False,
        max_parse_retries=1,
    )
    assert manager.df.loc[0, "prediction_candidate"] == "PARSE_ERROR"

    fixed = LLMPredictor(
        model="local-agent",
        client=_StubClient(['{"reasoning": "ok", "answer": "yes"}']),
    )
    manager.fix_errors(
        fixed,
        model_name="candidate",
        dataset_config=DATASET_CONFIGS["strategyqa"],
        questions=["question"],
        ground_truths=[True],
        parallel=False,
        max_parse_retries=1,
    )

    assert manager.df.loc[0, "prediction_candidate"] == "yes"
    assert manager.df.loc[0, "label_candidate"] == 0.0
