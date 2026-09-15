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

"""Tests for OpenRouter transport retry behavior."""

import copy

import pytest
import requests

from proeval.evaluator import OpenRouterClient


class _Response:
    def __init__(self, status_code, content="ok"):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _install_responses(monkeypatch, responses):
    queued = iter(responses)
    payloads = []

    def fake_post(url, **kwargs):
        payloads.append(copy.deepcopy(kwargs["json"]))
        return next(queued)

    sleeps = []
    monkeypatch.setattr("proeval.evaluator.client.requests.post", fake_post)
    monkeypatch.setattr("proeval.evaluator.client.time.sleep", sleeps.append)
    return payloads, sleeps


def test_rate_limit_retries_original_structured_output_request(monkeypatch):
    payloads, sleeps = _install_responses(
        monkeypatch, [_Response(429), _Response(200, content="done")]
    )
    response_format = {"type": "json_schema", "json_schema": {"name": "answer"}}
    client = OpenRouterClient(api_key="test-key")

    result = client.predict(
        "prompt",
        response_format=response_format,
        max_retries=2,
        retry_delay=0.25,
    )

    assert result == "done"
    assert [payload["response_format"] for payload in payloads] == [
        response_format,
        response_format,
    ]
    assert sleeps == [1.0]


def test_final_rate_limit_attempt_does_not_sleep_or_fallback(monkeypatch):
    payloads, sleeps = _install_responses(monkeypatch, [_Response(429)])
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(requests.HTTPError, match="429"):
        client.predict(
            "prompt",
            response_format={"type": "json_object"},
            max_retries=1,
        )

    assert len(payloads) == 1
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert sleeps == []


def test_bad_request_cascades_structured_output_fallback(monkeypatch):
    payloads, sleeps = _install_responses(
        monkeypatch, [_Response(400), _Response(400), _Response(200)]
    )
    client = OpenRouterClient(api_key="test-key")

    assert client.predict(
        "prompt",
        response_format={"type": "json_schema", "json_schema": {}},
        max_retries=1,
    ) == "ok"

    assert payloads[0]["response_format"]["type"] == "json_schema"
    assert payloads[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in payloads[2]
    assert sleeps == []
