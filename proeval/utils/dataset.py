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

"""General-purpose Dataset class for ProEval evaluation.

A :class:`Dataset` bundles three things that travel together during evaluation:

1. ``questions`` — the inputs sent to the model
2. ``ground_truths`` — the reference answers used for scoring
3. ``config`` — a :class:`~proeval.evaluator.DatasetConfig` that defines
   the prompt template, JSON schema, and prediction/score extractors

Use one of the constructors to build a :class:`Dataset`:

- :meth:`Dataset.from_builtin` — one of the 9 datasets shipped with ProEval
  (``svamp``, ``gsm8k``, ``strategyqa``, ...).
- :meth:`Dataset.from_lists` — pass questions/ground_truths/eval functions
  directly. The simplest way to bring a custom dataset.
- :meth:`Dataset.from_csv` — load questions and ground truths from a CSV.

Run predictions with :meth:`Dataset.predict` (or
:meth:`~proeval.evaluator.LLMPredictor.predict_dataset`).

Example — built-in::

    from proeval.utils import Dataset
    from proeval.evaluator import LLMPredictor

    ds = Dataset.from_builtin("svamp")
    predictor = LLMPredictor(model="google/gemma-3-4b-it")
    results = ds.predict(predictor, parallel=True, workers=10)

Example — custom (from lists)::

    ds = Dataset.from_lists(
        name="my_yesno",
        questions=["Is the sky blue?", "Is fire cold?"],
        ground_truths=["yes", "no"],
        prompt_template=lambda q: f"{q} Respond JSON: {{'answer': 'yes'|'no'}}",
        extract_prediction=lambda d: d["answer"],
        extract_ground_truth=lambda gt: str(gt).lower(),
        compare_predictions=lambda p, g: 0.0 if str(p).lower() == g else 1.0,
    )
    results = ds.predict(predictor)

Example — reuse a built-in config with custom data::

    from proeval.evaluator import DATASET_CONFIGS
    ds = Dataset.from_lists(
        name="my_strategyqa",
        questions=[...],
        ground_truths=[...],
        config=DATASET_CONFIGS["strategyqa"],
    )
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from proeval.evaluator.predictor import DATASET_CONFIGS, DatasetConfig


_DEFAULT_JSON_SCHEMA: Dict[str, Any] = {"type": "json_object"}


class Dataset:
    """A bundle of (questions, ground_truths, DatasetConfig) for evaluation.

    Attributes:
        name: Friendly dataset name (e.g. ``"svamp"``, ``"my_qa"``). Used as
            the default file/column prefix downstream.
        questions: List of inputs to send to the model. Items may be strings
            or dicts depending on what the prompt template expects.
        ground_truths: List of reference answers aligned 1:1 with *questions*.
        config: :class:`DatasetConfig` defining the prompt template, JSON
            schema, and extraction/comparison functions.
    """

    def __init__(
        self,
        name: str,
        questions: List[Any],
        ground_truths: List[Any],
        config: DatasetConfig,
    ):
        if len(questions) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(questions)} questions vs "
                f"{len(ground_truths)} ground_truths"
            )
        self.name = name
        self.questions = list(questions)
        self.ground_truths = list(ground_truths)
        self.config = config

    # Container protocol — supports len(), indexing, iteration. This also
    # gives the future sampler a uniform interface to operate on.

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        return self.questions[idx], self.ground_truths[idx]

    def __iter__(self):
        return iter(zip(self.questions, self.ground_truths))

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, size={len(self)})"

    # Constructors

    @classmethod
    def from_builtin(cls, name: str) -> "Dataset":
        """Load one of the datasets shipped with ProEval.

        Supported names: ``strategyqa``, ``gsm8k``, ``svamp``, ``mmlu``,
        ``mmlu_professionallaw``, ``jigsaw``, ``toxicchat``, ``gqa``,
        ``dices``, ``dices_t2i``.

        Requires the ``[datasets]`` extra for HuggingFace-backed loaders.
        """
        if name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown built-in dataset {name!r}. Available: "
                f"{sorted(DATASET_CONFIGS.keys())}"
            )
        # Import here to avoid pulling in pandas/datasets at module import time.
        from proeval.evaluator.csv_manager import load_dataset_data

        questions, ground_truths = load_dataset_data(name)
        return cls(
            name=name,
            questions=questions,
            ground_truths=ground_truths,
            config=DATASET_CONFIGS[name],
        )

    @classmethod
    def from_lists(
        cls,
        name: str,
        questions: List[Any],
        ground_truths: List[Any],
        config: Optional[DatasetConfig] = None,
        *,
        prompt_template: Optional[Callable[[Any], str]] = None,
        extract_prediction: Optional[Callable[[Dict[str, Any]], Any]] = None,
        extract_ground_truth: Optional[Callable[[Any], Any]] = None,
        compare_predictions: Optional[Callable[[Any, Any], float]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> "Dataset":
        """Build a Dataset from in-memory lists.

        Either pass a pre-built *config*, or pass the four eval functions
        (``prompt_template``, ``extract_prediction``, ``extract_ground_truth``,
        ``compare_predictions``) and a :class:`DatasetConfig` will be created
        internally. ``json_schema`` defaults to ``{"type": "json_object"}``.
        """
        cfg = _resolve_config(
            name=name,
            config=config,
            prompt_template=prompt_template,
            extract_prediction=extract_prediction,
            extract_ground_truth=extract_ground_truth,
            compare_predictions=compare_predictions,
            json_schema=json_schema,
        )
        return cls(name=name, questions=questions, ground_truths=ground_truths, config=cfg)

    @classmethod
    def from_csv(
        cls,
        path: str,
        question_col: str = "question",
        ground_truth_col: str = "ground_truth",
        config: Optional[DatasetConfig] = None,
        *,
        name: Optional[str] = None,
        prompt_template: Optional[Callable[[Any], str]] = None,
        extract_prediction: Optional[Callable[[Dict[str, Any]], Any]] = None,
        extract_ground_truth: Optional[Callable[[Any], Any]] = None,
        compare_predictions: Optional[Callable[[Any, Any], float]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> "Dataset":
        """Build a Dataset from a CSV file.

        Args:
            path: Path to the CSV file.
            question_col: Column name holding the input questions.
            ground_truth_col: Column name holding the reference answers.
            config: Pre-built :class:`DatasetConfig`, or pass the four eval
                functions inline (same as :meth:`from_lists`).
            name: Friendly dataset name. Defaults to the CSV file basename.

        Either *config* or the four eval functions must be provided.
        """
        import os
        import pandas as pd  # local import — pandas is already a project dep

        df = pd.read_csv(path)
        for col in (question_col, ground_truth_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {path}. "
                    f"Available columns: {list(df.columns)}"
                )

        resolved_name = name or os.path.splitext(os.path.basename(path))[0]
        cfg = _resolve_config(
            name=resolved_name,
            config=config,
            prompt_template=prompt_template,
            extract_prediction=extract_prediction,
            extract_ground_truth=extract_ground_truth,
            compare_predictions=compare_predictions,
            json_schema=json_schema,
        )
        return cls(
            name=resolved_name,
            questions=df[question_col].tolist(),
            ground_truths=df[ground_truth_col].tolist(),
            config=cfg,
        )

    # Prediction

    def predict(
        self,
        predictor,
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 3,
        show_progress: bool = True,
        skip_error: bool = False,
    ) -> List[Tuple[Any, Any, str, Any, float]]:
        """Run *predictor* over every example in this dataset.

        Args:
            predictor: An :class:`~proeval.evaluator.LLMPredictor` instance.
            parallel: If ``True`` (default), use
                :meth:`~proeval.evaluator.LLMPredictor.predict_batch_parallel`;
                otherwise use the sequential
                :meth:`~proeval.evaluator.LLMPredictor.predict_batch`.
            workers: Thread count for parallel mode.
            max_parse_retries: Retries per item.
            show_progress: Show a tqdm progress bar.
            skip_error: ``True``: mark parse errors as NaN (excluded from
                accuracy). ``False``: mark as 1.0 (counted as failure).

        Returns:
            List of ``(question, ground_truth, raw_response, prediction,
            score)`` tuples — the same shape as
            :meth:`~proeval.evaluator.LLMPredictor.predict_batch_parallel`.
        """
        if parallel:
            return predictor.predict_batch_parallel(
                self.questions,
                self.ground_truths,
                self.config,
                max_workers=workers,
                max_parse_retries=max_parse_retries,
                show_progress=show_progress,
                skip_error=skip_error,
            )
        return predictor.predict_batch(
            self.questions,
            self.ground_truths,
            self.config,
            show_progress=show_progress,
        )


# Internal helpers


def _resolve_config(
    *,
    name: str,
    config: Optional[DatasetConfig],
    prompt_template: Optional[Callable[[Any], str]],
    extract_prediction: Optional[Callable[[Dict[str, Any]], Any]],
    extract_ground_truth: Optional[Callable[[Any], Any]],
    compare_predictions: Optional[Callable[[Any, Any], float]],
    json_schema: Optional[Dict[str, Any]],
) -> DatasetConfig:
    """Return *config* if given, else build one from the inline functions."""
    if config is not None:
        return config

    missing = [
        n for n, v in [
            ("prompt_template", prompt_template),
            ("extract_prediction", extract_prediction),
            ("extract_ground_truth", extract_ground_truth),
            ("compare_predictions", compare_predictions),
        ]
        if v is None
    ]
    if missing:
        raise ValueError(
            f"Must pass either `config` or all of: prompt_template, "
            f"extract_prediction, extract_ground_truth, compare_predictions. "
            f"Missing: {missing}"
        )
    return DatasetConfig(
        name=name,
        prompt_template=prompt_template,
        json_schema=json_schema if json_schema is not None else _DEFAULT_JSON_SCHEMA,
        extract_prediction=extract_prediction,
        extract_ground_truth=extract_ground_truth,
        compare_predictions=compare_predictions,
    )
