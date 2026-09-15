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

"""Unified CSV manager for multi-model predictions with resume and fix-error.

Provides:

- :class:`UnifiedCSVManager` — Multi-model CSV storage with checkpoint resume
  and error-fixing capabilities.
- :func:`load_dataset_data` — HuggingFace/local dataset loading for all
  supported datasets.
- :func:`save_predictions_to_csv` / :func:`load_predictions_from_csv` — Simple
  single-model CSV I/O.

Example — full evaluation with resume::

    from proeval.evaluator import LLMPredictor, DATASET_CONFIGS
    from proeval.evaluator.csv_manager import UnifiedCSVManager

    csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
    csv_mgr.load_or_create(questions, ground_truths)

    predictor = LLMPredictor(model="google/gemma-3-27b-it")
    csv_mgr.run_evaluation(
        predictor, "gemma3_27b", DATASET_CONFIGS["gsm8k"],
        questions, ground_truths, parallel=True, workers=10,
    )

Example — fix errors in existing CSV::

    csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
    csv_mgr.load_or_create(questions, ground_truths)
    csv_mgr.fix_errors(predictor, "gemma3_27b", DATASET_CONFIGS["gsm8k"],
                       questions, ground_truths)
"""

import csv
import hashlib
import json
import os
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Numpy JSON serialisation helper


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(e) for e in obj]
        return type(obj)(converted)
    return obj


# UnifiedCSVManager


class UnifiedCSVManager:
    """Multi-model CSV storage with checkpoint resume and fix-error support.

    Each dataset has **one** CSV file with columns::

        index, question, ground_truth,
        prediction_<model>, label_<model>, raw_response_<model>, ...

    Args:
        dataset_name: e.g. ``"gsm8k"``
        output_dir: directory for CSV and checkpoint files
    """

    def __init__(self, dataset_name: str, output_dir: str = "."):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
        self.metadata_path = os.path.join(
            output_dir, f".{dataset_name}_dataset_metadata.json"
        )
        self.df: Optional[pd.DataFrame] = None
        self.dataset_fingerprint: Optional[str] = None

    # ── load / create ─────────────────────────────────────────────────

    def load_or_create(
        self, questions: List[Any], ground_truths: List[Any]
    ) -> pd.DataFrame:
        """Load existing CSV or create a new DataFrame.

        Returns the DataFrame (also stored as ``self.df``).
        """
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(
                self.csv_path,
                converters={"question": str, "ground_truth": str},
            )
            self._validate_dataset_alignment(questions, ground_truths)
            self._validate_dataset_metadata()
            # The converter read preserves exact CSV identity for validation,
            # but callers should see the same value types they supplied when
            # creating the manager, not CSV strings such as ``"False"``.
            self.df["question"] = pd.Series(
                questions, index=self.df.index, dtype=object
            )
            self.df["ground_truth"] = pd.Series(
                ground_truths, index=self.df.index, dtype=object
            )
            print(f"Loaded existing CSV: {self.csv_path} ({len(self.df)} rows)")
        else:
            if len(questions) != len(ground_truths):
                raise ValueError(
                    f"Length mismatch: {len(questions)} questions vs "
                    f"{len(ground_truths)} ground truths"
                )
            self.df = pd.DataFrame({"index": range(len(questions))})
            self.df["question"] = pd.Series(questions, dtype=object)
            self.df["ground_truth"] = pd.Series(ground_truths, dtype=object)
            self.dataset_fingerprint = self._fingerprint_dataset(
                questions, ground_truths
            )
            print(f"Created new CSV structure: {self.csv_path}")
        return self.df

    # ── model column helpers ──────────────────────────────────────────

    def has_model(self, model_name: str) -> bool:
        """Check if prediction + label columns exist for *model_name*."""
        if self.df is None:
            return False
        return (
            f"prediction_{model_name}" in self.df.columns
            and f"label_{model_name}" in self.df.columns
        )

    def add_model_predictions(
        self,
        model_name: str,
        predictions: List[Any],
        labels: List[float],
        raw_responses: Optional[List[str]] = None,
        rerun: bool = False,
    ) -> None:
        """Add or overwrite model prediction columns."""
        self._check_init()
        if self.has_model(model_name) and not rerun:
            print(f"Model '{model_name}' already evaluated. Use rerun=True to re-evaluate.")
            return
        self.df[f"prediction_{model_name}"] = predictions
        self.df[f"label_{model_name}"] = labels
        if raw_responses is not None:
            self.df[f"raw_response_{model_name}"] = raw_responses

    def get_model_accuracy(self, model_name: str) -> Optional[float]:
        """Return accuracy for *model_name* (``1 − mean(labels)``), or ``None``."""
        if not self.has_model(model_name):
            return None
        labels = self.df[f"label_{model_name}"]
        valid = labels.dropna()
        return float(1.0 - valid.mean()) if len(valid) > 0 else None

    # ── error detection & fixing ──────────────────────────────────────

    ERROR_SENTINELS = {"SKIPPED", "ERROR", "RATE_LIMITED", "PARSE_ERROR"}

    def get_error_indices(self, model_name: str) -> List[int]:
        """Return row indices with error predictions or NaN labels."""
        if not self.has_model(model_name):
            return []
        pred_col = f"prediction_{model_name}"
        label_col = f"label_{model_name}"
        mask = self.df[pred_col].isin(self.ERROR_SENTINELS) | self.df[label_col].isna()
        return self.df.index[mask].tolist()

    def update_predictions_at_indices(
        self,
        model_name: str,
        indices: List[int],
        predictions: List[Any],
        labels: List[float],
        raw_responses: Optional[List[str]] = None,
    ) -> None:
        """Update specific rows with new results."""
        self._check_init()
        for i, idx in enumerate(indices):
            self.df.at[idx, f"prediction_{model_name}"] = predictions[i]
            self.df.at[idx, f"label_{model_name}"] = labels[i]
            if raw_responses is not None:
                self.df.at[idx, f"raw_response_{model_name}"] = raw_responses[i]
        print(f"Updated {len(indices)} predictions for model: {model_name}")

    # ── save ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save DataFrame to CSV."""
        self._check_init()
        os.makedirs(self.output_dir, exist_ok=True)
        if self.dataset_fingerprint is None:
            self.dataset_fingerprint = self._fingerprint_dataset(
                self.df["question"].tolist(),
                self.df["ground_truth"].tolist(),
            )
        self.df.to_csv(self.csv_path, index=False)
        metadata = {
            "version": 1,
            "dataset_name": self.dataset_name,
            "rows": len(self.df),
            "fingerprint": self.dataset_fingerprint,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, sort_keys=True)
        print(f"Saved predictions to: {self.csv_path}")

    # ── high-level run & fix ──────────────────────────────────────────

    def run_evaluation(
        self,
        predictor,
        model_name: str,
        dataset_config,
        questions: List[Any],
        ground_truths: List[Any],
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 5,
        skip_error: bool = False,
        rerun: bool = False,
        checkpoint_interval: int = 50,
    ) -> None:
        """Run full evaluation with checkpoint resume support.

        Saves a ``.checkpoint_*.json`` file every *checkpoint_interval* items.
        If a checkpoint exists from a previous interrupted run, evaluation
        resumes from where it left off.

        Args:
            predictor: :class:`LLMPredictor` instance.
            model_name: Friendly name (used as column suffix).
            dataset_config: :class:`DatasetConfig`.
            questions: Full question list.
            ground_truths: Full ground-truth list.
            parallel: Use ``predict_batch_parallel`` (default True).
            workers: Thread count for parallel mode.
            max_parse_retries: Retries per item.
            skip_error: Mark evaluation failures as NaN instead of 1.0.
            rerun: Force re-evaluation even if columns exist.
            checkpoint_interval: Save checkpoint every N items (sequential).
        """
        self._check_init()
        self._validate_dataset_alignment(questions, ground_truths)

        # Skip if already done
        if self.has_model(model_name) and not rerun:
            acc = self.get_model_accuracy(model_name)
            errs = len(self.get_error_indices(model_name))
            accuracy = f"{acc:.2%}" if acc is not None else "N/A"
            print(f"Model '{model_name}' already evaluated (acc={accuracy}, {errs} errors).")
            print("Use rerun=True to force, or fix_errors() to fix failures.")
            return

        ckpt_path = os.path.join(
            self.output_dir, f".checkpoint_{self.dataset_name}_{model_name}.json"
        )
        predictor_model = getattr(predictor, "model", None)
        config_fingerprint = self._dataset_config_fingerprint(dataset_config)
        start_idx = 0
        completed: List[Tuple] = []

        # Resume from checkpoint
        if os.path.exists(ckpt_path) and not rerun:
            try:
                with open(ckpt_path, encoding="utf-8") as f:
                    ckpt = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: could not load checkpoint ({e}), starting fresh")
            else:
                start_idx, completed = self._validate_checkpoint(
                    ckpt,
                    model_name=model_name,
                    questions=questions,
                    ground_truths=ground_truths,
                    predictor_model=predictor_model,
                    config_fingerprint=config_fingerprint,
                    skip_error=skip_error,
                )
                print(f"Resuming from checkpoint: {start_idx}/{len(questions)}")

        all_results = list(completed)
        skipped = sum(
            1 for result in completed if result[3] in self.ERROR_SENTINELS
        )

        try:
            if parallel:
                remaining_q = questions[start_idx:]
                remaining_gt = ground_truths[start_idx:]
                if remaining_q:
                    batch_results = predictor.predict_batch_parallel(
                        remaining_q, remaining_gt, dataset_config,
                        max_workers=workers,
                        max_parse_retries=max_parse_retries,
                        skip_error=skip_error,
                    )
                    all_results.extend(batch_results)
                    skipped += sum(1 for r in batch_results if r[3] in self.ERROR_SENTINELS)
            else:
                for idx in range(start_idx, len(questions)):
                    result = predictor.predict_batch(
                        [questions[idx]],
                        [ground_truths[idx]],
                        dataset_config,
                        show_progress=False,
                        max_parse_retries=max_parse_retries,
                        skip_error=skip_error,
                    )[0]
                    if result[3] in self.ERROR_SENTINELS:
                        skipped += 1
                    all_results.append(result)

                    # Save checkpoint periodically
                    if (idx + 1) % checkpoint_interval == 0:
                        self._save_checkpoint(
                            ckpt_path,
                            idx,
                            all_results,
                            model_name,
                            predictor_model=predictor_model,
                            config_fingerprint=config_fingerprint,
                            skip_error=skip_error,
                        )

                # Final checkpoint
                self._save_checkpoint(
                    ckpt_path,
                    len(questions) - 1,
                    all_results,
                    model_name,
                    predictor_model=predictor_model,
                    config_fingerprint=config_fingerprint,
                    skip_error=skip_error,
                )

        except Exception:
            if all_results:
                last = start_idx + len(all_results) - 1 - len(completed)
                self._save_checkpoint(
                    ckpt_path,
                    last,
                    all_results,
                    model_name,
                    predictor_model=predictor_model,
                    config_fingerprint=config_fingerprint,
                    skip_error=skip_error,
                )
                print(f"Error! Progress saved ({len(all_results)} items). Re-run to resume.")
            raise

        # Store into CSV
        predictions = [r[3] for r in all_results]
        labels = [r[4] for r in all_results]
        raw_responses = [r[2] for r in all_results]
        self.add_model_predictions(model_name, predictions, labels, raw_responses, rerun=rerun)
        self.save()

        # Clean up checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # Report
        valid = [l for l in labels if not (isinstance(l, float) and np.isnan(l))]
        acc = 1 - (sum(valid) / len(valid)) if valid else None
        accuracy = f"{acc:.2%}" if acc is not None else "N/A"
        print(
            f"\nModel: {model_name} | Evaluated: {len(valid)} | "
            f"Skipped: {skipped} | Accuracy: {accuracy}"
        )

    def fix_errors(
        self,
        predictor,
        model_name: str,
        dataset_config,
        questions: List[Any],
        ground_truths: List[Any],
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 5,
        skip_error: bool = False,
    ) -> None:
        """Re-run failed predictions, including parse and backend errors.

        Args:
            predictor: :class:`LLMPredictor` instance.
            model_name: Friendly model name.
            dataset_config: :class:`DatasetConfig`.
            questions: Full question list (same as original run).
            ground_truths: Full ground-truth list.
        """
        self._check_init()
        self._validate_dataset_alignment(questions, ground_truths)
        if not self.has_model(model_name):
            print(f"Model '{model_name}' not found in CSV. Run evaluation first.")
            return

        error_idx = self.get_error_indices(model_name)
        if not error_idx:
            acc = self.get_model_accuracy(model_name)
            accuracy = f"{acc:.2%}" if acc is not None else "N/A"
            print(f"No errors for '{model_name}'. Accuracy: {accuracy}")
            return

        print(f"Fixing {len(error_idx)} errors for '{model_name}'...")
        err_q = [questions[i] for i in error_idx]
        err_gt = [ground_truths[i] for i in error_idx]

        if parallel:
            fix_results = predictor.predict_batch_parallel(
                err_q, err_gt, dataset_config,
                max_workers=workers,
                max_parse_retries=max_parse_retries,
                skip_error=skip_error,
            )
        else:
            fix_results = []
            for q, gt in tqdm(zip(err_q, err_gt), total=len(err_q), desc="Fixing errors"):
                result = predictor.predict_batch(
                    [q],
                    [gt],
                    dataset_config,
                    show_progress=False,
                    max_parse_retries=max_parse_retries,
                    skip_error=skip_error,
                )[0]
                fix_results.append(result)

        preds = [r[3] for r in fix_results]
        labels = [r[4] for r in fix_results]
        raws = [r[2] for r in fix_results]
        self.update_predictions_at_indices(model_name, error_idx, preds, labels, raws)
        self.save()

        still_bad = sum(1 for p in preds if p in self.ERROR_SENTINELS)
        print(f"Fixed: {len(error_idx) - still_bad} | Still failing: {still_bad}")
        acc = self.get_model_accuracy(model_name)
        if acc is not None:
            print(f"New accuracy: {acc:.2%}")

    # ── internals ─────────────────────────────────────────────────────

    def _check_init(self):
        if self.df is None:
            raise ValueError("DataFrame not initialised. Call load_or_create() first.")

    def _validate_dataset_alignment(
        self, questions: List[Any], ground_truths: List[Any]
    ) -> None:
        """Ensure caller inputs exactly match the stored dataset and row order."""
        self._check_init()
        if len(questions) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(questions)} questions vs "
                f"{len(ground_truths)} ground truths"
            )
        if len(self.df) != len(questions):
            raise ValueError(
                f"Row count mismatch: CSV has {len(self.df)}, "
                f"but {len(questions)} questions provided"
            )

        required_columns = ("question", "ground_truth")
        missing_columns = [
            column for column in required_columns if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Existing CSV is missing required dataset columns: "
                + ", ".join(missing_columns)
            )

        supplied_fingerprint = self._fingerprint_dataset(questions, ground_truths)
        if (
            self.dataset_fingerprint is not None
            and supplied_fingerprint != self.dataset_fingerprint
        ):
            raise ValueError(
                "Dataset fingerprint mismatch. Use the same questions, ground "
                "truths, value types, and row order as the loaded CSV."
            )

        for column in required_columns:
            stored_values = [
                self._csv_text(value) for value in self.df[column].tolist()
            ]
            supplied_source = questions if column == "question" else ground_truths
            supplied_values = [self._csv_text(value) for value in supplied_source]
            for row, (stored, supplied) in enumerate(
                zip(stored_values, supplied_values)
            ):
                if stored == supplied:
                    continue
                raise ValueError(
                    f"Dataset mismatch at row {row} for {column!r}: "
                    f"CSV has {stored!r}, but the caller provided "
                    f"{supplied!r}. Use the same "
                    "questions, ground truths, and row order as the CSV."
                )
        self.dataset_fingerprint = supplied_fingerprint

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return isinstance(missing, (bool, np.bool_)) and bool(missing)

    @classmethod
    def _csv_text(cls, value: Any) -> str:
        """Return the exact scalar text persisted by ``DataFrame.to_csv``."""
        if cls._is_missing_scalar(value):
            return ""
        return str(value)

    @classmethod
    def _fingerprint_dataset(
        cls, questions: List[Any], ground_truths: List[Any]
    ) -> str:
        """Build a typed, order-sensitive fingerprint for dataset identity."""
        rows = []
        for question, ground_truth in zip(questions, ground_truths):
            row = []
            for value in (question, ground_truth):
                if cls._is_missing_scalar(value):
                    row.append({"type": "missing"})
                    continue
                converted = convert_numpy_types(value)
                value_type = type(converted)
                row.append(
                    {
                        "type": f"{value_type.__module__}.{value_type.__qualname__}",
                        "repr": repr(converted),
                    }
                )
            rows.append(row)
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_dataset_metadata(self) -> None:
        """Validate the persisted dataset identity sidecar when present."""
        if not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path, encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read dataset metadata: {self.metadata_path}"
            ) from exc

        expected = {
            "dataset_name": self.dataset_name,
            "rows": len(self.df),
            "fingerprint": self.dataset_fingerprint,
        }
        mismatches = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Dataset metadata does not match the loaded CSV/caller inputs "
                f"({', '.join(mismatches)}). Use the original dataset and row order."
            )

    def _validate_checkpoint(
        self,
        checkpoint: Any,
        *,
        model_name: str,
        questions: List[Any],
        ground_truths: List[Any],
        predictor_model: Any,
        config_fingerprint: str,
        skip_error: bool,
    ) -> Tuple[int, List[Tuple]]:
        """Validate checkpoint identity and completed-row alignment."""

        def invalid(reason: str) -> ValueError:
            return ValueError(
                f"Checkpoint does not match the current evaluation ({reason}). "
                "Use the original dataset/model, delete the checkpoint, or "
                "pass rerun=True."
            )

        if not isinstance(checkpoint, dict):
            raise invalid("payload")
        if checkpoint.get("task") != self.dataset_name:
            raise invalid("dataset name")
        if checkpoint.get("model_name") != model_name:
            raise invalid("model name")
        if (
            "predictor_model" in checkpoint
            and checkpoint["predictor_model"] != predictor_model
        ):
            raise invalid("predictor model")
        if (
            "config_fingerprint" in checkpoint
            and checkpoint["config_fingerprint"] != config_fingerprint
        ):
            raise invalid("dataset configuration")

        last_completed = checkpoint.get("last_completed_idx")
        if (
            isinstance(last_completed, bool)
            or not isinstance(last_completed, Integral)
            or last_completed < -1
            or last_completed >= len(questions)
        ):
            raise invalid("last completed index")

        raw_results = checkpoint.get("results")
        if not isinstance(raw_results, list):
            raise invalid("results payload")
        if any(not isinstance(result, (list, tuple)) for result in raw_results):
            raise invalid("results payload")
        completed = [tuple(result) for result in raw_results]
        if len(completed) != last_completed + 1:
            raise invalid("result count")
        for row, result in enumerate(completed):
            if len(result) != 5:
                raise invalid(f"result row {row}")
            if isinstance(result[4], bool) or not isinstance(result[4], Real):
                raise invalid(f"result score at row {row}")
            try:
                score = float(result[4])
            except (OverflowError, TypeError, ValueError):
                raise invalid(f"result score at row {row}") from None
            if np.isinf(score):
                raise invalid(f"result score at row {row}")

        if "skip_error" in checkpoint:
            if checkpoint["skip_error"] != skip_error:
                raise invalid("skip-error policy")
        else:
            # Legacy checkpoints did not store the policy. Sentinel rows make
            # it observable: skipped failures have NaN scores, while failures
            # counted as errors have numeric scores.
            for result in completed:
                if result[3] not in self.ERROR_SENTINELS:
                    continue
                stored_skip_error = bool(np.isnan(float(result[4])))
                if stored_skip_error != skip_error:
                    raise invalid("skip-error policy")

        checkpoint_fingerprint = checkpoint.get("dataset_fingerprint")
        if checkpoint_fingerprint is not None:
            if checkpoint_fingerprint != self.dataset_fingerprint:
                raise invalid("dataset fingerprint")
        else:
            # Legacy checkpoints predate fingerprints. Compare the JSON form
            # they actually persisted so tuples and non-string mapping keys do
            # not become false mismatches after a JSON round trip.
            for row, result in enumerate(completed):
                expected_question = self._checkpoint_json(questions[row])
                expected_truth = self._checkpoint_json(ground_truths[row])
                if (
                    self._checkpoint_json(result[0]) != expected_question
                    or self._checkpoint_json(result[1]) != expected_truth
                ):
                    raise invalid(f"result prefix at row {row}")

        return last_completed + 1, completed

    @staticmethod
    def _checkpoint_json(value: Any) -> str:
        """Return the JSON representation used in checkpoint persistence."""
        return json.dumps(
            convert_numpy_types(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _dataset_config_fingerprint(dataset_config: Any) -> str:
        """Fingerprint stable, serializable parts of an evaluator config."""
        identity = {
            "name": getattr(dataset_config, "name", None),
            "json_schema": convert_numpy_types(
                getattr(dataset_config, "json_schema", None)
            ),
        }
        payload = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_checkpoint(
        self,
        path,
        last_idx,
        results,
        model_name,
        *,
        predictor_model,
        config_fingerprint,
        skip_error,
    ):
        data = convert_numpy_types({
            "last_completed_idx": last_idx,
            "results": results,
            "model_name": model_name,
            "task": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "predictor_model": predictor_model,
            "config_fingerprint": config_fingerprint,
            "skip_error": skip_error,
        })
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"💾 Checkpoint saved at {last_idx + 1}")


# Simple single-model CSV I/O (legacy compat)


def save_predictions_to_csv(
    results: List[Tuple], output_path: str, task: str = "generic"
) -> None:
    """Save prediction results to a simple CSV.

    Each row: ``index, question, ground_truth, raw_response, prediction, correct``.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "question", "ground_truth", "raw_response", "prediction", "correct"])
        for i, row in enumerate(results):
            w.writerow([i, *row])
    print(f"Saved {len(results)} predictions to {output_path}")


def load_predictions_from_csv(csv_path: str) -> Dict[str, List]:
    """Load predictions from a simple CSV.

    Returns dict with keys: ``questions``, ``ground_truths``, ``predictions``,
    ``correct_labels``.
    """
    questions, gts, preds, labels = [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(row["question"])
            gt = row["ground_truth"]
            gts.append(gt.lower() == "true" if gt.lower() in ("true", "false") else gt)
            p = row["prediction"]
            preds.append(p.lower() == "true" if p.lower() in ("true", "false") else p)
            labels.append(float(row["correct"]))
    return {"questions": questions, "ground_truths": gts, "predictions": preds, "correct_labels": labels}


# HuggingFace / local dataset loading


def load_dataset_data(task: str) -> Tuple[List[str], List[Any]]:
    """Load a full dataset from HuggingFace (or local files for GQA / DICES-T2I).

    Supported tasks: ``strategyqa``, ``gsm8k``, ``svamp``, ``mmlu``,
    ``mmlu_professionallaw``, ``jigsaw``, ``toxicchat``, ``gqa``, ``dices``,
    ``dices_t2i``.

    Returns ``(questions, ground_truths)``.
    """
    from datasets import load_dataset  # lazy import

    questions: List = []
    ground_truths: List = []

    if task == "strategyqa":
        ds = load_dataset("ChilleD/StrategyQA", split="train")
        for ex in ds:
            questions.append(ex["question"])
            ground_truths.append(ex["answer"])

    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            questions.append(ex["question"])
            ground_truths.append(ex["answer"])

    elif task == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="train")
        for ex in ds:
            questions.append(ex["Body"] + " " + ex["Question"])
            ground_truths.append(ex["Answer"])

    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
        df = ds.to_pandas()
        questions = df.to_dict(orient="records")
        ground_truths = df["answer"].tolist()

    elif task == "mmlu_professionallaw":
        ds = load_dataset("cais/mmlu", "professional_law", split="test")
        df = ds.to_pandas()
        questions = df.to_dict(orient="records")
        ground_truths = df["answer"].tolist()

    elif task == "jigsaw":
        ds = load_dataset("google/civil_comments", split="train")
        df = ds.to_pandas()
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)
        questions = df["text"].tolist()
        ground_truths = df["toxicity"].tolist()

    elif task == "toxicchat":
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        df = ds.to_pandas()
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)
        questions = df["user_input"].tolist()
        ground_truths = df["toxicity"].tolist()

    elif task == "gqa":
        gqa_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gqa_raw.csv")
        df = pd.read_csv(gqa_path)
        questions = df[["question", "image_id"]].to_dict(orient="records")
        ground_truths = df["ground_truth"].tolist()

    elif task == "dices":
        import tensorflow_datasets as tfds
        ds = tfds.load("dices/350", split="train", shuffle_files=True)
        for ex in tfds.as_numpy(ds.take(1500)):
            questions.append({"context": ex["context"].decode(), "response": ex["response"].decode()})
            ground_truths.append((float(ex["Q_overall"]) - 1.0) / 4.0)

    elif task == "dices_t2i":
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "dices_t2i_data",
            "latest_dv-t2i-full-filtered.jsonl",
        )
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompt = d.get("text_prompt", d.get("prompt", d.get("text", "")))
                rating = d.get("how_harmful", d.get("human_rating"))
                if not prompt or rating is None or rating == "unsure":
                    continue
                questions.append(prompt)
                ground_truths.append(float(rating) / 4.0)
        if len(questions) > 1500:
            idx = np.random.RandomState(42).choice(len(questions), 1500, replace=False)
            questions = [questions[i] for i in idx]
            ground_truths = [ground_truths[i] for i in idx]

    else:
        raise ValueError(f"Unknown task: {task}")

    return questions, ground_truths
