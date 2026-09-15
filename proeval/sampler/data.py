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

"""Data loading utilities for active evaluation.

Provides functions to load prediction CSVs, text embeddings, and prepare
train/test splits for Bayesian Quadrature sampling.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _coerce_real_array(values, *, name: str) -> np.ndarray:
    """Convert numeric input to float without silently discarding imaginary parts."""
    unconverted = np.asarray(values)
    if np.iscomplexobj(unconverted) or unconverted.dtype.kind in {"M", "m"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if unconverted.dtype.kind == "O" and any(
        isinstance(value, (complex, np.complexfloating, np.datetime64, np.timedelta64))
        for value in unconverted.flat
    ):
        raise ValueError(f"{name} must contain only real numeric values")
    try:
        return np.asarray(values, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc


def _prepare_score_features(
    source_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the linear-kernel features and prior from source scores."""
    prior_mean = np.mean(source_scores, axis=1)
    prior_covariance = np.cov(source_scores)
    if prior_covariance.ndim == 0:
        prior_covariance = np.array([[prior_covariance]])

    test_x = source_scores.T
    n_sources = test_x.shape[0]
    if n_sources > 1:
        test_x = (test_x - prior_mean) / np.sqrt(n_sources - 1)
    else:
        test_x = test_x - prior_mean

    return test_x, prior_mean, prior_covariance


def _default_data_dir() -> str:
    """Resolve the default data directory (data/)."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )


def load_predictions(
    dataset_name: str, data_dir: str = None
) -> pd.DataFrame:
    """Load predictions CSV for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'svamp', 'strategyqa').
        data_dir: Directory containing CSV files. Defaults to ``data/``.

    Returns:
        DataFrame with model prediction columns (``label_<model>``).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    csv_path = os.path.join(data_dir, f"{dataset_name}_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_embeddings(
    dataset_name: str,
    data_dir: str = None,
    embedding_model: str = "text_embedding_3_large",
) -> np.ndarray:
    """Load pre-computed text embeddings for a dataset.

    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory containing ``.npy`` files. Defaults to ``data/``.
        embedding_model: Embedding model name suffix.

    Returns:
        Embeddings array of shape ``(n_samples, n_features)``.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    if dataset_name == "gqa":
        embedding_path = os.path.join(data_dir, "gqa_embeddings.npy")
    else:
        embedding_path = os.path.join(
            data_dir, f"{dataset_name}_embeddings_{embedding_model}.npy"
        )
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    return np.load(embedding_path)


def extract_model_predictions(
    df: pd.DataFrame, dataset_name: str = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract a real-valued score matrix from a predictions DataFrame.

    Binary labels conventionally use **1=error, 0=correct**. General finite
    real-valued scores are also supported, as in the paper's score function.

    For DICES/DICES-T2I datasets, continuous error scores are binarised at
    0.5: scores greater than or equal to 0.5 are failures. This matches the
    evaluation convention used elsewhere in ProEval.
    For other datasets, ``label_`` columns are used directly and must contain
    finite real scores.

    Args:
        df: Predictions DataFrame with ``label_<model>`` columns.
        dataset_name: Dataset name for special-case handling.

    Returns:
        ``(prediction_matrix, model_names)`` where ``prediction_matrix`` has
        shape ``(n_samples, n_models)``. Non-DICES scores retain their scale.
    """
    model_columns = [
        column
        for column in df.columns
        if isinstance(column, str) and column.startswith("label_")
    ]
    if not model_columns:
        raise ValueError(
            "predictions must contain at least one column named 'label_<model>'"
        )
    model_names = [column[len("label_") :] for column in model_columns]
    if any(not name for name in model_names):
        raise ValueError("prediction label columns must include a model name")
    if len(set(model_columns)) != len(model_columns):
        raise ValueError("prediction label column names must be unique")
    if len(set(model_names)) != len(model_names):
        raise ValueError("prediction label model names must be unique")

    numeric_labels = _coerce_real_array(
        df[model_columns].to_numpy(), name="prediction labels"
    )
    if numeric_labels.shape[0] == 0:
        raise ValueError("predictions must contain at least one item")
    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError("prediction labels must contain only finite values")

    model_data = {}
    for model_index, model_name in enumerate(model_names):
        y_labels = numeric_labels[:, model_index]
        # Use raw labels: 1=error, 0=correct.
        # For DICES/DICES_T2I, continuous error scores at or above 0.5 count
        # as failures, matching the failure-discovery threshold.
        # For other datasets: raw labels are already 1=error, 0=correct.
        if dataset_name in ("dices", "dices_t2i"):
            y_error = (y_labels >= 0.5).astype(float)
        else:
            y_error = y_labels
        model_data[model_name] = y_error

    prediction_matrix = np.column_stack([model_data[n] for n in model_names])
    return prediction_matrix, model_names


def setup_train_test_split(
    prediction_matrix: np.ndarray,
    target_model: Union[int, str],
    pretrain_indices: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split prediction matrix into pre-training features and target model test data.

    Args:
        prediction_matrix: Full prediction matrix ``(n_samples, n_models)``.
        target_model: Name or index of the model to target for testing.
        pretrain_indices: Optional list of model indices to use for pre-training.
            If ``None``, uses all models except the target.
        model_names: List of model names (required when *target_model* is a
            string).

    Returns:
        ``(pretrain_matrix, test_x, test_y, prior_mean, prior_cov)``
    """
    # Resolve target model to index
    if isinstance(target_model, str):
        if model_names is None:
            raise ValueError(
                "model_names must be provided when target_model is a string"
            )
        if target_model not in model_names:
            raise ValueError(
                f"Model {target_model!r} not found. Available: {model_names}"
            )
        target_model_index = model_names.index(target_model)
    else:
        target_model_index = int(target_model)

    test_y = prediction_matrix[:, target_model_index]

    if pretrain_indices is not None:
        pretrain_matrix = prediction_matrix[:, pretrain_indices]
    else:
        pretrain_matrix = np.delete(
            prediction_matrix,
            slice(target_model_index, target_model_index + 1),
            axis=1,
        )

    test_x, u, S = _prepare_score_features(pretrain_matrix)

    return pretrain_matrix, test_x, test_y, u, S
