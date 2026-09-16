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

"""Pretrain model selection for BQ active sampling.

Provides automatic selection of pretrain models using GMM clustering on
reference benchmark features. The key idea: models that perform similarly
on reference benchmarks are likely to be similar on the target benchmark,
making them good pretrain sources — without needing the target model's
actual evaluation results.

Example::

    from proeval.sampler.pretrain_selector import select_pretrain_models_gmm

    indices, names = select_pretrain_models_gmm(
        target_benchmark="svamp",
        target_model="gemini25_flash",
    )

"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from proeval.sampler.data import _default_data_dir


# Reference benchmark discovery
def get_reference_benchmarks(
    target_benchmark: str, data_dir: Optional[str] = None
) -> List[str]:
    """Return all available benchmarks as references, excluding *target_benchmark*.

    Scans *data_dir* for ``*_predictions.csv`` files and returns every
    benchmark name found except the target itself.

    Args:
        target_benchmark: Benchmark being evaluated (will be excluded).
        data_dir: Directory containing prediction CSVs.  Defaults to ``data/``.

    Returns:
        Sorted list of benchmark names.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    benchmarks: List[str] = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith("_predictions.csv"):
            name = fname.replace("_predictions.csv", "")
            if name != target_benchmark:
                benchmarks.append(name)
    return benchmarks


# Feature extraction from reference benchmark predictions
def _load_benchmark_predictions(
    benchmark: str, data_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """Load finite real scores for all models on a single benchmark.

    Returns:
        ``(prediction_matrix, model_names)`` where ``prediction_matrix`` has
        shape ``(n_questions, n_models)``.
    """
    csv_path = os.path.join(data_dir, f"{benchmark}_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    model_cols = [c for c in df.columns if c.startswith("label_")]
    if not model_cols:
        raise ValueError(f"No label_ columns found in {csv_path}")

    try:
        prediction_matrix = df[model_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction scores must be numeric in {csv_path}") from exc
    if not prediction_matrix.shape[0] or not np.isfinite(prediction_matrix).all():
        raise ValueError(f"Prediction scores must be nonempty and finite in {csv_path}")
    if benchmark in ("dices", "dices_t2i"):
        prediction_matrix = (prediction_matrix >= 0.5).astype(float)
    model_names = [c[len("label_") :] for c in model_cols]
    return prediction_matrix, model_names


def _build_features(
    reference_benchmarks: List[str], data_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """Build per-model feature vectors from reference benchmark predictions.

    Concatenate each model's per-question scores across reference benchmarks,
    then retain 95% of their variance using PCA. Models absent from a benchmark
    get each missing score imputed with that question's mean across available
    models. An explicitly present score column must contain finite values.

    Returns:
        ``(features, model_names)`` where ``features`` has shape
        ``(n_models, n_pca_components)``.
    """
    benchmark_data: Dict[str, Dict] = {}
    all_models: set = set()

    for bench in reference_benchmarks:
        preds, models = _load_benchmark_predictions(bench, data_dir)
        benchmark_data[bench] = {"predictions": preds, "models": models}
        all_models.update(models)

    all_models_sorted = sorted(all_models)
    if not all_models_sorted:
        raise ValueError(
            f"No models found across reference benchmarks: {reference_benchmarks}"
        )

    # Build feature matrix with NaN for missing entries
    n_models = len(all_models_sorted)
    n_features = sum(data["predictions"].shape[0] for data in benchmark_data.values())
    features = np.full((n_models, n_features), np.nan)
    model_indices = {model: index for index, model in enumerate(all_models_sorted)}
    offset = 0
    for bench in reference_benchmarks:
        data = benchmark_data[bench]
        n_questions = data["predictions"].shape[0]
        for pred_idx, model in enumerate(data["models"]):
            features[model_indices[model], offset : offset + n_questions] = (
                data["predictions"][:, pred_idx]
            )
        offset += n_questions

    # Impute NaN with column means
    col_means = np.nanmean(features, axis=0)
    features = np.where(np.isnan(features), col_means, features)

    # PCA cannot report explained-variance ratios for identical behaviors.
    if np.all(features == features[0]):
        return np.zeros((n_models, 1)), all_models_sorted

    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(features)
    return pca.transform(features), all_models_sorted


# GMM clustering
def _find_optimal_clusters(
    features: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42,
) -> int:
    """Select optimal GMM cluster count using BIC."""
    n_samples = features.shape[0]
    n_distinct = np.unique(features, axis=0).shape[0]
    max_k = min(max_clusters, n_samples - 1, n_distinct)

    bics: List[Tuple[int, float]] = []
    for k in range(1, max_k + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                gmm = GaussianMixture(
                    n_components=k, random_state=random_state,
                    covariance_type="diag", reg_covar=1e-4,
                )
                gmm.fit(features)
            bic = gmm.bic(features)
            if np.isfinite(bic):
                bics.append((k, bic))
        except (ValueError, np.linalg.LinAlgError, ConvergenceWarning):
            continue

    if not bics:
        raise ValueError("GMM source selection abstained: no valid clustering could be fitted")
    return min(bics, key=lambda x: x[1])[0]


# Public API
def select_pretrain_models_gmm(
    target_benchmark: str,
    target_model: str,
    data_dir: Optional[str] = None,
    reference_benchmarks: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[List[int], List[str]]:
    """Select pretrain models via GMM clustering on reference benchmark features.

    Fits a GMM to per-model feature vectors derived from *reference_benchmarks*
    and returns the models in the same cluster as the target model.  This
    **does not** require knowing the target model's eval results on the target
    benchmark.

    Args:
        target_benchmark: Benchmark being evaluated (used to auto-select
            reference benchmarks if ``reference_benchmarks`` is ``None``).
        target_model: Name of the target model.
        data_dir: Directory containing prediction CSVs.  Defaults to
            ``data/``.
        reference_benchmarks: Benchmarks for clustering. If ``None``, use all
            available benchmarks except the target. The target benchmark is
            always excluded, including when explicitly listed.
        n_clusters: Number of GMM clusters.  ``None`` → auto-select via BIC.
        random_state: Seed for GMM.
        verbose: Print selection details.

    Returns:
        ``(pretrain_indices, pretrain_names)`` — indices into the model list
        of the *target benchmark's* prediction CSV, and the corresponding
        model names.  These indices can be passed directly as
        ``pretrain_indices`` to :meth:`BQPriorSampler.sample`.

    Raises:
        ValueError: If the target cluster provides fewer than three source
            models present in the target benchmark. Selection abstains rather
            than substituting models from other clusters.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    # Auto-select reference benchmarks if needed
    if reference_benchmarks is None:
        reference_benchmarks = get_reference_benchmarks(target_benchmark, data_dir)
    reference_benchmarks = list(dict.fromkeys(
        bench for bench in reference_benchmarks if bench != target_benchmark
    ))

    # Filter to only benchmarks that actually have the target model's predictions
    filtered_refs: List[str] = []
    for bench in reference_benchmarks:
        try:
            _, bench_models = _load_benchmark_predictions(bench, data_dir)
            if target_model in bench_models:
                filtered_refs.append(bench)
        except FileNotFoundError:
            continue
    if not filtered_refs:
        raise ValueError(
            f"Target model '{target_model}' not found in any reference benchmark."
        )
    reference_benchmarks = filtered_refs

    if verbose:
        print(f"\n{'='*60}")
        print("Pretrain Model Selection (GMM)")
        print(f"{'='*60}")
        print(f"Target benchmark:     {target_benchmark}")
        print(f"Target model:        {target_model}")
        print(f"Reference benchmarks: {reference_benchmarks}")

    # Build features from reference benchmarks
    features, ref_model_names = _build_features(reference_benchmarks, data_dir)
    if len(ref_model_names) < 4:
        raise ValueError(
            "GMM source selection abstained: at least three source models "
            "besides the target are required"
        )

    if verbose:
        print(f"Models in references: {len(ref_model_names)}")
        print(f"Feature dimensions:   {features.shape[1]}")

    # Verify target model is present
    if target_model not in ref_model_names:
        raise ValueError(
            f"Target model '{target_model}' not found in reference data. "
            f"Available: {ref_model_names}"
        )
    target_ref_idx = ref_model_names.index(target_model)

    # Determine cluster count
    if n_clusters is None:
        n_clusters = _find_optimal_clusters(features, random_state=random_state)
    else:
        n_clusters = min(n_clusters, len(ref_model_names))
    if verbose:
        print(f"GMM clusters:         {n_clusters}")

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_clusters, random_state=random_state,
        covariance_type="diag", reg_covar=1e-4,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            labels = gmm.fit_predict(features)
    except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
        raise ValueError(
            "GMM source selection abstained: clustering could not be fitted "
            "or did not converge"
        ) from exc
    target_cluster = labels[target_ref_idx]

    # Models in same cluster (excluding target)
    selected_ref_names = [
        ref_model_names[i]
        for i in range(len(ref_model_names))
        if labels[i] == target_cluster and i != target_ref_idx
    ]

    if verbose:
        print(f"\nSelected {len(selected_ref_names)} pretrain models (cluster {target_cluster}):")
        for i, name in enumerate(selected_ref_names, 1):
            dist = float(np.linalg.norm(
                features[ref_model_names.index(name)] - features[target_ref_idx]
            ))
            print(f"  {i}. {name} (dist: {dist:.4f})")

    # Map back to target benchmark's model indices
    # Need to load the target benchmark's model list to resolve indices
    target_csv = os.path.join(data_dir, f"{target_benchmark}_predictions.csv")
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Target predictions not found: {target_csv}")
    target_df = pd.read_csv(target_csv, nrows=0)  # header only
    target_model_cols = [c for c in target_df.columns if c.startswith("label_")]
    target_model_names = [c[len("label_") :] for c in target_model_cols]

    pretrain_indices: List[int] = []
    pretrain_names: List[str] = []
    for name in selected_ref_names:
        if name in target_model_names:
            pretrain_indices.append(target_model_names.index(name))
            pretrain_names.append(name)
        elif verbose:
            print(f"  Warning: '{name}' not in target benchmark, skipping.")

    if len(pretrain_indices) < 3:
        raise ValueError(
            "GMM source selection abstained: target cluster provides "
            f"{len(pretrain_indices)} source models present in {target_benchmark!r}; "
            "at least three are required"
        )

    if verbose:
        print(f"\nFinal pretrain set ({len(pretrain_names)} models): {pretrain_names}")
        print(f"{'='*60}")

    return pretrain_indices, pretrain_names
