# ProEval

![GitHub License](https://img.shields.io/github/license/google-deepmind/proeval)
[![PyPI version](https://img.shields.io/pypi/v/proeval.svg?logo=pypi&logoColor=white)](https://pypi.org/project/proeval/)
[![PyPI downloads](https://img.shields.io/pypi/dm/proeval.svg?logo=pypi&logoColor=white)](https://pypi.org/project/proeval/)
[![Python versions](https://img.shields.io/pypi/pyversions/proeval.svg)](https://pypi.org/project/proeval/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23099-b31b1b.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.23099)
[![Contact Us](https://img.shields.io/badge/Contact%20Us-proeval@google.com-4285F4?logo=gmail&logoColor=white)](mailto:proeval@google.com)

Slash GenAI evaluation costs by up to 100x while actively discovering model failure patterns to guide better AI development.

1. 💰 **Cut GenAI eval costs up to 100×** — achieve ±1% accuracy with a fraction of the samples
2. 🔍 **Discover failure cases** — proactively surface diverse bugs under strict evaluation budgets
3. 🧠 **Transfer learning over benchmarks** — pre-trained GP surrogates generalize to new models instantly
4. 🧩 **Easy Integration** - Easily to integrate into the GenAI evaluation systems with different modalities
5. ✅ **Validated on reasoning, safety & classification** — GSM8K, MMLU, StrategyQA, Jigsaw, and more

## Installation

### Published release

Install ProEval from [PyPI](https://pypi.org/project/proeval/). The current
`0.1.0` release requires Python 3.10 or newer:

```bash
python -m pip install --upgrade proeval
python -m pip install --upgrade "proeval[topic]"  # Optional topic generation
```

The `0.1.0` package predates recent API and sampling improvements on `main`,
installs PyTorch as a core dependency, and does not include the repository's
research data.

### Current `main`

The current source supports Python 3.9 or newer and provides the APIs used by
the examples below. Clone the repository for the latest library code or the
research artifacts under `data/`:

```bash
git clone https://github.com/google-deepmind/proeval.git
cd proeval
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[encoder]"   # PyTorch — for BQEncoderSampler and encoder training
python -m pip install -e ".[topics]"    # BERTopic + HDBSCAN — for TopicAwareGenerator
python -m pip install -e ".[datasets]"  # HuggingFace datasets — for evaluator.load_dataset_data
python -m pip install -e ".[all]"       # everything above
python -m pip install -e ".[dev]"       # pytest, ruff, build tooling
```

## Quick Start

This example targets the current `main` API. It uses an in-memory prediction
table, so it works without an API key or the repository's research data.
Prediction columns must be named `label_<model>` and use `1 = failure`,
`0 = correct`.

```python
import pandas as pd

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
    pretrain_mode="all",  # Required for an unnamed in-memory DataFrame.
    seed=42,
)

print(f"Estimated error rate: {result.estimates[-1]:.4f}")
print(f"Selected rows: {result.selected_indices}")
```

See the [Python API guide](./proeval/README.md) for custom datasets, model
evaluation, and test-case generation.

## Reproducing the Paper Experiments

The pre-computed prediction CSVs and embeddings under `data/` are research
artifacts stored only in the GitHub repository. They are approximately 381 MB
and are intentionally not included in the PyPI wheel or source distribution.
Clone the repository before using dataset-name shortcuts such as
`predictions="svamp"`, or pass your own DataFrame or explicit `data_dir`.

From a source checkout, run the bundled example with:

```bash
python experiment/sample_usage.py
```

## Experiments

Here is an example of how to run the experiments:

```shell
# BQ performance estimation (runs BQ-SF, BQ-RPF, etc.)
python -m experiment.exp_performance_estimation --dataset svamp --n-runs 5
```

You can find the comprehensive [experiment details](./experiment/README.md) and dataset settings [here](./data/README.md).

## Citation

If the work did some helps on your research/project, please cite our ICML 2026 paper. Thank you!

```
@inproceedings{huang2026proeval,
  title={{{ProEval}: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation}},
  author={Huang, Yizheng and Zeng, Wenjun and Kumaresan, Aditi and Wang, Zi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  url={https://arxiv.org/abs/2604.23099}
}
```

## License

```
Copyright 2026 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license. You
may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
```
