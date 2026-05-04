# ProEval

![GitHub License](https://img.shields.io/github/license/google-deepmind/proeval)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23099-b31b1b.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.23099)
[![Contact Us](https://img.shields.io/badge/Contact%20Us-proeval@google.com-4285F4?logo=gmail&logoColor=white)](mailto:proeval@google.com)

Slash GenAI evaluation costs by up to 100x while actively discovering model failure patterns to guide better AI development.

1. 💰 **Cut GenAI eval costs up to 100×** — achieve ±1% accuracy with a fraction of the samples
2. 🔍 **Discover failure cases** — proactively surface diverse bugs under strict evaluation budgets
3. 🧠 **Transfer learning over benchmarks** — pre-trained GP surrogates generalize to new models instantly
4. 🧩 **Easy Integration** - Easily to integrate into the GenAI evaluation systems with different modalities
5. ✅ **Validated on reasoning, safety & classification** — GSM8K, MMLU, StrategyQA, Jigsaw, and more

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from proeval import BQPriorSampler, LLMPredictor, DATASET_CONFIGS
from proeval.sampler import load_predictions, extract_model_predictions
import numpy as np

# Estimate a model's error rate with ~1% of the data
sampler = BQPriorSampler(noise_variance=0.3)
result = sampler.sample(predictions="svamp", target_model="gemini25_flash", budget=50)

# Compare against the true error rate
df = load_predictions("svamp")
pred_matrix, model_names = extract_model_predictions(df)
true_mean = np.mean(pred_matrix[:, model_names.index("gemini25_flash")])

print(f"Estimated error rate: {result.estimates[-1]:.4f}")
print(f"MAE: {result.mae(true_mean):.4f}")
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
@article{huang2026proeval,
  title={{{ProEval}: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation}},
  author={Huang, Yizheng and Zeng, Wenjun and Kumaresan, Aditi and Wang, Zi},
  journal={arXiv preprint arXiv:2604.23099 [cs.LG]},
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
