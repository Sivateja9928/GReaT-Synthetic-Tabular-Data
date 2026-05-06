# GReaT: Fine-Tuning LLMs for Synthetic Tabular Data Generation

> Reproduction and extension of GReaT: Generation of Realistic Tabular Data
> (Borisov et al., ICLR 2023) with novel I-LoRA and Statistical Prompt-Aided Sampling.

---

## Overview

Generating high-quality synthetic tabular data is a critical challenge in
machine learning, especially in privacy-sensitive or data-scarce settings.
Traditional GAN and VAE approaches suffer from three core problems:

| Problem | Description |
|---|---|
| Lossy Preprocessing | Encoding categories as numbers destroys semantic meaning |
| Contextual Knowledge | Features like age, education, occupation are semantically linked |
| Arbitrary Conditioning | Need to generate data conditioned on any subset of features |

GReaT solves all three by converting tabular rows into natural language
sentences and fine-tuning a pretrained GPT-2 language model on them.

Each row becomes a sentence like:
age is 39, workclass is State-gov, education is Bachelors,
occupation is Adm-clerical, income is <=50K

Feature order is randomly shuffled at training time, enabling arbitrary conditioning at inference.

---

## Key Results

### Fine-Tuning Variants

| Model | Discriminator | LR | DT | RF |
|---|---|---|---|---|
| LoRA | 0.793 | 0.602 | 0.651 | 0.756 |
| LoRA + Prompt | 0.762 | 0.752 | 0.742 | 0.797 |

Discriminator: lower is better (50% = perfect). ML Efficiency: higher is better.

### vs GReaT Paper Baselines (Adult Income)

| Method | Discriminator |
|---|---|
| CopulaGAN | 88.54 |
| TVAE | 88.49 |
| CTGAN | 97.23 |
| Distill-GReaT | 69.79 |
| GReaT (paper) | 62.84 |
| LoRA + Prompt (ours) | 76.2 |

Our approach outperforms all GAN and VAE baselines.

### I-LoRA Parameter Efficiency

| Method | Trainable Params | Ratio | Discriminator |
|---|---|---|---|
| Full Fine-Tuning | 126,061,824 | 100% | baseline |
| LoRA | 1,622,016 | 1.287% | 0.76 |
| I-LoRA (p=0.30) | 1,105,728 | 0.881% | 0.79 |

I-LoRA matches LoRA quality with 31.8% fewer trainable parameters.

---

## Novel Contributions

### 1. I-LoRA (Importance-Guided LoRA)

Selects the most important neurons before training via a calibration pass,
then applies low-rank adapters only to those neurons.

Step 1 - Neuron Importance Scoring:
Ii = sum || (dL/dWi) * Wi ||_F

Step 2 - Top-K Selection: keep top p fraction of neurons

Step 3 and 4 - Sparse scaled adapter:
W' = W0 + (a/r) * Pi_S B_S A

| Method | When scored | What it controls |
|---|---|---|
| AdaLoRA | During training | Rank per layer |
| RoseLoRA | During training | Sparsity of matrices |
| I-LoRA | Before training | Which rows get adapter |

### 2. Statistical Prompt-Aided Sampling

Injects MI-weighted conditional histograms into each generation step
to correct statistical imbalances in unconstrained sampling.
p*(Y) = sum MI(Xi,Y) * p(Y|Xi=xi) / sum MI(Xi,Y)

Features with low Mutual Information are discarded as weak signals.
Result: discriminator score improved from 0.793 to 0.762.

---

## Project Structure
GReaT-Synthetic-Tabular-Data/
|
|-- preprocessing/
|   |-- preprocess.py          # Data pipeline
|
|-- ilora/
|   |-- great_ilora.py         # I-LoRA main class
|   |-- ilora_linear.py        # Sparse linear layer
|   |-- importance_scorer.py   # Calibration pass
|   |-- utils.py
|   |-- init.py
|
|-- data/
|   |-- README.md
|
|-- reports/
|   |-- Group_17_Report.pdf
|
|-- results/
|   |-- README.md
|
|-- README.md

---

## Installation

```bash
git clone https://github.com/Sivateja9928/GReaT-Synthetic-Tabular-Data.git
cd GReaT-Synthetic-Tabular-Data
pip install be-great peft transformers torch pandas scikit-learn
```

---

## Usage

### Run Data Preprocessing

```python
from preprocessing.preprocess import run_pipeline
df, corpus = run_pipeline()
```

### Fine-tune with GReaT

```python
from be_great import GReaT
import pandas as pd

df = pd.read_csv('preprocessing/adult_income_cleaned.csv')
model = GReaT(llm='distilgpt2', epochs=100, batch_size=32)
model.fit(df)
synthetic_data = model.sample(n_samples=100)
```

### Fine-tune with I-LoRA

```python
from ilora.great_ilora import GReaTILoRA

ilora_config = {
    "r": 16,
    "lora_alpha": 32,
    "dropout": 0.05,
    "top_k_ratio": 0.30,
    "n_calibration_batches": 50,
    "target_modules": ["c_attn", "c_proj"]
}

model = GReaTILoRA(
    llm='gpt2',
    method='ilora',
    epochs=5,
    batch_size=4,
    ilora_cfg=ilora_config
)
model.fit(df)
synthetic_data = model.sample(n_samples=100)
```

---

## Team Contributions

| Contributor | Role | Contributions |
|---|---|---|
| Siva Teja Sivarathri (B23179) | Data, Preprocessing and Sampling | Dataset selection, full preprocessing pipeline, textual encoding with random feature permutation, LoRA and I-LoRA experiments, Statistical Prompt-Aided Sampling |
| Garima Ketan Chauhan (B22206) | Fine-Tuning and I-LoRA | I-LoRA architecture, importance scorer, fine-tuning pipeline |
| Gargi Ketan Chauhan (B22161) | Fine-Tuning and Evaluation | LoRA implementation, evaluation metrics, results analysis |
| Prisha Singh (B22168) | Prompt-Based Sampling | Statistical Prompt-Aided Sampling design, MI-weighted histogram construction |

---

## References

1. Borisov, V. et al. Language Models are Realistic Tabular Data Generators. ICLR 2023.
2. Hu, E. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
3. Zhang, Q. et al. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. ICLR 2023.
4. Liu, S. et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024.
5. Kohavi, R. Scaling up the accuracy of Naive-Bayes classifiers. KDD 1996.

---

IIT Mandi | CS-683 Generative AI | 2025