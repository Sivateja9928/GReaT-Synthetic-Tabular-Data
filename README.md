---

## Installation

```bash
# Clone the repo
git clone https://github.com/Sivateja9928/GReaT-Synthetic-Tabular-Data.git
cd GReaT-Synthetic-Tabular-Data

# Install dependencies
pip install be-great peft transformers torch pandas scikit-learn
```

---

## Usage

### Run Data Preprocessing
```python
from preprocessing.preprocess import run_pipeline

# Runs full pipeline — loads, cleans, encodes, saves
df, corpus = run_pipeline()
```

### Fine-tune with GReaT (Full Fine-Tuning)
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
| **Siva Teja Sivarathri (B23179)** | Data, Preprocessing & Sampling | Dataset selection and justification, full preprocessing pipeline (missing value handling, rare category consolidation, duplicate removal), textual encoding with random feature permutation (Definitions 1 & 2 from GReaT paper), LoRA and I-LoRA fine-tuning experiments, Statistical Prompt-Aided Sampling (MI-weighted conditional histogram construction for guided generation) |
| **Garima Ketan Chauhan (B22206)** | Fine-Tuning & I-LoRA | I-LoRA architecture design, importance scorer implementation, fine-tuning pipeline |
| **Gargi Ketan Chauhan (B22161)** | Fine-Tuning & Evaluation | LoRA implementation, evaluation metrics, results analysis |
| **Prisha Singh (B22168)** | Report making  | Statistical Prompt-Aided Sampling design, MI-weighted histogram construction |

---

## References

1. Borisov, V. et al. **Language Models are Realistic Tabular Data Generators.** ICLR 2023.
2. Hu, E. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022.
3. Zhang, Q. et al. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** ICLR 2023.
4. Liu, S. et al. **DoRA: Weight-Decomposed Low-Rank Adaptation.** ICML 2024.
5. Kohavi, R. **Scaling up the accuracy of Naive-Bayes classifiers.** KDD 1996.
6. Jordon, J. et al. **Synthetic data — what, why and how?** arXiv:2205.03257, 2022.

---

## Citation

```bibtex
@inproceedings{borisov2023language,
  title={Language Models are Realistic Tabular Data Generators},
  author={Vadim Borisov and Kathrin Sessler and Tobias Leemann
          and Martin Pawelczyk and Gjergji Kasneci},
  booktitle={ICLR},
  year={2023}
}
```

---

<p align="center">
  IIT Mandi | CS-683 Generative AI | 2025
</p>