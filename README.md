# 🎵 RecLLM Fairness Benchmark — Phi-3 Replication (Experiment 2)

> **Replicating and extending** *"Understanding Biases in ChatGPT-based Recommender Systems"* (Deldjoo, ACM TORS 2025) using the open-source **Microsoft Phi-3-mini-4k-instruct** model.

---

## 📌 Overview

This project replicates **Experiment 2** (Sequential In-Context Learning for music recommendation) from the paper:

> Deldjoo, Y. (2025). *Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency.* ACM Transactions on Recommender Systems, Vol. 4, No. 2, Article 17. [DOI: 10.1145/3690655](https://doi.org/10.1145/3690655)

The original paper used **ChatGPT (GPT-3.5)**. This project swaps in **Phi-3-mini-4k-instruct** — a 3.8B open-source model by Microsoft — to study whether the paper's fairness and accuracy findings generalise across model families and scales.

---

## 🔍 What Was Studied

Given a user's listening history, can an LLM accurately predict the next songs they would listen to? And does its behaviour change depending on:

- How the user's history is sampled (random vs. frequent vs. recent)?
- Whether user demographic information is included in the prompt?
- Whether in-context examples are provided (zero-shot vs. few-shot)?
- Whether counterfactual prompting is applied?

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Dataset | LastFM-1K |
| Users | 80 |
| Artists | 5,500 |
| Listening events | 277,607 |
| Task | Sequential top-3 music recommendation |
| Source | [Official paper repo](https://github.com/yasdel/Benchmark_RecLLM_Fairness) |

---

## ⚙️ Experimental Design

**72 total conditions** from a full factorial design:

| Dimension | Values |
|---|---|
| Counterfactual prompting | True / False |
| History sampling strategy | random, frequent, recent-frequent |
| User demographics in prompt | none, gender, age-group, intersectional |
| ICL interaction type | zero-shot, ICL-1 (1-shot), ICL-2 (2-shot) |

---

## 🤖 Model

| Property | Value |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct` |
| Parameters | 3.8 billion |
| Precision | float16 |
| Hardware | Kaggle GPU (T4) |
| Runtime | ~8 hours |

---

## 📊 Key Results

### Accuracy (HR@all across all 72 conditions)

| Interaction Type | Mean HR@all | vs. Paper (ChatGPT) |
|---|---|---|
| Zero-shot | 0.000108 | 0.204 |
| ICL-1 (1-shot) | 0.000091 | 0.154 |
| ICL-2 (2-shot) | 0.000093 | 0.154 |

| Sampling Strategy | Mean HR@all |
|---|---|
| random | 0.000070 |
| frequent | 0.000073 |
| **recent-frequent** | **0.000149** ✅ best |

### Key Findings

1. **Zero-shot outperforms ICL** — consistent with the paper's result on ChatGPT. This appears to be a model-agnostic property of LLM-based sequential recommendation.

2. **Recent-frequent sampling is best** — providing the most recent listening history as context yields ~2× higher hit rate than random sampling. Replicates the paper's finding on a different model.

3. **Demographics had no effect on Phi-3** — all four demographic conditions produced identical results (HR@all = 0.000097). This differs from ChatGPT where age-group context improved accuracy — a notable cross-model difference.

4. **Model scale gap is significant** — Phi-3 (3.8B) shows substantially lower absolute hit rates than ChatGPT (175B). This is expected and highlights model scale as an important confound in RecLLM fairness research.

---

## 📁 Project Structure

```
├── notebooks/
│   └── phi3_experiment2.ipynb      # Full reproducible experiment notebook
├── src/
│   └── evaluate.py                 # Evaluation logic (HR@1, HR@all)
├── results/
│   ├── phi3_experiment_final.csv   # Raw model outputs (prompts + responses)
│   ├── phi3_experiment_scored.csv  # Evaluated results with HR metrics per user/condition
│   └── phi3_summary_metrics.csv    # Aggregate summary
├── data/
│   └── README.md                   # Data sourcing instructions
└── README.md
```

---

## 🚀 How to Reproduce

### 1. Clone and install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/reclmm-phi3-fairness
cd reclmm-phi3-fairness
pip install transformers accelerate bitsandbytes pandas tqdm torch
```

### 2. Get the data
```bash
git clone https://github.com/yasdel/Benchmark_RecLLM_Fairness.git
cp -r Benchmark_RecLLM_Fairness/data ./data/
```

### 3. Run the experiment
Open `notebooks/phi3_experiment2.ipynb` and run all cells.
A GPU with ≥16GB VRAM is recommended (Kaggle T4 / Colab A100).

---

## 🔭 Next Steps (Planned — Phase 2)

This project is the **baseline** for a steering vector extension:

- Compute steering vectors from Phi-3's internal activations by contrasting "fair" vs "popularity-biased" recommendation prompts
- Apply the vectors at inference (`h' = h + α × v`) to steer recommendations toward fairer item distributions
- Measure the accuracy–fairness trade-off (HR@all vs. Gini Index / Entropy)
- Compare representation-level intervention (steering) vs. prompt-level intervention (the paper's approach)

> 📂 The steering vector extension will be published in a separate repository linked here once complete.

---

## 📎 References

- Deldjoo, Y. (2025). Understanding Biases in ChatGPT-based Recommender Systems. *ACM TORS*. https://doi.org/10.1145/3690655
- Official benchmark repo: https://github.com/yasdel/Benchmark_RecLLM_Fairness
- Phi-3 model: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

---

## 📜 License

This project is for academic research purposes. The experimental framework is based on the original benchmark repo (MIT licensed). Results and code additions are released under MIT.
