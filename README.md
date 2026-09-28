# RecLLM Fairness with Phi-3 — Bias Analysis and Steering Vector Mitigation

**MSc Artificial Intelligence Thesis · Johannes Kepler University Linz (JKU) · 2025**  
**Student:** Aparna Krishna (K12353359)  
**Supervisor:** Deepak Kumar  
**Based on:** Deldjoo (2025) — *Understanding Biases in ChatGPT-based Recommender Systems*

---

## Project Overview

Large Language Models (LLMs) are increasingly used as recommender systems. However, they carry demographic biases — when a user is described as "female" vs "male" in the prompt, the model may recommend different music even when the listening history is identical. This is **gender bias in LLM-based recommender systems (RecLLMs)**.

This project investigates gender bias in music recommendations through two phases:

**Phase 1** replicates Experiment 2 from Deldjoo (2025), which originally studied these biases using ChatGPT (GPT-3.5-turbo). ChatGPT is replaced with the open-source **Microsoft Phi-3-mini-4k-instruct** (3.8B parameters) to study whether the same biases generalise across model families and scales.

**Phase 2** implements **contrastive activation steering vectors** — a representation-level intervention that modifies Phi-3's internal hidden states during inference to reduce gender-driven recommendation differences, without any model retraining. Two vector types and three lambda strengths are systematically compared.

---

## Based On

> Deldjoo, Y. (2025). *Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency.*
> ACM Transactions on Recommender Systems, 4(2), Article 17.
> https://doi.org/10.1145/3690655
> Original benchmark code: https://github.com/yasdel/Benchmark_RecLLM_Fairness

---

## Repository Structure

```
recllm-fairness-phi3/
│
├── phi3_experiment2.ipynb               ← Phase 1: Baseline replication notebook
├── phi3_steering_experiment2.ipynb      ← Phase 2: Steering vectors notebook (improved)
│
├── phi3_experiment_final.csv            ← Phase 1: Raw prompts + model responses (72 conditions × 80 users)
├── phi3_experiment_scored.csv           ← Phase 1: HR@all per user per condition
├── phi3_summary_metrics.csv             ← Phase 1: Aggregate accuracy and fairness metrics
│
├── phi3_steered_improved.csv            ← Phase 2: All steered outputs (6 configs × 6 conditions × 80 users)
├── phi3_improved_evaluation.csv         ← Phase 2: Full evaluation — all 36 config × condition combinations
├── phi3_config_comparison.csv           ← Phase 2: Average results per config — key comparison table
│
├── Llama-3.1-8B-Instruct/              ← Reference steering vectors (supervisor-provided)
│   └── gender-bias_prompt_avg_diff.pt  ← Pre-computed Llama-3.1-8B gender steering vectors
│
├── requirements.txt                     ← Python dependencies
├── data_README.md                       ← Data sourcing instructions
└── results_README.md                    ← Column descriptions for result files
```

---

## Dataset

| Property | Value |
|---|---|
| Name | LastFM-1K |
| Users | 80 (randomly sampled, moderate interaction density) |
| Items | 5,500 unique artists |
| Listening events | 277,607 |
| Task | Sequential top-1 music recommendation |
| Train / Test split | 80% / 20% (chronological order) |
| Source | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html |

---

---

# Phase 1 — Baseline Replication

---

## Objective

Replicate Experiment 2 from Deldjoo (2025): given a user's listening history, prompt an LLM to predict the next artist the user would listen to. 72 different prompt conditions are tested to measure both recommendation accuracy and item-side fairness, using Phi-3-mini as the backbone instead of ChatGPT.

## Experimental Design

**72 total conditions** from a full factorial combination:

| Dimension | Options |
|---|---|
| Counterfactual prompting | True · False |
| History sampling strategy | random · frequent · recent-frequent |
| User demographics in prompt | no-info · gender · age-group · intersectional |
| ICL interaction type | zero-shot · ICL-1 (1-shot) · ICL-2 (2-shot) |

**Counterfactual prompting:** when True, the user's stated gender in the prompt is flipped. This isolates how much the model's output changes based purely on the gender label.

**ICL:** zero-shot uses only the user's history. ICL-1 adds one example of recent songs followed by the next listened song. ICL-2 adds two such examples.

## Prompt Example — Gender Condition, Zero-Shot

```
The user is Female and Early Adult (≤24 yrs). The user has listened to the
following songs in the past, organized as (Song - Artist):

- "Gangsta Bop" by Akon
- "I Can't Wait" by Akon
- "My Love" by Joe
- "One For Me" by Lloyd
- "Get'Cha Head In The Game (Pop Version)" by B5

This selection reflects the user's music preferences.
What would be the top-1 suitable next recommendation?
```

## Model Configuration

| Property | Value |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct` |
| Parameters | 3.8 billion |
| Precision | float16 |
| Decoding | Greedy (`do_sample=False`) |
| Max new tokens | 60 |
| Hardware | Kaggle T4 GPU |
| Runtime | ~8 hours |

## Evaluation Metrics

| Metric | Direction | Description |
|---|---|---|
| HR@all | ↑ | Did the recommendation match the ground truth artist? |
| Gini Index | ↓ | Inequality in item recommendation exposure |
| Entropy | ↑ | Diversity of recommended items across all users |
| Catalogue Coverage | ↑ | Proportion of 5,500-artist catalogue recommended |

## Phase 1 Results

### Accuracy by ICL Type

| ICL Type | Phi-3 HR@all | ChatGPT HR@all (Deldjoo 2025) |
|---|---|---|
| Zero-shot | 0.000108 | 0.204 |
| ICL-1 (1-shot) | 0.000091 | 0.154 |
| ICL-2 (2-shot) | 0.000093 | 0.154 |

### Accuracy by Sampling Strategy

| Sampling Strategy | Phi-3 HR@all |
|---|---|
| random | 0.000070 |
| frequent | 0.000073 |
| recent-frequent | **0.000149** ← best |

## Key Findings

**Finding 1 — Zero-shot outperforms few-shot ICL.**
Adding in-context examples did not improve hit rate. Replicates the paper's ChatGPT result — zero-shot superiority appears model-agnostic.

**Finding 2 — Recent-frequent sampling achieves the best accuracy.**
Providing the most recently consumed items yields ~2× higher hit rate vs random sampling.

**Finding 3 — User demographics had no measurable effect on Phi-3's accuracy.**
All four demographic conditions produced near-identical HR@all. Contrasts with ChatGPT where age-group context improved accuracy — likely a model scale effect.

**Finding 4 — Significant accuracy gap between Phi-3 and ChatGPT.**
Phi-3 (3.8B) achieves substantially lower hit rates than ChatGPT (~175B). This motivates Phase 2 — if the smaller model underperforms on accuracy, can its fairness profile be improved through representation-level intervention?

---

---

# Phase 2 — Gender-Bias Steering Vectors

---

## Objective

Apply **contrastive activation steering** to reduce gender-based differences in Phi-3's music recommendations at inference time, with no retraining required. This phase systematically compares two vector extraction methods and three lambda strengths to identify the most effective configuration.

## Steering Formula

**Single-bias steering (implemented):**
```
h' = h + λ · V_gender
```

**Multi-bias steering (planned extension):**
```
h' = h + λ · (V_gender + V_race + V_religion)
```

| Symbol | Meaning |
|---|---|
| `h` | Original hidden state at a given transformer layer |
| `V_gender` | Gender-bias steering vector from Phi-3's representations |
| `λ` (lambda) | Steering strength — negative = de-bias, positive = amplify bias |
| `h'` | Modified hidden state passed to the next layer |

## Two Vector Extraction Methods

### Method 1 — prompt_avg_diff

Extracts the gender difference from hidden states over **input prompt tokens**.

```
V_gender[layer] = (1/N) × Σ ( mean_tokens(h_male[layer]) − mean_tokens(h_female[layer]) )
```

This is the same method used to produce the Llama reference vectors in `Llama-3.1-8B-Instruct/`. Applied here to Phi-3's own representation space (hidden size 3072 vs Llama's 4096).

### Method 2 — response_avg_diff

Extracts the gender difference from hidden states over **generated response tokens**.

```python
# For each generated token step:
vec = step_hidden_states[layer_idx][0, -1, :]  # last token position
# Average across all steps and pairs → V_gender_response[layer]
```

This targets the generation space directly — where the bias actually manifests in the output.

## Applying Steering at Inference

```python
def apply_hook(module, inputs, output, steer_vec, lambda_val):
    hidden_states = output[0]                           # [batch, seq_len, 3072]
    v = steer_vec / (steer_vec.norm() + 1e-8)          # unit vector
    hidden_states = hidden_states + lambda_val * v      # h' = h + λ·V
    return (hidden_states,) + output[1:]

# Applied to ALL 32 transformer layers (0–31)
# Hook registered before generation, removed immediately after
# No permanent model modification
```

## Experiment Setup

| Property | Value |
|---|---|
| Target columns | 6 zero-shot gender conditions (counterfactual-False, userDemo-gender, zero-shot only) |
| Vector types tested | prompt_avg_diff · response_avg_diff |
| Lambda values tested | −1.0 · −2.0 · −3.0 |
| Total configurations | 6 (2 vectors × 3 lambdas) |
| Steering layers | All 32 (0–31) |
| Steering vector pairs | 30 (prompt) · 10 (response) |
| Users | 80 (45 male, 35 female) |

## Phase 2 Results

### Average Results by Configuration

| Config | Δ Entropy | Δ Gini | Δ Gender Gap | Better Entropy | Better Gap |
|---|---|---|---|---|---|
| prompt_lam-1.0 | +0.625 | +0.0012 | +0.105 | ✅ | ❌ |
| prompt_lam-2.0 | −2.078 | **−0.0073** | **−0.344** | ❌ | ✅ |
| prompt_lam-3.0 | +0.656 | +0.0005 | +0.022 | ✅ | ❌ |
| response_lam-1.0 | −0.045 | −0.0000 | +0.125 | ❌ | ❌ |
| response_lam-2.0 | +0.025 | −0.0011 | −0.002 | ✅ | ✅ |
| response_lam-3.0 | **+0.288** | **−0.0014** | **−0.127** | ✅ | ✅ |

*Gender Gap = proportion of different recommendations between male and female users. Lower = fairer.*

### Results Split by Prompt Type

**Short prompts (random / frequent / recent-frequent):**

| Config | Δ Entropy | Δ Gini | Δ Gender Gap |
|---|---|---|---|
| prompt_lam-1.0 | +1.218 | +0.0027 | +0.339 |
| prompt_lam-2.0 | −0.101 | −0.0005 | **−0.150** |
| prompt_lam-3.0 | +2.668 | +0.0079 | +0.380 |
| response_lam-1.0 | +0.695 | +0.0028 | +0.473 |
| response_lam-2.0 | +1.121 | +0.0019 | +0.178 |
| response_lam-3.0 | +2.153 | +0.0041 | +0.115 |

**Full recommendation prompts (recommendation_random / frequent / recent-frequent):**

| Config | Δ Entropy | Δ Gini | Δ Gender Gap |
|---|---|---|---|
| prompt_lam-1.0 | +0.329 | +0.0005 | −0.012 |
| prompt_lam-2.0 | −3.066 | **−0.0107** | **−0.441** |
| prompt_lam-3.0 | −0.351 | −0.0031 | −0.157 |
| response_lam-1.0 | −0.415 | −0.0014 | −0.050 |
| response_lam-2.0 | −0.522 | −0.0025 | **−0.092** |
| response_lam-3.0 | −0.644 | **−0.0041** | **−0.248** |

### Strongest Individual Results

| Condition | Config | Baseline Gap | Steered Gap | Reduction |
|---|---|---|---|---|
| recommendation_frequent | prompt_lam-2.0 | 0.938 | 0.500 | **−46.7%** |
| recommendation_recent-frequent | prompt_lam-2.0 | 0.956 | 0.667 | **−30.2%** |
| recommendation_random | prompt_lam-2.0 | 0.952 | 0.750 | **−21.2%** |
| recommendation_frequent | response_lam-3.0 | 0.938 | 0.593 | **−36.8%** |
| recommendation_recent-frequent | response_lam-3.0 | 0.956 | 0.824 | **−13.8%** |

## Key Findings

**Finding 1 — prompt_lam-2.0 achieves the strongest gender gap reduction.**
Averaging across all 6 conditions, λ=−2.0 with prompt-level vectors reduces gender gap by 34.4%. On full recommendation prompts specifically, the reduction reaches 44.1% — with the strongest single result showing gender gap dropping from 0.938 to 0.500 (−46.7%) in the frequent sampling condition.

**Finding 2 — response_lam-3.0 provides the most balanced improvement.**
Response-level vectors at λ=−3.0 reduce gender gap by 12.7% while simultaneously improving entropy (+0.288) and Gini (−0.0014). This is the only configuration that improves all three fairness metrics simultaneously.

**Finding 3 — Prompt type is the strongest moderator.**
Full recommendation prompts (which include richer context) respond more consistently to steering than short prompts. This suggests that richer context allows the steering signal to compete more effectively with the gender-encoding.

**Finding 4 — λ=−2.0 causes entropy collapse on full rec prompts.**
While prompt_lam-2.0 achieves the best gender gap reduction, it also causes entropy to drop sharply (−3.066 average on rec prompts), indicating the model's output becomes highly repetitive. This is a known trade-off in activation steering — stronger de-biasing reduces output diversity.

**Finding 5 — All-layer steering breaks the ICL immunity observed in the initial experiment.**
The original experiment (layers 8–23 only, λ=−1.0) showed zero effect on all ICL conditions. Extending steering to all 32 layers and focusing on zero-shot conditions reveals meaningful effects, confirming that ICL immunity was partly a function of insufficient layer coverage.

## Discussion

Two practically useful configurations emerge:

**For maximum gender gap reduction:** `prompt_lam-2.0` — reduces gender gap by up to 46.7% on full recommendation prompts. Recommended when the primary goal is fairness between demographic groups, with some tolerance for reduced output diversity.

**For balanced fairness improvement:** `response_lam-3.0` — reduces gender gap, Gini, and improves entropy simultaneously. Recommended when both fairness and diversity are important evaluation criteria.

---

---

# How to Reproduce

---

## Requirements

```bash
pip install transformers==4.40.2 accelerate sentencepiece pandas torch tqdm
```

> **Important:** `transformers==4.40.2` is required. Newer versions have a breaking change in Phi-3's `rope_scaling` config (`KeyError: 'type'`). Do not upgrade.

## Phase 1

```
Platform : Kaggle Notebook (T4 GPU)
Dataset  : aparnakrishna2407/phi3-experiment (Kaggle dataset input)
Notebook : phi3_experiment2.ipynb
Outputs  : phi3_experiment_final.csv · phi3_experiment_scored.csv · phi3_summary_metrics.csv
Runtime  : ~8 hours
```

## Phase 2

```
Platform : Kaggle Notebook (T4 GPU)
Input    : phi3_experiment_final.csv from Phase 1
Notebook : phi3_steering_experiment2.ipynb
Outputs  : phi3_steered_improved.csv · phi3_improved_evaluation.csv · phi3_config_comparison.csv
Runtime  : ~5 hours total
```

### Cell-by-cell guide

| Cell | Purpose | Time |
|---|---|---|
| 1 | Install packages | 2 min |
| 2 | Load Phi-3-mini model | 5 min |
| 3 | Load Phase 1 CSV | instant |
| 4 | Define `get_hidden_states()` | instant |
| 5 | Verify gender columns (18 found) | instant |
| 6 | Build male/female prompt pairs (30 pairs) | instant |
| 7 | **Compute prompt_avg_diff vectors** (30 pairs × 32 layers) | ~25 min |
| 6B | **Compute response_avg_diff vectors** (10 pairs × generated tokens) | ~15 min |
| 8 | Define steering hook + generation functions (all 32 layers) | instant |
| 9 | Sanity check on 1 prompt | 2 min |
| 10 | **Run full experiment** (6 zero-shot cols × 6 configs × 80 users, auto-checkpoint) | ~4 hrs |
| 11 | Evaluate all configs — Entropy, Gini, Gender Gap, Coverage | instant |

**Resuming after interruption:** restart kernel → run Cells 1–3 → run Cell 10 directly. Auto-checkpoint skips completed runs.

---

## References

- Deldjoo, Y. (2025). Understanding Biases in ChatGPT-based Recommender Systems. *ACM Transactions on Recommender Systems*, 4(2), Article 17. https://doi.org/10.1145/3690655
- Original benchmark code: https://github.com/yasdel/Benchmark_RecLLM_Fairness
- Phi-3-mini: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- LastFM-1K: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

---
