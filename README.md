# RecLLM Fairness with Phi-3 — Bias Analysis and Steering Vector Mitigation

**MSc Artificial Intelligence Thesis · Johannes Kepler University Linz (JKU) · 2025**  
**Student:** Aparna Krishna  
**Supervisor:** Deepak Kumar  
**Based on:** Deldjoo (2025) — *Understanding Biases in ChatGPT-based Recommender Systems*

---

## Project Overview

Large Language Models (LLMs) are increasingly used as recommender systems. However, they carry demographic biases — when a user is described as "female" vs "male" in the prompt, the model may recommend different music even when the listening history is identical. This is **gender bias in LLM-based recommender systems (RecLLMs)**.

This project investigates gender bias in music recommendations through two phases:

**Phase 1** replicates Experiment 2 from Deldjoo (2025), which originally studied these biases using ChatGPT (GPT-3.5-turbo). ChatGPT is replaced with the open-source **Microsoft Phi-3-mini-4k-instruct** (3.8B parameters) to study whether the same biases generalise across model families and scales.

**Phase 2** goes beyond measurement and implements **contrastive activation steering vectors** — a representation-level intervention that modifies Phi-3's internal hidden states during inference to reduce gender-driven recommendation differences, without any model retraining.

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
├── phi3_steering_experiment2.ipynb      ← Phase 2: Steering vectors notebook
│
├── phi3_experiment_final.csv            ← Phase 1: Raw prompts + model responses (72 conditions × 80 users)
├── phi3_experiment_scored.csv           ← Phase 1: HR@all per user per condition
├── phi3_summary_metrics.csv             ← Phase 1: Aggregate accuracy and fairness metrics
├── phi3_steering_evaluation.csv         ← Phase 2: Baseline vs steered results (18 conditions)
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

Preprocessing follows the original paper — interactions are sorted chronologically per user and the final 20% is held out as the test set.

---

---

# Phase 1 — Baseline Replication

---

## Objective

Replicate Experiment 2 from Deldjoo (2025): given a user's listening history, prompt an LLM to predict the next artist/song the user would listen to. 72 different prompt conditions are tested to measure both recommendation accuracy and item-side fairness, using Phi-3-mini as the backbone instead of ChatGPT.

## Experimental Design

**72 total conditions** from a full factorial combination:

| Dimension | Options |
|---|---|
| Counterfactual prompting | True · False |
| History sampling strategy | random · frequent · recent-frequent |
| User demographics in prompt | no-info · gender · age-group · intersectional |
| ICL interaction type | zero-shot · ICL-1 (1-shot) · ICL-2 (2-shot) |

**Counterfactual prompting:** when True, the user's stated gender in the prompt is flipped — a female user is described as male and vice versa. This isolates how much the model's output changes based purely on the gender label rather than listening history.

**ICL (In-Context Learning):** zero-shot uses only the user's history. ICL-1 adds one example of recent songs followed by the next listened song. ICL-2 adds two such examples.

**Demographics:** prompts optionally include phrases like `"The user is female"` or `"The user is young"` to examine how demographic disclosure influences recommendations.

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
| Decoding strategy | Greedy (`do_sample=False`) |
| Max new tokens | 60 |
| Hardware | Kaggle T4 GPU |
| Total runtime | ~8 hours (72 conditions × 80 users) |

## Evaluation Metrics

| Metric | Direction | Description |
|---|---|---|
| HR@all | ↑ higher is better | Did the recommendation match the ground truth artist? |
| Gini Index | ↓ lower is fairer | Inequality in item recommendation exposure |
| Entropy | ↑ higher is more diverse | Diversity of items recommended across all users |
| Catalogue Coverage | ↑ higher is broader | Proportion of 5,500-artist catalogue that was recommended |

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
Adding in-context examples did not improve hit rate. This replicates the paper's ChatGPT result, suggesting zero-shot superiority in sequential recommendation is model-agnostic.

**Finding 2 — Recent-frequent sampling achieves the best accuracy.**  
Providing the model with the user's most recently consumed items yields approximately 2× higher hit rate compared to random sampling, consistent with the original paper.

**Finding 3 — User demographics had no measurable effect on Phi-3's accuracy.**  
All four demographic conditions (no-info, gender, age-group, intersectional) produced near-identical HR@all values. This contrasts with ChatGPT where age-group context improved accuracy, suggesting this effect is tied to model scale or training data.

**Finding 4 — Significant accuracy gap between Phi-3 and ChatGPT.**  
Phi-3-mini (3.8B) achieves substantially lower hit rates than ChatGPT (estimated ~175B parameters). This scale gap is expected and motivates Phase 2: if the smaller model underperforms on accuracy, can its fairness profile be improved through intervention?

---

---

# Phase 2 — Gender-Bias Steering Vectors

---

## Objective

Apply **contrastive activation steering** to reduce gender-based differences in Phi-3's music recommendations at inference time, with no retraining required.

When Phi-3 processes "The user is female" vs "The user is male" — with an identical listening history — the internal transformer representations differ at every layer. These differences encode how the model responds to gender. The steering vector approach extracts and counteracts this encoding during generation.

## Steering Formula

**Single-bias steering (implemented in this project):**
```
h' = h + λ · V_gender
```

**Multi-bias steering (planned future extension):**
```
h' = h + λ · (V_gender + V_race + V_religion)
```

| Symbol | Meaning |
|---|---|
| `h` | Original hidden state at a given transformer layer |
| `V_gender` | Gender-bias steering vector computed from Phi-3's representations |
| `λ` (lambda) | Steering strength — negative de-biases, positive amplifies bias |
| `h'` | Modified hidden state passed to the next transformer layer |

## Computing the Steering Vector

The steering vectors in the `Llama-3.1-8B-Instruct/` folder are supervisor-provided reference vectors computed on Llama-3.1-8B (hidden size 4096). These cannot be directly applied to Phi-3 (hidden size 3072) — the representation spaces are different models entirely.

Phi-3-specific gender steering vectors are computed from scratch using the **prompt_avg_diff** method, which is the same method used to produce the Llama reference vectors:

**Step 1.** Select 30 prompts from the gender-condition columns of the Phase 1 CSV.

**Step 2.** For each prompt, produce two versions — one with "male" and one with "female" as the stated gender — keeping the entire listening history identical.

**Step 3.** Run both versions through Phi-3 with `output_hidden_states=True` to capture internal activations at every layer.

**Step 4.** At each of the 32 transformer layers, compute the average difference:

```
V_gender[layer] = (1/N) × Σ ( mean_over_tokens(h_male[layer]) − mean_over_tokens(h_female[layer]) )
```

**Step 5.** Output: 32 steering vectors, one per layer, each of shape `[3072]`, saved as `phi3_gender-bias_prompt_avg_diff.pt`.

## Applying Steering at Inference

Steering is applied using **PyTorch forward hooks** — lightweight interceptors that modify the model's internal computation mid-pass without any permanent change to model weights:

```python
def apply_hook(module, inputs, output, steer_vec, lambda_val):
    hidden_states = output[0]                           # shape: [batch, seq_len, 3072]
    v = steer_vec / (steer_vec.norm() + 1e-8)          # normalize to unit vector
    hidden_states = hidden_states + lambda_val * v      # h' = h + λ·V_gender
    return (hidden_states,) + output[1:]

# Applied to transformer layers 8–23 (middle layers handle semantic content)
# Hook is registered before generation and removed immediately after each call
# No permanent modification to model weights
```

## Experiment Setup

| Property | Value |
|---|---|
| Target columns | 18 gender conditions (counterfactual-False, userDemo-gender, all ICL × sampling combinations) |
| Lambda (λ) | −1.0 |
| Steering layers | 8 – 23 out of 32 |
| Vector type | prompt_avg_diff |
| Steering vector pairs used | 30 |
| Users evaluated | 80 (45 male, 35 female) |
| Checkpoint | Auto-saves after every column — fully resumable on session restart |

## Phase 2 Results

### Summary Averages Across All 18 Gender Conditions

| Metric | Baseline | Steered (λ = −1.0) | Δ |
|---|---|---|---|
| Gini Index ↓ | −0.9867 | −0.9870 | −0.0003 |
| Entropy ↑ | 3.897 | 3.877 | −0.019 |
| Catalogue Coverage | 0.01248 | 0.01227 | −0.00021 |

### Results by Condition

| Prompt Type | Sampling | ICL Type | Baseline Entropy | Steered Entropy | Δ Entropy | Baseline Gini | Steered Gini | Δ Gini |
|---|---|---|---|---|---|---|---|---|
| short prompt | random | zero-shot | 1.662 | 1.361 | −0.301 | −0.99300 | −0.99595 | −0.00295 |
| short prompt | frequent | zero-shot | 1.918 | 1.425 | −0.493 | −0.99279 | −0.99563 | −0.00284 |
| short prompt | recent-frequent | zero-shot | 2.073 | 2.655 | **+0.582** | −0.99152 | −0.99052 | +0.001 |
| short prompt | random | ICL-1 | 4.382 | 4.382 | 0.000 | −0.98545 | −0.98545 | 0.000 |
| short prompt | random | ICL-2 | 4.382 | 4.382 | 0.000 | −0.98545 | −0.98545 | 0.000 |
| full rec prompt | random | zero-shot | 3.857 | 3.757 | −0.100 | −0.98600 | −0.98647 | −0.00046 |
| full rec prompt | frequent | zero-shot | 4.035 | 3.923 | −0.112 | −0.98576 | −0.98602 | −0.00026 |
| full rec prompt | recent-frequent | zero-shot | 4.118 | 4.157 | +0.039 | −0.98566 | −0.98562 | +0.00004 |
| full rec prompt | frequent | ICL-1 | 4.382 | 4.382 | 0.000 | −0.98545 | −0.98545 | 0.000 |
| full rec prompt | recent-frequent | ICL-1 | 4.382 | 4.382 | 0.000 | −0.98545 | −0.98545 | 0.000 |
| full rec prompt | random | ICL-2 | 4.365 | 4.365 | 0.000 | −0.98546 | −0.98546 | 0.000 |
| full rec prompt | frequent | ICL-2 | 4.313 | 4.330 | +0.017 | −0.98553 | −0.98550 | +0.00003 |

*Full results for all 18 conditions are in `phi3_steering_evaluation.csv`.*

## Key Findings

**Finding 1 — ICL conditions show zero response to steering.**  
All 12 ICL-1 and ICL-2 conditions show Δ Entropy = 0.000 and Δ Gini = 0.000. The few-shot examples embedded in the prompt dominate the model's generation so completely that representation-level modification at layers 8–23 has no measurable effect. This is a significant finding — it suggests that prompt-level information overrides activation-level steering in Phi-3 when in-context examples are present.

**Finding 2 — Steering effect is mixed in zero-shot conditions.**  
Among the 6 zero-shot conditions, results are inconsistent. The short prompt with recent-frequent sampling shows a positive entropy improvement (+0.582), while the other zero-shot conditions show small negative or negligible deltas. This suggests the steering vector's effectiveness depends on how strongly the prompt already constrains the model's output.

**Finding 3 — Gini index worsens slightly in short-prompt zero-shot conditions.**  
The two short-prompt zero-shot conditions with random and frequent sampling show increased Gini (−0.003), indicating slightly more concentrated item recommendations after steering — the opposite of the intended fairness direction.

**Finding 4 — Overall effect is limited at λ = −1.0.**  
Averaged across all 18 conditions, entropy decreases by −0.019 and Gini decreases by −0.0003. The λ = −1.0 steering strength is insufficient to produce consistent improvements across condition types when ICL-dominated conditions are included in the average.

## Discussion

The results indicate that gender bias mitigation via contrastive activation steering is effective only in specific settings — zero-shot prompts where the model has more generation freedom. When few-shot examples are present, the ICL signal completely overrides the steering intervention. This is an important distinction for future work on fairness interventions in RecLLMs.

**Possible improvements:**

| Direction | Rationale |
|---|---|
| Stronger lambda (λ = −2.0, −3.0) | May produce larger effects in zero-shot conditions |
| Response-level vector extraction | Computing V_gender from generated tokens rather than prompt tokens targets the generation space more directly |
| Multi-layer steering across all 32 layers | Currently limited to layers 8–23; full-depth steering may break ICL dominance |
| Multi-bias steering (V_gender + V_age) | Extension of the whiteboard formula to combine multiple demographic vectors |

---

---

# How to Reproduce

---

## Requirements

```bash
pip install transformers==4.40.2 accelerate sentencepiece pandas torch tqdm
```

> **Important:** `transformers==4.40.2` is required. Newer versions introduced a breaking change in Phi-3's `rope_scaling` configuration that causes `KeyError: 'type'` on model load. Do not upgrade.

## Phase 1

```
Platform : Kaggle Notebook (T4 GPU)
Dataset  : aparnakrishna2407/phi3-experiment (add as Kaggle dataset input)
Notebook : phi3_experiment2.ipynb
Outputs  : phi3_experiment_final.csv · phi3_experiment_scored.csv · phi3_summary_metrics.csv
Runtime  : ~8 hours
```

Run all cells top to bottom. The notebook generates all 72 prompt conditions, runs Phi-3 inference on 80 users, and evaluates HR@all per condition.

## Phase 2

```
Platform : Kaggle Notebook (T4 GPU)
Input    : phi3_experiment_final.csv from Phase 1
Notebook : phi3_steering_experiment2.ipynb
Outputs  : phi3_gender-bias_prompt_avg_diff.pt · phi3_steered_experiment2.csv · phi3_steering_evaluation.csv
Runtime  : ~7 hours total
```

### Cell-by-cell guide

| Cell | Purpose | Estimated time |
|---|---|---|
| 1 | Install packages (`transformers==4.40.2`) | 2 min |
| 2 | Load Phi-3-mini model | 5 min |
| 3 | Load Phase 1 CSV | instant |
| 4 | Define `get_hidden_states()` function | instant |
| 5 | Build male/female contrastive prompt pairs (30 pairs) | instant |
| 6 | **Compute V_gender** — extract hidden states, compute per-layer difference | ~25 min |
| 7 | Define steering hook and generation functions | instant |
| 8 | Sanity check — run baseline vs steered on 1 prompt | 2 min |
| 9 | **Run full experiment** — 18 gender columns × 80 users, auto-checkpoint per column | ~6 hours |
| 10 | Evaluate — compute Gini, Entropy, Coverage for baseline vs steered | instant |

**Resuming after interruption:** if the session dies during Cell 9, restart the kernel, run Cells 1–3 to reload the model and data, then run Cell 9 directly — it reads the checkpoint file and skips already-completed columns automatically.

---

## References

- Deldjoo, Y. (2025). Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency. *ACM Transactions on Recommender Systems*, 4(2), Article 17. https://doi.org/10.1145/3690655
- Original benchmark code: https://github.com/yasdel/Benchmark_RecLLM_Fairness
- Phi-3-mini model: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- LastFM-1K dataset: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

---

## License

This project is for academic research purposes. The experimental framework is based on the original benchmark repository (MIT licensed). All code additions and results are released under MIT.

---

---

# Planned Improvements and Future Work

---

## Why the Current Results Are Limited

The current experiment uses a single lambda value (λ = −1.0) with steering applied to middle layers (8–23) only. Three factors constrain the results:

1. **ICL dominance** — when few-shot examples are present in the prompt, they override the steering signal entirely. 12 of the 18 conditions are ICL conditions, which all show Δ = 0. This brings the average improvement down significantly.

2. **Lambda strength** — λ = −1.0 is a conservative starting point. The steering vector may need a stronger push to meaningfully shift the model's output distribution.

3. **Prompt-level vectors** — the current V_gender is computed from hidden states over prompt tokens. The bias, however, manifests in what the model *generates*, not what it reads. Vectors computed over response tokens may be more directly effective.

## Concrete Next Steps

### Step 1 — Lambda sweep on zero-shot conditions (highest priority)

Since ICL conditions show zero response to steering regardless of lambda, focus the lambda sweep on the 6 zero-shot conditions where the effect is visible.

In `phi3_steering_experiment2.ipynb`, change Cell 9 as follows:

```python
# Change this line in Cell 9:
MAIN_LAMBDA = -1.0

# To test stronger de-biasing:
MAIN_LAMBDA = -2.0   # try this first
# or
MAIN_LAMBDA = -3.0   # if -2.0 still shows limited effect
```

And filter to zero-shot columns only:

```python
# Replace the gender_cols_to_steer filter with:
gender_cols_to_steer = [
    col for col in prompt_df.columns
    if "userDemo-gender" in col
    and "counterfact-False" in col
    and "zero-shot" in col          # zero-shot only
]
print(f"Zero-shot gender columns: {len(gender_cols_to_steer)}")  # should be 6
```

This reduces runtime from ~6 hours to ~2 hours and focuses on the conditions where improvement is achievable.

---

### Step 2 — Steer all 32 layers

Currently only layers 8–23 are steered. Extending to all 32 layers may be enough to overcome ICL dominance.

In Cell 7, change:

```python
# Current:
STEER_LAYERS = list(range(8, 24))   # layers 8-23

# Change to:
STEER_LAYERS = list(range(0, 32))   # all 32 layers
```

Run with λ = −1.0 first on zero-shot conditions to isolate the effect of layer coverage before combining with a lambda change.

---

### Step 3 — Compute response-level steering vectors

The current V_gender is a `prompt_avg_diff` vector — the difference in hidden states over the input tokens. An alternative is `response_avg_diff` — the difference in hidden states over the *generated* tokens. This targets the generation space more directly.

Add this cell after Cell 6 in the notebook:

```python
# CELL 6B — Compute response-level steering vectors (response_avg_diff)

response_diffs_per_layer = defaultdict(list)

print("Extracting response-level hidden states...")
for male_p, female_p in tqdm(prompt_pairs[:10], desc="Response pairs"):  # use 10 pairs (slower)
    for gender_prompt, sign in [(male_p, +1), (female_p, -1)]:
        inputs = tokenizer(gender_prompt, return_tensors="pt",
                          truncation=True, max_length=512).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )

        # gen_out.hidden_states: tuple of length max_new_tokens
        # each element: tuple of (num_layers+1) tensors of shape [batch, 1, hidden_size]
        for step_hs in gen_out.hidden_states:
            for layer_idx in range(NUM_LAYERS):
                vec = step_hs[layer_idx + 1].squeeze().cpu().float()  # [3072]
                response_diffs_per_layer[layer_idx].append(vec * sign)

# Average across all steps and pairs
phi3_sv_gender_response = {
    l: torch.stack(v).mean(dim=0)
    for l, v in response_diffs_per_layer.items()
}

# Save
torch.save(phi3_sv_gender_response,
           "/kaggle/working/phi3_gender-bias_response_avg_diff.pt")
print(f"Response-level vectors saved. Shape: {phi3_sv_gender_response[0].shape}")
```

Then in Cell 9, switch to using this vector:

```python
# In generate_steered(), change sv_dict to use response vectors:
response = generate_steered(
    str(prompt),
    lam=MAIN_LAMBDA,
    sv_dict=phi3_sv_gender_response   # use response-level vector
)
```

---

### Step 4 — Multi-bias steering (longer term)

Based on the full formula:

```
h' = h + λ · (V_gender + V_race + V_religion)
```

Once V_gender improvements are validated, the same extraction pipeline can be applied to age-group and intersectional conditions to compute V_age and V_intersectional, and all three can be combined in a single steering pass.

---

## Expected Impact of Each Improvement

| Improvement | Effort | Expected impact | Conditions affected |
|---|---|---|---|
| λ = −2.0 on zero-shot | Low (1 param change, 2 hrs) | Moderate — stronger push in conditions already showing effect | 6 zero-shot columns |
| Steer all 32 layers | Low (1 param change, 6 hrs) | Potentially breaks ICL dominance | All 18 columns |
| Response-level vectors | Medium (1 extra cell, 6 hrs) | Higher — targets generation space directly | All 18 columns |
| Multi-bias steering | High (new pipeline) | Broader fairness coverage | All demographic conditions |

> **Note:** Running the same experiment with identical settings will not change the results. Phi-3 uses greedy decoding (`do_sample=False`), which is fully deterministic — the same prompt always produces the same output. Meaningful improvement requires one of the changes above.
