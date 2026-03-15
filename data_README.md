# Data

The raw LastFM-1K dataset and pre-processed splits used in this experiment come from the official paper repository:

**Source:** https://github.com/yasdel/Benchmark_RecLLM_Fairness/tree/main/data

## Files Used

| File | Description |
|---|---|
| `test_data_lastFM1k_fullInteraction_80users.csv` | Test split — 80 users |
| `train_data_lastFM1k_fullInteraction_80users_part_*.csv` | Training interaction history (4 parts) |
| `train_prompt_df_lastFM1k_fullInteraction_80users_nTrack10_lContext5_top_3.csv` | Pre-built prompt templates |

## How to Get the Data

```bash
git clone https://github.com/yasdel/Benchmark_RecLLM_Fairness.git
cp Benchmark_RecLLM_Fairness/data/test_data_lastFM1k_fullInteraction_80users.csv ./
cp Benchmark_RecLLM_Fairness/data/train_data_lastFM1k_* ./
cp Benchmark_RecLLM_Fairness/data/train_prompt_df_lastFM1k_fullInteraction_80users_nTrack10_lContext5_top_3.csv ./
```

> The raw LastFM-1K dataset is originally from: http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html
