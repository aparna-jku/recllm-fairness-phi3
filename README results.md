# Results

## Files

| File | Description | Committed |
|---|---|---|
| `phi3_summary_metrics.csv` | Overall mean HR@1 and HR@all across all conditions | ✅ Yes |
| `phi3_experiment_scored.csv` | Per-user HR@1 and HR@all for all 72 conditions | ✅ Yes |
| `phi3_experiment_final.csv` | Raw prompt text + model responses for all users/conditions | ⬇️ Download (large) |

## Summary

| Metric | Value |
|---|---|
| HR@1 (mean, all conditions) | 0.000000 |
| HR@all (mean, all conditions) | 0.000097 |
| Best HR@all (single condition) | 0.000183 |
| Best condition | counterfact-True + recent-frequent + ICL-1 |

## Reproducing

Run `python src/evaluate.py results/phi3_experiment_scored.csv` to regenerate the summary from the scored CSV.
