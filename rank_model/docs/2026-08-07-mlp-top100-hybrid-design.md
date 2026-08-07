# Top100-Aware Hybrid MLP Design

## Scope

Add one model, `mlp_top100_hybrid_rank`, to the existing 10-day rank-model
pipeline. Keep `mlp_rank_regression`, `mlp_pairwise_rank`, their run bundles,
the prepared dataset, and the evaluation contract unchanged.

The new model is a development-stage candidate selected only with the 2023
validation period. It must not read or derive any signal from the locked
2024-2025 test period.

## Objective

The primary selection metric is validation `NDCG@100`. Validation Rank IC is
recorded alongside it so that a gain at the top of the ranking is not mistaken
for broad cross-sectional improvement.

The training loss for a batch of complete date groups is:

```text
loss = mean_date_rank_mse + 0.1 * mean_date_top100_pairwise_logistic
```

The rank-MSE term uses every finite training row and preserves a broad ranking
signal. The pairwise term compares stocks in each date's realized training
Top100 with stocks below that cutoff. It is training-label logic only; future
returns are never features.

## Architecture And Optimization

- Input width comes from the persisted feature schema.
- Hidden widths remain `[128, 64, 32]` with ReLU and dropout `0.1`.
- AdamW uses learning rate `0.0003` and weight decay `0.0001`.
- Gradient norm is clipped at `1.0`.
- A batch contains eight complete, contiguous trading-date groups.
- The optimizer steps after every date batch. This avoids the existing
  pairwise baseline's one-update-per-epoch behavior.
- Training is deterministic with seed `42` and deterministic PyTorch
  algorithms.
- Maximum epochs are `30`; early-stopping patience is `4` epochs.

## Top100 Pair Sampling

For each training date, `rank_target_10d >= 0.9` defines the positive group.
Each positive stock receives an equal fixed number of opponents:

- four are sampled from the boundary band
  `0.7 <= rank_target_10d < 0.9`;
- four are sampled from the broader lower-ranked set
  `rank_target_10d < 0.9`.

Sampling is with replacement when required. It never crosses dates, never
creates self-pairs, and never creates equal-target pairs. Sampling changes by
epoch using `seed + epoch`, which expands coverage while preserving exact
reproducibility.

Within each date, the MSE and pairwise components are each reduced to one date
mean before dates are averaged. A date therefore cannot dominate merely
because it has more valid pairs.

## Validation And Early Stopping

At the end of each epoch, the model scores all 2023 validation rows. The
pipeline calculates mean daily `NDCG@100` from `rank_target_10d`; a strict
improvement greater than `0.0001` saves a CPU copy of the model state. The
NDCG gain is `exp2(rank_target_10d) - 1`, matching the existing evaluator.
Rank IC is calculated and recorded but does not override the primary selection
rule.

Training stops after four consecutive non-improving epochs or after epoch 30.
The best state, not the last state, is restored before validation predictions
and model persistence. The manifest records best epoch, stopped epoch, best
NDCG@100, Rank IC at the best epoch, and per-epoch training history.

## Configuration And Compatibility

The model is registered through the existing model registry and receives one
configuration table in `rank_model/config.toml`. Parameters are explicit so a
later, bounded 2023-only tuning exercise does not require another Python file.

The standard commands remain valid:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml train `
  --model mlp_top100_hybrid_rank --run-id real-2023-mlp-top100-hybrid

& $python -m rank_model.pipeline --config rank_model\config.toml evaluate `
  --run-id real-2023-mlp-top100-hybrid --split validation
```

The model uses the existing PyTorch persistence path. The saved architecture
identifies the new model so reload reconstructs the correct network. The
prediction parquet and all evaluation outputs retain their current schemas.

## Validation And Failure Handling

Training must fail clearly when:

- the training or validation frame is empty;
- rank labels are non-finite after the standard finite-label filter;
- a date has no Top100 or no eligible lower-ranked opponents;
- date batches are not contiguous or pair indices cross date groups;
- validation scores or selection metrics are non-finite;
- a reloaded model differs from pre-save predictions beyond the existing MLP
  tolerance.

## Tests

Implementation follows red-green-refactor tests for:

- registry and configuration acceptance;
- deterministic Top100 pair sampling and no cross-date pairs;
- equal-date hybrid loss behavior;
- one optimizer update per date batch;
- NDCG-based best-epoch restoration and patience stopping;
- persistence and reload prediction equality;
- unchanged prediction and evaluation schemas.

Temporary generated run directories and test data are removed after
verification. No real run result is overwritten.

## Acceptance Criteria

- Existing seven model names and run bundles remain usable.
- The new model trains, persists, reloads, evaluates, and compares through the
  existing CLI.
- Repeated runs with identical inputs and seed produce matching predictions.
- The manifest contains enough detail to reproduce the selected epoch.
- Only 2019-2022 rows update model weights; only 2023 rows select the epoch;
  2024-2025 remain untouched.
