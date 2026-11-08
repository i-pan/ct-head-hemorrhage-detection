# SKP evidence sources

Discover the active run from the conversation, project experiment history, and live processes. Paths and metric names vary across downstream projects.

## Locate records

- Standard artifacts live under `experiments/<config>/<run_id>/`, unless `cfg.save_dir` overrides the root. Console logs default to `<split>/training.log`; inspect `save_console_log` and `console_log_filename`. Launchers may also write logs or `queue.json` elsewhere. Verify queue status against the actual service/process; a waiting controller may not own the training process tree.
- `config.pkl` at the run root records the effective configuration after completion. During training use the launch command, source config and overrides. Load only trusted local pickle files if needed; never instantiate models for monitoring.
- Some downstream projects write scheduler audits or epoch/step/LR JSONL files. Use them when present, but do not assume a particular sampler, epoch size, or audit filename is part of the template.
- Use tracker metadata or logs to identify the best checkpoint's epoch rather than loading full model checkpoints.

## Metrics

Resolve the existing configured MLflow backend from effective config/environment without exposing credentials. The template normally requires a reachable tracking server. For HTTP backends, use read-only MLflow client queries (`search_runs`, `get_run`, `get_metric_history`), constrained to the exact experiment/run. Do not create runs, change tracking configuration, start a server, or enable local fallback for monitoring. If unavailable, report it and use existing logs.

For an explicitly local SQLite tracking store in a downstream project, open a read-only connection (`sqlite3.connect('file:/absolute/path?mode=ro', uri=True)`). Inspect schema only when unknown; reuse established table/key conventions for fast updates. Typical tables are `runs`, `tags`, `params`, `metrics`, and `latest_metrics`. `mlflow.runName` uses `<config_name>/<run_id>`. Do not select a run solely by its latest timestamp.

`cfg.val_metric` and `cfg.val_track` determine primary metric and optimization direction. Verify the actual tracker key; do not assume classification, a particular target class, or a pooling method. Keep slice/example-level and series/volume-level metrics distinct. Retrieve metric history for trends and matched-budget comparisons; latest metrics alone lose previous evaluations.

SKP logs `training/epoch` using zero-based epochs at `global_step`. Verify a validation evaluation's step convention against the training log before deriving completed epochs. With S optimizer steps per epoch, an evaluation after k complete epochs may be logged at k*S or k*S-1 depending on logger conventions. Avoid timestamp-only joins across different evaluations.

## Timing and comparisons

Exclude Lightning sanity validation and partial-batch validation from full epoch results. Validation may cover the full dataset even when training epochs sample only a subset. Use effective sampler length, world size, batch size, accumulation, and training limits to establish the optimizer-step budget. DDP ranks execute in parallel; do not multiply iteration duration by GPU count.

A progress bar labeled `Epoch 2` is the third epoch. At validation cadence five, the next scheduled evaluation follows completed epoch five, not epoch three. Epoch-bar elapsed time may already include validation/checkpoint time; avoid counting it twice.

Discover related ablation arms from queue records, protocol notes, and conversation context, including companion queues and reused completed baselines. Align completed validation epochs or cumulative steps, not row positions or run age. A new arm at epoch 10 should be compared to the baseline at epoch 10; baseline best/final performance is separate context.

For 50/75/100-epoch cosine-schedule ablations, equal epochs can mean equal data exposure but different LR and warmup phases. Report those distinctions and final results at their stated budgets. Do not treat equal fractions of the schedule as equal compute. Confirm the metric definition and evaluation cohort are shared before interpreting deltas.

## Checkpoint-specific prediction archives

For updated exporters, identify `validation_predictions_best.npz` and
`validation_predictions_final.npz` separately. Final refers to the latest saved
checkpoint; verify epoch/step against the actual endpoint. SKPv2 embeds checkpoint
provenance in NPZ metadata; ICH also writes JSON sidecars with checkpoint and archive
hashes. A legacy `validation_predictions.npz` may contain best-checkpoint predictions
only. Never infer final provenance from that ambiguous filename. Report missing
exports accurately and do not launch backfill from a monitoring-only request.
