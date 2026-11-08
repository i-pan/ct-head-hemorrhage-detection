---
name: training-monitor
description: Summarize active or completed ML training when asked for a progress update, training status, latest epoch results, performance trends, resource usage, or ETA. Read logs and experiment records to report metrics, iteration/epoch and validation timing, GPU memory, RAM, and time to the next result and training completion. Monitoring alone does not launch or alter training.
---

# Training Monitor

Give a fresh, concise assessment grounded in the current run's evidence. Include the latest completed validation results, trends, a brief interpretation, training/validation timing, GPU memory, RAM, and ETAs. Read-only monitoring is the default; an update request does not authorize restarting, stopping, changing, or adding jobs, inference, or periodic monitoring.

## Fast updates

Use fast mode when the user asks for a fast/quick/brief update, including `$training-monitor fast`. Otherwise retain the full report. Fast mode changes the depth of evidence gathering, not which metric or ablation comparison matters.

Reuse the known active run, paths, metric keys, and logger conventions from the conversation; verify the current queue and process/service state once. If the active controller merely waits for a separately owned training process, check the recorded training launcher PID too. Read only the latest 1–2 completed validation evaluations and the baseline evaluation at the latest shared epoch. Query the known local metrics store read-only with bounded results; avoid dumping schemas, all run histories, or full logs. Retrieve best-so-far through a small aggregate query if useful.

Use `scripts/snapshot.py --fast --service <service> --pid <training-launcher-if-separate> --log <log>`. This skips process-tree/PSS and GPU-process attribution, limits the default log tail to 8 lines/16 KiB, and uses shorter command timeouts. It still reports current service/root-PID status, host RAM, device-wide GPU memory/utilization, and log freshness. Do not describe device-wide GPU figures as target-job allocations. Job RAM is deliberately omitted, not zero.

Respond with current epoch/batch, latest primary metric and matched-epoch baseline delta, a brief interpretation, and a compact GPU/host-RAM summary. Include approximate next-result/end ETAs only if timing estimates already available remain applicable to this same run and configuration; identify reused timings and their age, and recalculate remaining work from current progress. Otherwise say timing is not refreshed in fast mode. Do not rescan historical logs to reconstruct phase timing or compute a queue ETA. Skip per-process RAM and detailed timing tables.

Aim for one batched evidence pass and a short response; do not promise a fixed latency. If logs or metrics are missing/stale, or a job may have failed, make only the targeted follow-up needed to establish status and state any remaining uncertainty. Do not automatically expand a fast request into a full audit or wait for the next validation.

## Identify the current work

Use conversation context and the project workspace to find the active run, launcher/service, queue, logs, and tracking store. Verify process/service state alongside the queue: a queue marked `training` can be stale. Distinguish active, queued, finished, failed, and superseded runs. A restarted run must not inherit the old run's elapsed time, metrics, or ETA.

Read the effective run configuration and recorded overrides. Establish:
- Primary validation metric, aggregation, and optimization direction; use the user's current selection.
- Epoch budget, current epoch/batch, batches per epoch, validation cadence, gradient accumulation, and any early-stopping rule.
- Meaning of an epoch, especially sampled-series or iteration-defined epochs. Distinguish dataloader iterations, global samples, and optimizer steps; DDP ranks execute in parallel.

For SKP runs, read [references/skp.md](references/skp.md). Other projects should use their native structured logs or experiment tracker.

## Gather fresh evidence

Prefer existing structured metrics and logs over importing training code or loading checkpoints. Retrieve the last 3–5 completed validation evaluations and the best primary metric so far, with epoch/step identifiers. Include relevant training/validation loss where available. Do not confuse sanity-validation output with a completed epoch result or a progress-bar batch loss with epoch-average loss. Clarify zero-based log epochs when reporting them.

Use the bundled Linux collector for resource snapshots and bounded, normalized log tails:

```bash
python <skill-dir>/scripts/snapshot.py --service <user-service> --log <training-log>
# Or identify a launcher process explicitly; descendants include workers/DDP ranks:
python <skill-dir>/scripts/snapshot.py --pid <launcher-pid> --log <training-log>
```

It requires only Python's standard library, with optional `nvidia-smi` and `systemctl`. It outputs JSON; it does not infer metrics or ETAs. Missing tools, permissions, PIDs, or logs are reported as unavailable, not as zero usage. On other platforms, use equivalent read-only tools.

Capture current per-GPU allocated memory/total capacity and utilization. Distinguish target-job GPU allocation from other processes when GPUs are shared. Report host used/total/available RAM and job RAM when available. Prefer summed process-tree PSS for job RAM; summed RSS double-counts shared pages. Include workers. A single snapshot cannot establish peak usage, a leak, or sustained utilization. Do not confuse `nvidia-smi` allocation with framework tensor-only allocation.

Record snapshot time and log/metric freshness. If evidence suggests a stall, take a second short observation to check progress and process state before calling it stalled. Do not wait for the next epoch just to answer an update request.

## Completed runs: final and best checkpoints

Whenever a completed run is displayed, in full or fast mode, report **both final and best checkpoint metrics**, with their epoch/step identifiers. Lead with final results and use final checkpoints as the default basis for completed-run rankings and conclusions: the user prefers final checkpoints to avoid additional validation-driven checkpoint-selection bias. Best results are secondary context, not a replacement for final results. If both coincide, explicitly label the value as final and best.

Use a compact table such as `Run | Final metric (epoch) | Best metric (epoch)`. Determine best using the configured primary metric and optimization direction. Report relevant secondary metrics from those same two evaluations, rather than independently cherry-picking each metric's best epoch.

Final means the checkpoint at the actual training endpoint. Verify the planned budget and termination status. If the endpoint was not evaluated, report final metrics as unavailable and identify the latest evaluated checkpoint separately; do not relabel it final. Label early-stopped, interrupted, or failed runs accordingly instead of implying planned-budget completion. Distinguish logged final-epoch metrics from verified saved-checkpoint availability, and identify EMA or other weight variants where relevant. Predictions exported from the best checkpoint do not establish final-checkpoint performance.

Retrieve final and best records with small read-only queries, including in fast mode. Missing metrics remain unavailable; monitoring does not authorize new inference, checkpoint changes, or alterations to training/checkpoint selection. This preference changes reporting, not the active experiment protocol.

## Compare ablations fairly

When monitoring an ablation, report matched-budget comparisons against its designated baseline and relevant completed or active arms, alongside each run's latest result and best-so-far. Retrieve historical evaluations from the baseline; do not compare a new arm's early epoch only against another arm's final or best checkpoint.

- For non-schedule ablations with the same epoch definition and training budget, compare primary metrics at the latest completed validation epoch shared by the runs. Show the epoch, each metric, and the difference from baseline; include earlier shared epochs when they clarify the trajectory. If the active run is ahead of a peer, identify the older shared comparison separately from its latest result.
- Check that the evaluation cohort, metric/aggregation, epoch definition, effective batch size, optimizer-step counts, and LR schedule are comparable. Identify intended ablation differences and material confounders such as pretraining data. Matching epoch numbers alone does not establish an otherwise controlled comparison.
- For training-schedule ablations, choose comparison axes that answer the experiment's question: cumulative optimizer steps, examples/series exposures, elapsed training time, or fraction of the planned schedule. Report final-budget results separately. Use matched epochs when informative, but explain differing LR/warmup phases: epoch 25 of a 50-epoch cosine schedule is not the same schedule phase as epoch 25 of a 100-epoch schedule. Equal schedule fractions also do not imply equal compute or data exposure. If epoch definitions differ, do not treat equal epoch numbers as equal training budgets.
- Make the best supported comparison with available records. When exact shared evaluations are absent, use clearly labeled nearest observed points with both epochs/steps/budgets, or state that no comparison is available yet. Do not silently interpolate metrics, launch evaluations, or wait for a shared point just to answer an update.
- Interpret whether an arm is ahead, behind, catching up, or inconclusive at the matched budget. Distinguish learning speed from final attainable performance. Treat small single-seed differences as provisional; best-so-far comparisons have different selection opportunities when validation counts differ.

## Timing and ETA

Use measured steady-state durations, preferably the median of the most recent comparable complete phases. Exclude initialization, sanity checks, compilation, paused periods, and interrupted epochs. With only startup timing, label the estimate provisional.

- **Iteration:** seconds per training dataloader batch, from increasing batch counters and elapsed-time deltas within the same epoch. De-duplicate progress-bar redraws. If accumulation is used, separately explain optimizer-step timing when useful.
- **Epoch:** training-only duration and, when measurable, total duration including validation/checkpoint overhead. Infrequent validation means ordinary and validation epochs take different times.
- **Validation:** duration of a full validation pass; never extrapolate two sanity batches as a measured full validation time. TQDM's final epoch elapsed time can include validation, so do not count that time twice.
- **Next result:** remaining training until the next scheduled validation **plus that validation's remaining duration**. If already validating, use remaining validation batches and observed validation throughput. A loss-log update is not a new validation result.
- **Run end:** remaining training batches across all remaining epochs + remaining scheduled validation passes + observed checkpoint overhead. Include partially completed phases only once. Count validation boundaries using the actual cadence rather than assuming every epoch validates.
- **Queue end:** if requested or relevant, give a separate estimate for later queued runs, with explicit assumptions about matching model/batch/hardware throughput. Identify post-training export/evaluation time separately from training end.

If full validation timing is unavailable, give a training-only lower bound or an explicitly qualified range; state what is missing. If the current stage/total work is unknown, say ETA is unavailable. Report durations rounded to the precision supported by the data; use the user's timezone for optional clock-time estimates. Scheduled completion estimates do not predict early stopping or convergence.

## Present the update

Lead with the run's state and position. A useful compact response consists of:
- A small table of the latest completed validation epochs: primary metric, change, and best-so-far where useful.
- For ablations, a compact matched-epoch or justified matched-budget comparison with baseline deltas; keep latest and best-so-far results separately labeled.
- One or two sentences interpreting the trend: improving, roughly flat, noisy, or declining. Base overfitting concerns on validation versus training evidence; falling training loss alone is not evidence of better generalization. Do not overinterpret a single evaluation or compare metrics with different definitions/cohorts.
- A compact timing/resource summary covering iteration time, epoch time, validation time, GPU memory/utilization, host/job RAM, ETA to next validation result, and ETA to run end. Mention queue end separately when useful.
- A link to the active log or tracker and any material missing/stale information.

Before the first full validation, state that no completed epoch validation result exists yet. Still report progress, measured training speed/resources, and qualified timing estimates. If a job failed or finished, report that directly rather than giving a running-job ETA. Keep recommendations brief and evidence-based; do not implement them as part of monitoring.
