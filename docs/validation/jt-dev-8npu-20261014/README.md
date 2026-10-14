# JT DeepSeek V3: eight-910B validation evidence

This immutable snapshot documents completed runs and known failures. It does not claim the complete parallel matrix is accepted.

- Base: hf1s/hyper-parallel JT_dev at ad1c55379767d63ece7e084b0623340b37b83d93.
- Included PR #180 head: 5de3cceab094bbed18915d1a9c9231ca220629ca (integration 81f141d).
- Four-fix integration: 6e762bec039ced4e2fc5cdcfbf6326a179810e03 (whole-step token normalization, CP chunked projection loss, complete recompute coverage, replay-safe routing counts). Each fix is tracked separately; these multi-fix integration measurements are not isolated per-PR tests.
- CANN 9.2.0-beta.2 V100R001C12B086, PyTorch 2.12.0, torch_npu 2.12.0, Python 3.12.14, transformers 5.13.1; eight Ascend910B3 64GiB.
- Sequence length4096, global batch8, micro batch1, seed42, constant LR0.0001, unchanged Muon/AdamW recipe and initialization. Every compared step verifies canonical input hashes; complete native initial-state hashes match before partitioning.

## Evidence semantics

Loss errors are absolute maxima over every measured step and all eight ranks. The threshold is0.005 for each model-logged component and total. MTP is weighted by0.3 and aux by0.0001; reports also expose unweighted error diagnostics. The curves above show rank0; lower panels show the maximum error across ranks at every step. A completed process is not a passing alignment result.

Step time is the median after excluding the first20 steps of that measured run. Allocated and reserved memory are the maximum reported per-device peaks. Timings include the same lightweight scalar/input observations; runtime parameter-probe jobs are excluded. These are single-run comparisons, not confidence intervals.

## Completed checks

| Run | LM max error | MTP max error | Aux max error | Median seconds | Peak allocated GiB | Alignment |
|---|---:|---:|---:|---:|---:|---|
| FSDP8 off0-200 |0|0|0|0.939278|22.633759|PASS|
| FSDP8 resume200-400 versus continuous400 |0|0|0|0.938972|22.633759|PASS|
| FSDP8 full recompute |0|0|0|1.025728|19.538601|PASS|
| FSDP8 selective recompute |0|0|0|1.019489|20.227203|PASS|
| TP2+EP2 before token fix |0.922379971|0.290920973|0.000037092|1.407143|21.334600|FAIL|
| TP2+EP2 after token fix |0.012481689|0.004909277|0.000013115|1.404477|21.334347|FAIL|
| EP2 original |0.014165878|0.003723621|0.000012476|1.019160|30.605194|FAIL|
| CP2 after chunk support |0.022237778|0.005699873|0.000021477|1.283979|31.183506|FAIL|

Remaining TP/CP/EP drift is being diagnosed. The six-fix follow-up is documented below; EP/CP memory regressions are still running. Full/selective module ablations, additional hybrids, and their acceptance remain pending.

Figures are available as PNG and editable SVG; JSON contains all-rank step errors and provenance names. CSV contains scalar measurements and input hashes only, with no training text, weights, checkpoints, or credentials.

## Six-fix FSDP and resumed-counter regression

The follow-up source is `ec2e16bce26e497519c573f0a8d87a538a329197`: the four fixes above plus cumulative metric restore and initialization-parameter lifetime cleanup. The baseline runs steps 1-200, saves a complete checkpoint at 200, and the next process resumes steps 201-400. The reference is an independent uninterrupted 400-step run with #180 included.

Both baseline and resume match every LM/MTP/aux/total loss and gradient norm exactly on all eight ranks. Baseline/resume median step times are 0.937752/0.937632 seconds; maximum allocated memory is 22.633759 GiB for both. Inputs, initialization, learning rate, and coverage are checked separately.

The old resume reset cumulative counters: at step 201 it logged 8 samples and 6304 supervised tokens, compared with the continuous reference's 1608 samples and 3720095 tokens. The fix has zero mismatches in cumulative and per-step sample/token counters across all 200 resumed steps and eight ranks. This reporting defect did not cause a loss mismatch. Old checkpoints without meter state remain loadable and cannot reconstruct unavailable historical counts; weights-only restore does not apply meter state.

![Resume counter regression](figures/resume-counters.png)

![Resumed loss alignment](figures/reviewed-fsdp8-resume.png)

`scalars/resume-counter-evidence.csv` contains the exact all-rank source measurements for these comparisons. The large checkpoint was removed only after loss and counter comparisons passed; compact metadata and the file-size inventory are retained locally.
