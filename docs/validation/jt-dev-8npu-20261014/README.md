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

Remaining TP/CP/EP drift is being diagnosed. The six-fix follow-up is documented below; EP/CP same-topology memory regressions have passed and are documented below. Module cost ablations and the MLP+norm 200-step check are documented below; additional hybrids remain pending.

Figures are available as PNG and editable SVG; JSON contains all-rank step errors and provenance names. CSV contains scalar measurements and input hashes only, with no training text, weights, checkpoints, or credentials.

## Six-fix FSDP and resumed-counter regression

The follow-up source is `ec2e16bce26e497519c573f0a8d87a538a329197`: the four fixes above plus cumulative metric restore and initialization-parameter lifetime cleanup. The baseline runs steps 1-200, saves a complete checkpoint at 200, and the next process resumes steps 201-400. The reference is an independent uninterrupted 400-step run with #180 included.

Both baseline and resume match every LM/MTP/aux/total loss and gradient norm exactly on all eight ranks. Baseline/resume median step times are 0.937752/0.937632 seconds; maximum allocated memory is 22.633759 GiB for both. Inputs, initialization, learning rate, and coverage are checked separately.

The old resume reset cumulative counters: at step 201 it logged 8 samples and 6304 supervised tokens, compared with the continuous reference's 1608 samples and 3720095 tokens. The fix has zero mismatches in cumulative and per-step sample/token counters across all 200 resumed steps and eight ranks. This reporting defect did not cause a loss mismatch. Old checkpoints without meter state remain loadable and cannot reconstruct unavailable historical counts; weights-only restore does not apply meter state.

![Resume counter regression](figures/resume-counters.png)

![Resumed loss alignment](figures/reviewed-fsdp8-resume.png)

`scalars/resume-counter-evidence.csv` contains the exact all-rank source measurements for these comparisons. The large checkpoint was removed only after loss and counter comparisons passed; compact metadata and the file-size inventory are retained locally.

## Initialization parameter lifetime: EP2 and CP2

The scheduler and HSDP state retained full original Parameter objects through source-metadata dictionary keys and explicit replicate-parameter sets after shard construction replaced those parameters. CPU weak-reference tests cover the real shard cloning path, caller-container preservation, and tied aliases. All six old-code lifetime cases fail; the revised five-file CPU suite passes 112 tests and 12 subtests. The new test has no pylint findings; 15 production-file findings and two complexity warnings are unchanged from the base.

| Configuration | Allocated before / after GiB | Saved GiB | Reserved before / after GiB | Median seconds before / after | Same-topology loss and gradient errors |
|---|---:|---:|---:|---:|---:|
| EP2 | 30.605194 / 21.183023 | 9.422171 | 36.138672 / 26.724609 | 1.019160 / 1.020075 | exactly 0 |
| CP2 | 31.183506 / 21.058210 | 10.125296 | 36.394531 / 26.021484 | 1.283979 / 1.285255 | exactly 0 |

Each comparison covers 200 steps and all eight ranks. Time medians exclude 20 warmup steps. The actual saved memory agrees with the independently estimated retained parameter storage. Timing differences below 0.1% in these single runs are not evidence of a throughput change. FSDP8 allocated memory remains 22.633759 GiB.

The EP2 pre-fix source is the #180 integration `81f141d`; the CP2 pre-fix source is the four-fix integration `6e762be`. Both follow-ups use `ec2e16b`. Other integrated fixes are disclosed above; exact same-topology loss equality does not mean EP2 or CP2 now meets the cross-topology FSDP tolerance. Those outstanding errors remain listed in the earlier table.

![Measured memory and step time](figures/source-parameter-lifetime.png)

![EP2 same-topology loss regression](figures/reviewed-ep2-memory.png)

![CP2 same-topology loss regression](figures/reviewed-cp2-memory.png)

## CP chunked-loss distributed regression

The four-process NPU launcher `tests/torch/context_parallel/test_jt_mla_cp.py` passed in 47.19 seconds on `ec2e16b` (four inherited unknown-pytest-marker warnings). Worker output records 16 steps: two steps for each of six existing CP strategy/TP/SP combinations, plus two steps each for chunk size 48 under TP1+CP4 and TP2+SP+CP2. The latter also uses EP4 for fused experts. Existing ST relative tolerances are distinct from the 200-step absolute loss gate; no stricter claim is inferred from this test.

The ordinary CP2 training demo with chunk size 4096 now completes 200 steps without the previous configuration rejection. Its remaining cross-topology loss error is still a failure and is under numerical diagnosis.

## Ranked findings, recompute costs and precision diagnostics

[中文问题排序与完整曲线](REPORT_ZH.md) distinguishes completed fixes from the outstanding cross-topology gate. The six draft PRs each have one commit; [current-head receipts](reports/pr-verification.json) do not establish CI success.

Seven recompute policies were compared over 40 steps, with events collected in separate 12-step runs. Every observed loss and gradient matches the off reference exactly on all eight ranks. MLP+norm additionally completed 200 exact steps, saving 1.1654 GiB for about 2% additional step time. Full/selective cost about 9.13%/8.50% in the same-source 40-step study. Module paths are experimental wrappers, not an existing JT YAML feature. [Measurement definitions](reports/recompute-cost-analysis.md) and [pinned current MF/Megatron source comparison](reports/recompute-reference-comparison.md) document the boundaries.

![Recompute cost](figures/recompute-module-cost.png)

FP32 expert weight gradients plus the complete FP32 NS pipeline do **not** resolve the gate. Completed EP2 and CP2 diagnostics use a matching 200-step FSDP reference; all input, initialization, LR and coverage checks pass, but LM maxima are 0.0194788 and 0.0152731. FSDP median time rises to6.39175s, roughly6.8 times the original recipe, so this is not a recommended production recipe. TP2+EP2 and actual production-option integration tests were still pending at this snapshot.

The additional raw-component audit divides logged MTP/aux by their verified weights0.3/0.0001 and applies0.005 separately. EP2 raw maxima are0.0301822/0.128801; CP2 raw maxima are0.0355840/0.170822. The original weighted reports remain intact. [Detailed diagnostic evidence](reports/parallel-numerical-diagnosis.md) states which findings are sampled and which cover every step/rank.

![EP2 diagnostic raw losses](figures/fpgmm-nsall32-ep2-200-raw.png)

![CP2 diagnostic raw losses](figures/fpgmm-nsall32-cp2-200-raw.png)

The [precision CSV](scalars/precision-diagnostics.csv) contains9600 scalar rows from six completed runs, with a distinct diagnostic-hook flag. No training text or weight tensors are published.
