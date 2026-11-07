# Evidence for HyperParallel PRs #1480 and #1481

This bundle records targeted tests for [swap #1480](https://github.com/mindspore-ai/hyper-parallel/pull/1480) at `bd037f88994d41aa3bde45823266dcf32ae3ba22` and [context parallel #1481](https://github.com/mindspore-ai/hyper-parallel/pull/1481) at `af64e123e509085e69dc31578eee712155813538`. Each is a single commit based on `c086152596226c99f5c06597491ee94b4c26d26c`. This is diagnostic evidence, not a full gate result. Snapshot assembled 2026-11-07T05:04:52+00:00. Terminal CI writebacks remain separately verifiable on the linked PRs.

## Swap: native four-device short comparison

On 129, physical Ascend 910B3 devices 4–7 were visible as 0–3. The unmodified `test_swap_muon_1` launches four independent single-rank workers; this is not the complete eight-card CI workload. The same image, isolated editable source, CANN 9.2.0 (V100R001C12SPC001B237), Python 3.12.14, Torch 2.12.0+cpu, torch_npu 2.12.0.post3, OMP/MKL threads 2, eight CPU quota and 32 GiB container memory were used for sequential ABBA. Every controller returned zero with 10 monitor samples, zero foreign and zero unknown samples; all 16 workers passed.

| Order | Source | Container command wall (s) | pytest call (s) |
|---:|---|---:|---:|
| 1 | base | 32.591 | 29.41 |
| 2 | head | 32.902 | 29.95 |
| 3 | head | 32.589 | 29.22 |
| 4 | base | 32.622 | 29.76 |

The wall field covers container command execution, rather than the entire controller interval that also verifies source and device admission. These are launcher and worker timings, not steady-state training-step measurements.

The wall means are 32.606 and 32.745 seconds, respectively; call means are both 29.585 seconds based on two-decimal logs. This short comparison did not reproduce a sustained difference. It does not prove absence of a regression or measure steady-state throughput. CI used Python 3.10, Torch 2.9.0+cpu and PTA 2.9.0.post5. CI 1501 recorded 47 NPU passes, while the four swap entries exceeded the device-time quota by approximately 2.031 seconds of wall time (0.51%). A single retest of unchanged `bd037` was requested at 04:55:41 UTC, comment 6702508146; no final outcome is claimed here. The cause of the quota overrun is not established.

## CP: native four-device test

The current `af64` source ran the native four-card callback test with the original tolerances: TP2/CP2, two checkpointed microbatches, and three representative expanded-Ulysses/latent-KV paths. The controller returned zero, 16 monitor samples had no foreign or unknown status, and all four ranks passed. Rank 0 reported maximum gradient relative L2 0.00379561724947224. This is a targeted test, not a new long training run. The prior `b1f` monitor failure is a separate historical result and is not replaced by this receipt. Earlier `4a4`/`57a7` reports apply only to their own revisions. CI 1505 Lizard and Pylint passed; its full pipeline was not complete when this bundle was assembled, and an empty NPU stage log is not evidence of a native run.

## CPU checks and method

The CP CPU case checks callback installation and callable boundaries; numerical runtime coverage uses actual NPU kernels. The swap test checks rejection of registered hooks before runtime/adapter side effects and successful construction after hook removal.

Swap CPU tests: 130 passed plus 26 subtests passed. CP CPU tests: 9 passed plus 18 subtests passed; the recorded Pylint check exited zero. `cp/complexity.json` records CCN 20 to 14 for `__call__` and 7 for the extracted helper; the QKV preparation block was moved verbatim.

`cpu/bench.py` compares the actual constructor methods extracted by AST from the two Git source files, checking that the sole method change is the guard. It uses empty optimizer state and a 2×2 CPU parameter, 12 batches of 200 constructions per version. Medians in nanoseconds: AdamW base 2991.8475, head 3097.2525; Muon base 3537.095, head 3598.825. The first construction was measured base before head and has a cold-order bias. These CPU construction measurements do not establish NPU throughput or explain the CI overrun.

## Verify and reproduce

Run `python verify.py` offline. It checks the complete file manifest, raw embedded log hashes, 20 worker results, five controller receipts and monitor records, source identity cross-checks, and recomputes the comparison. `source/identities.json` records Git blob IDs and SHA-256 of `git show` bytes; the CP receipt independently records matching production, launcher and worker hashes. Source bytes are not bundled. To obtain the benchmark inputs from the repository, using separate checkouts and an editable environment:

```sh
git show c086152596226c99f5c06597491ee94b4c26d26c:hyper_parallel/core/optimizer/swap_optimizer_base.py > cpu/base.py.txt
git show bd037f88994d41aa3bde45823266dcf32ae3ba22:hyper_parallel/core/optimizer/swap_optimizer_base.py > cpu/head.py.txt
python cpu/bench.py
```

For the swap native launcher, check out the chosen commit separately, install that checkout editable in the prepared NPU environment, set the test package root explicitly, and use its required working directory:

```sh
cd /work/source/tests/torch/swap_optimizer
PYTHONPATH=/work/source:$PYTHONPATH python -m pytest -s --durations=0 test_swap_muon.py::test_swap_muon_1
```

The native CP launcher is `python -m pytest -s tests/torch/context_parallel/test_mla_cp.py::test_mla_local_callbacks_npu_4_cards` from its separate editable checkout root. Commands require appropriate CANN/PTA and device access. The exact container commands and source comparisons are in the original receipts embedded in `swap/runs.json` and `cp/run.json`; raw worker stdout and its original path and SHA-256 are in the two `workers.json` files. `prepare.py` documents assembly from the local review directory; `manifest.json` covers all files including nested files, except itself.

