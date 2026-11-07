# Native MLA callback test review

This report covers the focused test revision in upstream PRs [#1480](https://github.com/mindspore-ai/hyper-parallel/pull/1480) and [#1481](https://github.com/mindspore-ai/hyper-parallel/pull/1481). Each PR has one commit, based on master c086152596226c99f5c06597491ee94b4c26d26c. Exact head, tree, blob and SHA256 identities are retained under source/.

## What the tests protect

| Boundary | Coverage |
| --- | --- |
| CPU callback configuration | One existing-style TestCase method checks callable identity through wrapper installation and rejects invalid callback values. It never executes an NPU runtime on fabricated device metadata. |
| Native TP2/CP2 callbacks | One lightweight four-card launcher reuses torchrun_case, the existing unsharded reference, TP projection/CP plans and the existing gradient-comparison helper. Three distinct paths cover expanded synchronous, expanded overlapping and latent overlapping preparation. |
| Observable callback effects | Distinct per-global-head output gains detect an incorrect or ignored TP-local slice; recorded Q/shared-K rotary calls prove both dispatches occur. |
| Swap construction | The focused test checks real AdamW and Muon, existing pre/post hooks, both entry points, rejection before side effects, and wrapping after hook removal. The parameter snapshot precedes both rejection checks. |

The fake NPU proxy and private-validator capture are absent from the revised test. Production files are byte-identical to the preceding verified PR versions. No production CPU path, extra validation layer, test skip or relaxed numerical assertion was introduced.

## Actual results and limits

- CPU: MLA CP 9 tests/18 subtests; swap 130 tests/26 subtests passed.
- The repository native launcher and all four rank pytest workers each reported one passed case, with no skips. All three internal cases completed two microbatches with non-reentrant checkpointing and unchanged output, logical-input-gradient and parameter-gradient checks.
- Rank 0 reported parameter-gradient relative L2 values 0.0037913018135992313, 0.0037913018135992313 and 0.00379561724947224. These are its three printed values, not an invented all-rank maximum; all ranks enforced the existing bound of 0.025.
- The task-owned host controller returned failure during process teardown. A previously owned PID remained visible with the same starttime but a root cgroup and D state, triggering its conservative foreign-holder check. The failed receipt remains failed. Worker test success is therefore reported separately from controller-level acceptance.
- No more identical host-monitor runs were performed. External CI must independently validate its current head; no pass label is manufactured from these local results.

Environment: Python3.12.14, Torch2.12.0+cpu, torch_npu2.12.0.post3, pytest9.1.1, CANN9.2.0 B237, Ascend910B3. Physical devices4-7 map to logical0-3. The tests use synthetic inputs only. Earlier native single-card and broader four-card production checks remain separate frozen evidence.

## Reproduce and verify

Install the selected PR as editable and run from its source root:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q tests/ut/distributed/context_parallel/test_mla_context_parallel.py
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q tests/ut/core/optimizer/test_swap_optimizer_base.py
    python -B -m pytest -s tests/torch/context_parallel/test_mla_cp.py::test_mla_local_callbacks_npu_4_cards

The last command requires four actual NPUs and a compatible runtime. Its launcher is framework-free at collection and starts the existing worker through torchrun_case.

Run python verify.py in this report directory to verify all exported SHA256 values and independently check the launcher/rank test outcomes while retaining the failed controller result. This offline verification does not re-execute the device tests. Full source archives, command lines, monitoring records and failed predecessor attempts remain in the task's local evidence archive.
