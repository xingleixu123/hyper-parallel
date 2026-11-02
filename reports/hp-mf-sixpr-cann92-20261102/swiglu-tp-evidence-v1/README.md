# Packed SwiGLU: TP2 shared-expert micro reproduction

This archive preserves an isolated, synthetic four-rank BF16 test of a `SwiGLUMLP` under a DP2×TP2 mesh. It is not a model-training run or performance benchmark. No production YAML or full model configuration is included.

The measured run used HyperParallel integration source HEAD `2dd33e02f46fc5d6810e806b9d7d4717cf10a3e4` (tree `3d4a21c306e69bfb2c0d17ddc3d8519af0c6ef76`), Python 3.12.14, PyTorch 2.12.0+cpu, torch_npu 2.12.0.post3, and four Ascend 910B3 NPUs with CANN 9.2.0 B237. Use compatible hardware and software and the indicated HyperParallel source to reproduce; the version list describes the measured environment, not an official compatibility certification.

On the measured `default` plan, all four ranks recorded an exception during FC2 forward: the fused SwiGLU result and FC2 weight had incompatible K dimensions. The diagnostic processes exited normally after recording the exception; their exit status is not a correctness pass. On the measured `replicate` override, four ranks completed forward/backward and matched the reference output, input gradient and weight gradients at rtol 0.008, atol 0, with observed max absolute difference 0. Both cases used identical seeded and broadcast synthetic weights and input. The override changes weight placements and input/output contracts together, so it does not isolate the individual cause. Full rank JSON, per-rank completion records and complete stdout/stderr are under `reports/swiglu-tp/{default,replicate}`. Both measured stdout files are empty.

The exact measured Python entry and task bootstrap are preserved as `repro/executed-entry.py` (SHA256 `b1adcc3e4bfdcebb372bf4a43a70a39dc7418192e53140a3a72572a0060af03f`) and `repro/executed-bootstrap.py` (SHA256 `295713efeb75cb490cbd48454c320b4a3539b7c143be5c77fec091eaafb46dbf`). They retain their original task-specific handshake and should be read as evidence, not used as the standalone command.

`repro/repro.py` is a standalone candidate with standard torchrun rank/device setup and `--case` in place of the task configuration file and release handshake. The calculation, model construction, broadcast, reference, planner, apply, observations and result body beginning with the HyperParallel imports are text-identical to the measured entry (`cpu-check.json` verifies this). The simplified bootstrap was **not rerun on NPU** and supplies no new accuracy or performance evidence. Run each case into a fresh output directory, with four visible devices and the desired Ascend environment already configured:

```bash
torchrun --standalone --nproc-per-node=4 repro/repro.py --case default --output ./result-default
torchrun --standalone --nproc-per-node=4 repro/repro.py --case replicate --output ./result-replicate
```

Each process creates a new `<output>/<case>/rank-N/` directory and writes `micro-result.json` and `complete.json`. The test records exceptions inside the JSON; inspect classification on all ranks rather than treating process exit alone as success. Select an isolated set of four available devices before execution.

`github-refs.json` records a read-only, proxy-routed check of [PR #1413](https://github.com/mindspore-ai/hyper-parallel/pull/1413): its head is `hf1s/hyper-parallel:feat/xing40-standard-muon` at `4463d5a30440aedc1e9d455332a328b816c9bde9`; upstream master was `4144bb242f1a696308e75bd609cb61c9143b3d8f`. The two currently modified core files were byte-identical between those two refs. A dependent stacked fix can target the PR head; a genuinely independent core fix may target upstream master after validation there. Neither requires including the unrelated integrated commits. The reference facts alone do not choose the base or prove the fix.

The mechanical scanner checks the selected artifacts for credential patterns, private host paths and configuration-file references; it found none, but this is not a substitute for human publication review. `SHA256SUMS` lists package files. The archive contains evidence and a standalone reproduction candidate; it does not assert a repaired implementation.

The measured source can be rebuilt from the eight original public PR commits plus the code-only CE patch. See [source reconstruction](repro/README-source.md) and [exact source identifiers](repro/source-versions.json). A recreated commit ID may differ; the required tracked tree is `3d4a21c306e69bfb2c0d17ddc3d8519af0c6ef76`.
