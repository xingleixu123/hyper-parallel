# Parallel numerical drift: staged evidence

All observations below use the same native initial tensors, data and original BF16-forward Muon/AdamW recipe. Source: `ec2e16b`, including PR #180. None establishes the outstanding 200-step cross-topology acceptance.

## Step 1 expert gradients and actual parameter updates

The probe reconstructs global tensors before taking identical strided coordinates. These are **sampled relative L2 errors**, not full-tensor error bounds. Update error compares `(parameter_after - parameter_before)`, not the much larger absolute parameter tensor.

| Layout vs FSDP8 | Expert pre-clip gradient error range | Expert update error range |
|---|---:|---:|
| ep2 | 0.1334%–0.1461% | 5.090%–7.102% |
| cp2 | 0.1163%–0.1223% | 4.525%–6.622% |
| tp2ep2 | 0.9143%–1.6783% | 13.570%–17.166% |

EP2 initial expert samples, first-step gate-gradient/update samples, and first-step routing counts/bias are exactly equal. The observed dense/attention full small tensors also have equal pre-clip gradients. Expert updates differ before later routing counts drift. This localizes a precision-sensitive path; it does not prove that optimizer precision alone resolves it.

## Grouped-GEMM partition experiment

On the installed 910B/CANN stack, splitting two token batches and summing their BF16 grouped weight gradients produces relative L2 error 0.00233560 versus computing their concatenation. Accumulating directly through `npu_grouped_matmul_add_` into FP32 reduces the partition difference to 6.94e-8. Empty-expert output remains zero. This is an operator experiment, not a training acceptance result.

The ordinary JT grouped SwiGLU path currently returns BF16 weight gradients, then FSDP converts/reduces them. Returning FP32 from an autograd function whose weight input is BF16 does not by itself preserve FP32 through autograd. A production fused-gradient design must account for that dtype boundary, FSDP reduce/scatter ownership, empty experts, replay, and accumulated micro-batches.

## FP32 Newton–Schulz ablation: completed

Four three-step diagnostic runs used FP32 Newton–Schulz normalization/polynomial computation while preserving the existing BF16 optimizer inputs and returned updates. Installed NPU `matmul.allow_hf32` was verified false. Each topology is compared with the matching FP32-NS FSDP reference. These are instrumented diagnostics, not performance measurements or 200-step acceptance.

| Layout vs matching FP32-NS FSDP8 | Step-1 expert update sampled relative L2 error |
|---|---:|
| EP2 | 4.055%–5.627% |
| CP2 | 3.606%–5.175% |
| TP2+EP2 | 12.038%–14.542% |

The FP32-NS FSDP run itself has exactly the same first-step sampled pre-clip gradients as the original recipe, while its sampled expert updates differ by 4.834%–7.164%. The hook therefore changes the intended computation, but increasing only NS internal precision does not eliminate the cross-topology sensitivity. It is not a proven fix.

## FP32 expert weight-gradient delivery: completed short diagnostic

This ablation preserves the BF16 forward/input-gradient kernels and original Muon recipe, but accumulates grouped weight gradients directly in FP32. An explicit diagnostic callback delivers that buffer to the existing FSDP unsharded accumulation buffer, avoiding autograd's cast back to the BF16 weight dtype. A zero weight gradient keeps the normal FSDP backward scheduling. This narrow experimental callback is not a proposed production API.

Four three-step NPU runs completed. Each topology is compared with its matching FP32-weight-gradient FSDP reference. The following numbers are relative L2 errors of the four expert parameter samples at step 1:

| Layout | Pre-clip gradient error | Actual update error |
|---|---:|---:|
| EP2 | 1.738e-7–2.168e-7 | 0.005302–0.006933 |
| CP2 | 1.036e-5–2.823e-4 | 0.013294–0.029719 |
| TP2+EP2 | 0.009061–0.016721 | 0.133370–0.169300 |

FP32 delivery removes almost all sampled EP gradient discrepancy but not the optimizer update sensitivity. TP still has larger differences upstream of expert weight-gradient rounding. All eight ranks recorded FP32 delivery. These measurements do not establish 200-step acceptance.

## Full FP32 Newton–Schulz pipeline with FP32 expert gradients

A second four-run, three-step diagnostic keeps the optimizer's momentum-to-NS input, NS arithmetic, output scaling and relay buffers in FP32. It preserves the NS coefficients, iteration count, momentum, LR and all data. `matmul.allow_hf32=False` is recorded. Each layout uses the matching FSDP reference with these same numerical settings.

| Layout | Step-1 expert update sampled relative L2 error |
|---|---:|
| EP2 | 1.555e-5–1.849e-5 |
| CP2 | 0.0002174–0.0027904 |
| TP2+EP2 | 0.111432–0.131487 |

EP update error is much smaller; TP remains unresolved. The completed 200-step FSDP reference has median 6.391754 seconds after excluding 20 steps, allocated 22.282196 GiB and reserved 27.855469 GiB, about 6.8 times the original BF16-NS recipe's step time. The completed EP2 run fails over all eight ranks: LM max error 0.019478798 (step188), weighted MTP 0.009054661 (step97), weighted aux 0.000012880133 (step200), and total 0.022395134 (step188). Input identities, initial tensors, LR and complete step coverage match. EP median is 6.599639 seconds, allocated 21.183060 GiB and reserved 26.685547 GiB.

The model-logged MTP and aux values include weights 0.3 and 0.0001. A separate unweighted audit reports maximum errors 0.030182203 and 0.128801330 respectively, both above 0.005. Weighted and raw values are shown separately; scaling cannot hide a discrepancy. CP2 also completed 200 steps and failed: LM 0.015273094 (step148), weighted MTP 0.010675192 (step97), weighted aux 0.000017082202 (step198), and total 0.015007019 (step146). Unweighted MTP/aux errors are 0.035583973/0.170822022. Its median is 7.063104 seconds, allocated 20.706646 GiB, reserved 26.255859 GiB. All input/initial/LR/coverage checks pass. TP2+EP2 is still running. These remain diagnostic hooks, not completed production validation. See `fpgmm-nsall32-ep2-200-comparison.json`, `fpgmm-nsall32-ep2-200-raw-component-gate.json`, and both weighted/unweighted figures.

The separate NS batch-partition operator probe found exact BF16 equality for batching eight matrices versus two groups of four for both tested shapes. Eight-matrix versus individual-matrix BF16 execution differed by approximately 0.0091 relative L2. Thus matrix batching can matter, but that eight-versus-four probe does not explain the measured EP2 difference.

The production FP32 expert-gradient patch has passed both independent review axes and the targeted CPU regression (87 tests, 25 subtests); its real two-rank NPU ST is queued. CPU regression now covers mixed ordinary/fused gradients, shared uses, non-reentrant checkpoint replay, actual world-size-one no-sync→sync scheduling, unused units, frozen weights and rejection of distributed tensors at the local-kernel boundary. Actual multi-rank NPU integration remains required. Compact evidence and SHA256 inventory: [diagnostic inventory](fpgmm-nsall32-evidence-inventory.json).

The separate production `ns_dtype` option has passed CPU checkpoint and update regressions (44 tests, 52 subtests), including the real trainer/DCP loader, both FQN formats, fp32 main weights, old/malformed checkpoints, empty chains and callback-output alias isolation. It has not passed the required NPU training matrix and is not a validated solution to this drift. The next isolated diagnostics compare full-width module activations at step 1 and preserve raw FP32 outputs when measuring NS batch size and HF32 effects; the earlier NS operator probe rounded its outputs to BF16 and could hide smaller differences.

Megatron exposes Newton–Schulz matmul precision and documents a separate matrix-batching rounding tradeoff. This supports testing the two effects independently; it is not proof that this NPU implementation shares the same kernel behavior.

- [Megatron pinned Muon configuration](https://github.com/NVIDIA/Megatron-LM/blob/07046cc6b8b7479ba05e6b47ed25f3f8e6634573/megatron/core/optimizer/optimizer_config.py#L273)
- [Ascend grouped-matmul-add API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_add.md)
- [Ascend grouped-matmul API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)

## Next-forward weight samples and remaining NS batching question

The 97,280 observed expert/router main-weight coordinates were rounded to BF16, matching the next-forward parameter dtype. At step1, EP2 has zero differing observed BF16 coordinates, CP2 has83 and TP2+EP2 has3392; at step2 those counts become4919,6779 and7715. Zero sampled differences do not imply equality of every parameter. All eight ranks of all four layouts have identical router bias at every observed stage of steps1–3; early divergence therefore cannot be attributed to a different stored bias at these observed points. See `nsall32-next-forward-weight-samples.json`.

Muon merges same-shape matrices across the parameters assigned to each compute owner and squeezes a singleton matrix batch to2D. Changing topology can change both owner assignment and the merged matrix count. Synthetic8-versus4 expert probes alone do not cover all actual dense/MoE batch shapes. The queued short runs now record actual NS batch shapes and parameter groups without copying tensors or changing arithmetic. This extends the evidence before attributing the drift to a particular rounding boundary.
