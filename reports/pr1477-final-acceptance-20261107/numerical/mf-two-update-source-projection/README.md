# MF code-revision bounded comparison

This projection contains only three pairs of native raw LOSS/raw MTP/raw AUX,237-item difference counts,changed FQN sets and code revision/hash identities. It omits private YAML,RNG,token data/input hashes,native tensor hashes/payloads and machine identifiers. It has not been published by this exporter.

The cases are old52f/a1,old52f/a2 (a same-source,same-device repeat),and new62ab/a1. Each case completed two native updates1501–1502. Initial all237 items matched exactly in all three cases. The same-source repeat was itself nonexact after updates:4 model+29 optimizer-only item hashes at1501,and56+153 at1502. Across-revision pairs are also nonexact. Full changed names and raw signed values are preserved without smoothing or a numerical threshold; AUX remains diagnostic.

The active source difference under the audited non-reentrant,MC2-disabled path is persistent GroupedMatmul instances plus list/Tensor return normalization. Constructor attrs and eight supplied operands are unchanged; the list-return path still selects output[0]. New reentrant/MC2 bridge paths were not exercised. Source structural reasoning and the observed native repeat variability are separate evidence; close loss is not a semantic proof.

This is a two-update compatibility/control result,not a new1000-step result for62ab. A single repeat does not define a variability envelope,identify a kernel,or prove every across-revision difference has the same cause. The original long baseline retains its locked52f identity.

- three-pair-raw-losses.csv:all raw values and exact actual-minus-reference signs.
- all237-difference-counts.csv:aggregate counts only,no tensor hashes.
- changed-fqn-sets.json:complete changed-name sets for each pair/step.
- source-revisions.json:fixed source revisions,six-file delta and code hashes.
