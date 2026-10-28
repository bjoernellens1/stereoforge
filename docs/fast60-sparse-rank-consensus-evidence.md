# Sparse Census and vertical rank consensus: analytic development evidence

This exploratory CPU comparison uses one previously inspected 1280×720 analytic
scene, D224, a 16-pixel cell and a minimum margin of one. The `rank` profile
keeps a valid baseline Census match only when a separate vertical rank search
and reverse search agree. The cues use the same imagery and are correlated;
this is not independent statistical confirmation. Both profiles use exactly
the same image-only selected points per selector. Settings and decisions were
persisted before truth was loaded, but the scene was inspected previously, so
this is not blind preregistration or held-out evaluation.

**Rank consensus fails the near and thin gates for both selectors and both
eyes.** It rejects every thin and near match in this scene. Overall accuracy
above 95% on the survivors does not compensate for missing surfaces; neither
profile is promoted as a depth algorithm.

| Selector | Matcher | Eye | All correct / accepted | Near correct | Thin correct / accepted | Edge correct |
|---|---|---|---:|---:|---:|---:|
| combined | single | left | 3,589 / 3,666 | 23 | 11 / 13 | 20 |
| combined | single | right | 3,721 / 3,800 | 25 | 41 / 47 | 35 |
| combined | rank | left | 295 / 295 | 0 | 0 / 0 | 0 |
| combined | rank | right | 271 / 271 | 0 | 0 / 0 | 1 |
| combined_edge | single | left | 26,936 / 27,540 | 75 | 40 / 47 | 119 |
| combined_edge | single | right | 27,038 / 27,642 | 71 | 81 / 94 | 135 |
| combined_edge | rank | left | 1,844 / 1,845 | 0 | 0 / 0 | 2 |
| combined_edge | rank | right | 1,813 / 1,813 | 0 | 0 / 0 | 4 |

All rank profiles have 0 / 0 correct / accepted for each eligible thin surface
(IDs 6, 7 and 8), and an empty denominator fails. The exploratory gates require
at least 95% correct over all accepted, at least two correct and 90% correct
per eligible thin surface, at least one correct each in near and edge, and at
most 5% unconfirmed-or-false across all accepted thin points. The latter uses
`(accepted - confirmed correct) / accepted`, fails when empty, and treats
outside-eligibility accepts as unconfirmed, not confirmed false. Known false
eligible counts and outside-eligibility counts are separately recorded. Correct
means disparity error at most one pixel under the base scorer's shared analytic
support rules. These sparse counts are not dense completeness.

| Selector | Baseline forward / reverse candidate upper bounds | Rank forward / reverse actual Hamming comparisons | Rank attempted points |
|---|---:|---:|---:|
| combined | 2,874,097 / 1,921,064 | 1,483,526 / 286,872 | 7,466 |
| combined_edge | 23,962,648 / 14,440,906 | 10,835,879 / 1,774,821 | 55,182 |

Counts sum both eyes. The rank profile reuses the baseline pass: total candidate
work is the baseline upper bound plus exact rank comparisons, a mixed upper
bound, not two baseline passes.
The full-frame rank descriptor preprocessing is **separate**: 1,822,720 pixel
positions and 14,581,760 comparisons per selector. These comparisons are not
the complete preprocessing cost: baseline Census preprocessing, rank support
and contrast work, selection, sorting, and other operations are excluded. Baseline counts are upper bounds because searches with fewer than two
candidates skip comparison; rank counts represent executed supported Hamming
candidate comparisons. None of these counts is a GPU time.

Rank ambiguity is the dominant additional rejection reason. For `combined`,
forward rank ambiguity rejected 2,953 left and 3,068 right baseline-valid
anchors; reverse rank ambiguity rejected 393 and 437. For `combined_edge`, the
corresponding counts were 22,711 / 22,811 and 2,557 / 2,574. The report also
records low contrast, unsupported footprints, disagreement and reverse mismatch.
This is synthetic CPU research, with no physical accuracy, dense reconstruction,
GPU performance, sustained live throughput, or 60 FPS claim.

## Reproduction and identity

The fixture and recorded output are ignored local artifacts and may be absent
from fresh checkouts. Choose a new output directory:

```bash
PYTHONPATH=python:. .venv/bin/python benchmarks/evaluate_sparse_rank_consensus.py \
  --coarse artifacts/fast60-refinement-20261028/analytic-family-720p/half/manifest.json \
  --target artifacts/fast60-refinement-20261028/analytic-family-720p/full/manifest.json \
  --family artifacts/fast60-refinement-20261028/analytic-family-720p/family.json \
  --output artifacts/fast60-refinement-20261028/sparse-rank-consensus-rerun
```

Recorded output: `artifacts/fast60-refinement-20261028/sparse-rank-consensus-development-v1/`.
Verified SHA-256:

- Policy: `6b9f125ca7a5281aa82726e56357153071d9ccfd3d984e253e2cfa07746f453f`
- Decisions: `b2636373266451f14222fe3d88a9050405a0b6e4128337f12945e49b4a0f5db9`
- Evaluator: `447af6bcc70b66dfa5d9529d571c796f582a5dc634e98a8483a4dc5b5944ec2f`
- Rank matcher: `9d037f219f0dedeeb22bc55125fb0d47a28c836057dcdca5ac240729d1ea38c2`
- Baseline matcher: `762e696747176c84c4ab6072140376391633968bd59ad44491763b285e517559`
- Base scorer: `6d1b4e4f358546b4cdfda652e5881b0c275c728f7887ba6780c68166d953a2bf`
- Thin scorer: `78ec765113540e1f5a615054239074e9d8934445055089419f6c01118dc9a041`
- Coarse manifest: `98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad`
- Target manifest: `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`
- Family manifest: `db7f6265a1be538d6a01091cea54f699a3564a39d289ca2dc2fe3d54080c9b9b`

The report includes input luma and support hashes, the calibration/capture
association and source-identity hash, dependency hashes, and NumPy version.
A precomputed baseline is safe to reuse only within this evaluator's matched
run: the matcher checks source identity, but a source ID alone does not
authenticate images or matching settings. External baseline reuse requires
explicit verified image, support, calibration, and parameter identities.
An independent v2 repeat produced byte-identical decisions and report files;
this confirms determinism for this fixture and environment only.
