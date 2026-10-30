# Single VIBE method: HingeZero learned-support 32,768

This branch supplies **only** the previously measured HingeZero learned-support VIBE adapter.
It replaces the earlier C++ v3 proposal in upstream PR #16. No two-step alternative or
secondary HingeZero method is registered here.

## Exact evaluated implementation

- Original code and benchmark commit: `60121431afc95979065e1b2ade204a538c3d7be1`
- Exact Python implementation: `vibe/algorithms/hingezero_support/module.py` (source code copied unchanged)
- Dependency: `vibe/algorithms/hingezero_cpp20/module.py` plus `native/HingeZero_VIBE_Cpp20.cpp` (source files copied unchanged)
- Index: single-core FAISS `IndexFlatIP` over the 32,768 highest-frequency corpus vectors
  selected using the **learning split's** provided ground-truth neighbor IDs.
- Query: exact top-100 inner-product search in the selected pool, followed by native
  HingeZero rank/identity verification. HingeZero refinement steps = **0**.
- Dataset: VIBE `yi-128-ip`; evaluation uses **1000 test queries** and recall@100.
- Parameters: `field_size=256`, `pool_size=32768`.
- Registered VIBE algorithm: `hingezero-support-lab`, OOD enabled.
- VIBE evaluation: `python run.py --local --force --runs 2 --parallelism 1 --dataset yi-128-ip --algorithm hingezero-support-lab --count 100`.
- Export: `python export_results.py --parallelism 1 --skip-pca-mahalanobis`.

## Recorded benchmark results and variability

The previous published VIBE harness run on a GitHub Actions CPU runner returned:

- **5,010.826592 QPS**, **98.617% recall@100**, `pool_size=32768`:
  https://github.com/HingeZero/vibe/actions/runs/40445051981
- A later same-configuration runner reported **2,326.520725 QPS** at the
  **same 98.617% recall**, indicating machine/run throughput variability:
  https://github.com/HingeZero/vibe/actions/runs/40445259052

These are VIBE-generated summary scores, not independently certified results
on the VIBE maintainers' reference CPU. They do not guarantee 5,011 QPS on a rerun.
Accuracy may change on other datasets; no cross-dataset claim is made.

The independent OOD learning split's ground-truth neighbor IDs were used in
training the search-support pool, per VIBE's documented `fit_ood` interface.
**Test query labels are never used to build the pool.**

## Submission clarification — 30 October 2026

**This is a VIBE submission, not a NeurIPS Big ANN `openai-100K` leaderboard result.** The previously reported 5,010.826592 QPS / 98.617% recall@100 belongs to VIBE's **`yi-128-ip`** test, using a restricted learned 32,768-vector support pool. It is neither a full-corpus general-purpose result nor an `openai-100K` result. The same configuration reproduced 98.617% recall at lower throughput on other shared runners.

The separate [Big ANN PR #352](https://github.com/harsha-simhadri/big-ann-benchmarks/pull/352) reports a different implementation and evaluation: `openai-100K` recall@10 **0.914250**, **165.889364 QPS**, four threads, eight refinement steps. **Do not substitute either test's QPS/recall for the other.**

**Review request:** This VIBE method builds a support subset from the independent OOD learning split's provided neighbour labels. It does not use test labels. Maintainers should assess whether this training mechanism and restricted-pool search qualify for inclusion under the intended VIBE leaderboard comparison; the score is not certified until reviewed and evaluated on reference hardware.

## Reproduce

Build the HingeZero support image using VIBE's container method:
```bash
./install.sh --algorithm hingezero_support
python3 run.py --dataset yi-128-ip --algorithm hingezero-support-lab --count 100
./export_results.sh --parallelism 1
```

To reproduce the exact pinned GitHub Actions harness and inspect full logs,
run `.github/workflows/hz-support-vibe.yml`.

This is the **single** HingeZero method proposed in PR #16 and it does not
assert independent acceptance, a leaderboard ranking, or a speed advantage
over methods that use the full corpus without a learned restricted pool.


## Single-method PR rerun (source unchanged)

A fresh run after consolidating upstream PR #16 to this one configuration succeeded:
https://github.com/HingeZero/vibe/actions/runs/40495156042

- `yi-128-ip`, `pool_size=32768`, refinement steps **0**
- Recall@100: **98.617%**
- Query throughput on that run: **2,202.23695 QPS**
- Original prior run: **5,010.82659 QPS** at the same recall and code provenance.

The single-source implementation was not tuned or modified to achieve these
numbers. The QPS value varies materially across shared GitHub Actions runs;
the original 5,011 QPS must not be represented as a guaranteed repeated
throughput or an independently verified official leaderboard score.
