# [v0.5.0](https://github.com/alondmnt/codon-bias/releases/tag/v0.5.0)
*Released on 2026-05-16*

if v0.4.0 moved the k_mer=1 hot paths from pandas to numpy, this release
does the same for k_mer>1. `CodonCounter.count_array` is now a single
vectorised entry point for any k_mer in [1, 3], and every score routes
through it on the hot path. `count(seqs)` becomes a thin formatter on top.

alongside the perf work, the package consolidates a few subclass-as-config
patterns into single classes with a kwarg (`WeightOptimizer(strategy=...)`,
`Permuter(scope=...)`), promotes underscore-prefixed counter internals
onto the documented public surface that scores depend on, and factors
the str/list/ndarray dispatch shared by `get_score` / `get_vector` /
`get_weights` into a single `Score._dispatch` helper.

per-call timings on a 3.6 kb sequence (200 calls, mean):

| score / path                          | before    | after    | speedup |
|---------------------------------------|----------:|---------:|--------:|
| ENC k_mer=2, bg_correction=True       | 22.9 ms   | 0.21 ms  |  ~110x  |
| CAI k_mer=2                           | 0.82 ms   | 0.05 ms  |   ~16x  |
| RSCU                                  | ~0.22 ms  | 0.016 ms |   ~14x  |
| `CodonCounter.count(seq).counts` k=2  | 0.15 ms   | 0.04 ms  |  ~3.5x  |
| RCB                                   | ~0.15 ms  | 0.04 ms  |  ~3.5x  |
| ENC k_mer=1 (default, bg_correction)  | (vectorised in v0.4.0)        | within noise |
| CUFS.get_score                        | 1.02 ms   | 0.042 ms |   ~24x  |
| CUFS.get_score synonymous=True        | 1.77 ms   | 0.045 ms |   ~39x  |

(the ~110x on ENC k_mer=2 came from replacing a python listcomp over
3,721 codon-pair strings with `BNC[codon_base_idx_kmer].prod(axis=1)`.)

## performance

- `CodonCounter.count_array` generalised to k_mer in [1, 3] (#27):
  sliding window over codon ids, combined into a single bucket id per
  k-mer, bincounted into a dense aligned ndarray.
- `CodonCounter.count(seqs)` becomes a thin formatter on top of
  `count_array` (#27); k-mer concat-string index built lazily via the
  new `kmer_index` property.
- CAI and CodonPairBias k_mer>1 wired to `count_array` with pre-aligned
  log/linear weight arrays (#27), skipping the per-call pandas reindex.
- RSCU and RCBS rewritten as stateless ndarray calls on `count_array`
  with weights pre-aligned at init (#24).
- ENC k_mer=1 / k_mer>1 paths unified around `count_array` (#28); the
  two `_calc_*_single_kmer` / `_calc_*` methods collapse to one body
  driven by `aa_group_kmer` / `codon_base_idx_kmer` LUTs.
- `BaseCounter.count_array` vectorised for any k_mer (#27); sliding
  window over base ids respects `frame` / `step` semantics.
- `geomean_array` / `mean_array` helpers in `utils` for the count_array
  hot path (#26, #27), aligned-ndarray siblings of the pandas
  `geomean` / `mean`.
- `pairwise.CodonUsageFrequency` rewritten on `count_array`: per-seq
  arrays stacked, normalised with `np.add.at` per-aa-group sums
  (synonymous) or a single divide (non-synonymous). count_array order
  replaces the prior alphabetic codon order; KL pair score is invariant
  under consistent reordering.

## refactors

- `MaxWeight` / `MinWeight` / `BalancedWeight` collapsed into
  `WeightOptimizer(strategy="max"|"min"|"balanced")` (#20). old names
  emit `FutureWarning` via shims; will be removed in v0.6.0.
- `IntraSeqPermuter` / `IntraPosPermuter` collapsed into
  `Permuter(scope="intra_seq"|"intra_pos")` (#23). old names emit
  `FutureWarning` via shims; will be removed in v0.6.0.
- counter internals promoted to the public surface that scores depend
  on (#22, #25, #27, #28): `count_array`, `aa_group`, `n_aa`,
  `codon_index`, `codon_base_idx`, `kmer_index`, `aa_group_kmer`,
  `codon_base_idx_kmer`. `_codon_lex_to_aa` renamed to
  `_codon_lex_to_idx` (it stored codon indices, not aa indices).
- temporal coupling fixed in `RelativeSynonymousCodonUsage` and
  `RelativeCodonBiasScore` (#24): `_calc_score` no longer relies on a
  prior `_calc_seq_weights` populating `self.counter.counts`, which
  broke concurrent use.
- `Score._dispatch` shared by `ScalarScore.get_score`,
  `VectorScore.get_vector` and `WeightScore.get_weights`; each base
  previously carried its own near-identical str/list/ndarray dispatch
  shell. public API unchanged.
- `pairwise.PairwiseScore.get_matrix` n_jobs=1 path fixed: the
  sequential `starmap` branch was unconditionally overwritten by the
  next-line `Pool.starmap`, so n_jobs=1 silently paid fork overhead
  with no parallelism. output unchanged.

## breaking changes

- `CodonCounter` and `BaseCounter` no longer accept `k_mer >= 4` for
  `count_array` (which now drives `count()` too); the python `Counter`
  fallback is retired (#27). no in-package score uses k_mer > 2; the
  documented k-mer support is now [1, 3]. attempts above raise
  `NotImplementedError` with an explicit message.
- `MaxWeight` / `MinWeight` / `BalancedWeight` (#20) and
  `IntraSeqPermuter` / `IntraPosPermuter` (#23) now emit
  `FutureWarning` on instantiation. behaviour is preserved via shims;
  the names will be removed in v0.6.0.
- RSCU, RCBS and CAI no longer populate `self.counter.counts` as a
  side effect of `get_score`/`get_weights` (#24). callers reading
  `counter.counts` after a score call should either call
  `counter.count(seqs)` explicitly or use the score's public weight
  output.

## tooling

- CI runtime cut from ~13 min to ~3 min (#21). E. coli regression
  tests subsample to 500 sequences by default (set `ECOLI_FULL=1` to
  run on the full corpus); python 3.8 dropped from the matrix.
- RSCU regression baseline added against pre-deep-modules main
  (commit 2bc54b3). pins the four `(directional, mean)` combinations
  on the first 500 E. coli sequences (#25).
- property tests for `CodonPairBias` via `hypothesis`, covering the
  k_mer=2 `count_array` / weights path against the pre-refactor
  pandas reference.
- `CHANGELOG.md` rendered into the Sphinx site via `myst-parser`, so
  the readthedocs build now publishes the release notes alongside the
  API reference.
- `issue_module_depth.md` (local plan) drove the deepening work
  through eight candidates; all closed in this release.

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.4.0...v0.5.0

---

# [v0.4.0](https://github.com/alondmnt/codon-bias/releases/tag/v0.4.0)
*Released on 2026-05-02*

this release moves the scalar-score hot paths from pandas to numpy, with
follow-on vectorisation of ENC's `bg_correction` path and a partial
vectorisation of RCB. on E. coli K-12: ~47x on ENC default, ~25x on
FOP/CAI/tAI/nTE, ~40x on ENC with `bg_correction=True`, 1.7x on RCB.
public APIs are unchanged; two opt-in code paths have small behaviour
changes (see breaking).

big thanks to @RedPenguin100, who set up the regression test suite and
CI that this release was built on, and kicked off the numpy migration
with the first ENC rewrite (PR #13) that everything else followed from.

## performance

- scoring pipeline moved to numpy across ENC, FOP, CAI, tAI, nTE
  (#8). PR #13 by @RedPenguin100 rewrote `EffectiveNumberOfCodons` on
  ndarrays and seeded the `_count_single` numpy return for `k_mer=1`;
  the #8 closing commits extended the same approach to FOP, CAI, tAI,
  and nTE, with weights pre-aligned at init.
- `CodonCounter._count_single` (k_mer=1) vectorised via a base-5 packed
  codon LUT and one numpy advanced-indexing op.
- ENC `bg_correction=True` vectorised (#18). `_calc_BCC` precomputes a
  codon-to-base-index matrix; `_calc_BNC` reaches through to
  `BaseCounter._count_single`, skipping pandas scaffolding.
- RCB per-sequence background partially vectorised (refs #19).

| score                    | before | after  | total |
|--------------------------|-------:|-------:|------:|
| ENC (default)            | ~6.8 s | 146 ms |  ~47x |
| FOP / CAI / tAI / nTE    | ~1.0 s |  40 ms |  ~25x |
| ENC (bg_correction=True) | ~9.8 s | 246 ms |  ~40x |
| RCB.get_score            | 4.78 s | 2.85 s |  1.7x |

(of the ~40x on `bg_correction=True`, ~3x came from #8 - which left it
at 3.1 s, dominated by `BaseCounter._count_single` and `_calc_BCC` -
and the remaining 11.9x from #18.)

## fixes

- ENC weighted mean now filters undersampled amino acids (#15).
  previously, pseudocount-only AAs could dilute weighted scores toward
  zero; on the `bg_correction` path, F=inf could poison degeneracy
  groups (silently capped by the `min(len(P), ENC)` guard).
- centralised in-frame codon iteration via `utils.iter_codons`. fixes a
  latent crash in `Permuter._preprocess_seq` on non-multiple-of-3 input
  and tightens `k_mer >= 2` iteration so trailing partial k-mers are no
  longer emitted.
- `Permuter` RNG modernised (#17): per-group independent PCG64 streams
  via `np.random.default_rng`, replacing `np.random.seed` calls that
  leaked into global state and re-seeded each group identically.
- `fetch_GCN_from_GtRNAdb` sets a descriptive User-Agent so GtRNAdb's
  bot-filter no longer 403s fresh installs.

## breaking changes

- `Permuter` permutation output is bit-exactly different for any given
  `random_state` (#17). statistical correctness is improved (independent
  per-group streams) - downstream z-scores/p-values should be more
  accurate. permutations are stable for a given input but not against
  input perturbation (adding/removing sequences shifts group indices).
- `get_vector` / `_calc_vector` for `k_mer >= 2` returns a vector
  `k_mer - 1` elements shorter (the previously-NaN trailing slots for
  partial k-mers are no longer emitted). element values at retained
  positions are identical. callers depending on
  `len(vector) == len(seq) // 3` should update to
  `len(vector) == len(seq) // 3 - (k_mer - 1)`.
- ENC weighted-mean filter (#15) changes scores for
  `EffectiveNumberOfCodons(mean="weighted")` with `robust=False` and/or
  `pseudocount=0` on sequences with undersampled AAs. default
  configuration on well-sampled sequences is unaffected.

## tooling

- GitHub Actions CI with pytest regression tests on a vendored E. coli
  K-12 corpus (FOP, CAI, tAI, nTE, ENC), parallelised with caching
  (PRs #10, #12, #14 by @RedPenguin100).
- `ruff` for formatting and lint, enforced on push (PR #16).

## acknowledgements

- @RedPenguin100 - landed the test suite and CI (#10, #12, #14) that
  made the rest of this release safe to do, and the first numpy ENC
  rewrite (#13) that the wider migration built on.

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.3.5...v0.4.0

---

# [v0.3.5](https://github.com/alondmnt/codon-bias/releases/tag/v0.3.5)
*Released on 2025-03-13T09:38:20Z*

- improved: input type validation

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.3.4...v0.3.5

---

# [v0.3.4](https://github.com/alondmnt/codon-bias/releases/tag/v0.3.4)
*Released on 2025-03-06T12:17:31Z*

- added: FrequencyOfOptimalCodons: weights arg (closes #5)
- fixed: pandas v2 compatibility (closes #5)
- improved: FrequencyOfOptimalCodons: default thresh 0.95 -> 0.8

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.3.3...v0.3.4

---

# [v0.3.3](https://github.com/alondmnt/codon-bias/releases/tag/v0.3.3)
*Released on 2025-02-11T02:04:11Z*

## What's Changed
* Use `np.prod` instead of `np.product` by @l-benedetti-insta in https://github.com/alondmnt/codon-bias/pull/4

## New Contributors
* @l-benedetti-insta made their first contribution in https://github.com/alondmnt/codon-bias/pull/4

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.3.2...v0.3.3

---

# [v0.3.2](https://github.com/alondmnt/codon-bias/releases/tag/v0.3.2)
*Released on 2024-12-20T04:55:25Z*

## What's Changed
* tAI fixes/improvements by @moritzburghardt in https://github.com/alondmnt/codon-bias/pull/3
    - new: optimize_s_values method
    - new: allow custom s_values
    - fixed: handle invalid literal for int in gtrnadb table
    - fixed: CERTIFICATE_VERIFY_FAILED error on GtRNAdb

## New Contributors
* @moritzburghardt made their first contribution in https://github.com/alondmnt/codon-bias/pull/3

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/v0.3.1...v0.3.2

---

# [v0.3.1](https://github.com/alondmnt/codon-bias/releases/tag/v0.3.1)
*Released on 2023-06-14T15:53:08Z*

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/0.3.0...v0.3.1

---

# [v0.3.0](https://github.com/alondmnt/codon-bias/releases/tag/0.3.0)
*Released on 2022-10-28T17:10:42Z*

- new:
    - new module: `optimizers` with the classes MaxWeight, MinWeight, and BalancedWeight
    - new module: `random` with the classes Permuter, IntraSeqPermuter and IntraPosPermuter
    - added utils.ReferenceSelector
- changed:
    - utils.translate returns a dataframe by default
- fixed:
    - avoid numpy deprecation warning in VectorScore's get_vector function

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/0.2.0...0.3.0

---

# [v0.2.0](https://github.com/alondmnt/codon-bias/releases/tag/0.2.0)
*Released on 2022-09-13T15:58:19Z*

- new:
    - added scores.NormalizedTranslationalEfficiency
    - added scores.CodonPairBias
    - added stats.BaseCounter for nucleotide and k-mer statistics across reading frames
    - added `k_mer` parameter to:
        - stats.CodonCounter
        - scores.CodonAdaptationIndex
        - scores.EffectiveNumberOfCodons
        - pairwise.CodonUsageFrequency
    - added abstract class scores.WeightScore that computes a weight vector for each input sequence, with the following children:
        - scores.CodonPairBias
        - scores.EffectiveNumberOfCodons
        - scores.RelativeSynonymousCodonUsage
        - scores.RelativeCodonBiasScore
- improved:
    - various improvements to scores.EffectiveNumberOfCodons
        - background correction
        - improved estimation
    - added count() method to counter classes
    - added `pseudocount` parameter to models

**Full Changelog**: https://github.com/alondmnt/codon-bias/compare/0.1.0...0.2.0

---

# [v0.1.0](https://github.com/alondmnt/codon-bias/releases/tag/0.1.0)
*Released on 2022-08-27T19:54:54Z*

First release.

- stats.CodonCounter
- scores.FrequencyOfOptimalCodons (FOP)
- scores.RelativeSynonymousCodonUsage (RSCU)
- scores.CodonAdaptationIndex (CAI)
- scores.EffectiveNumberOfCodons (ENC)
- scores.TrnaAdaptationIndex (tAI)
- scores.RelativeCodonBiasScore (RCBS + DCBS)
- pairwise.CodonUsageFrequency (CUFS)

---
