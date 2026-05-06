"""Equivalence test for EffectiveNumberOfCodons._calc_BCC.

The vectorised path must match a python-level reference for arbitrary
BNC distributions. ``_calc_BCC`` returns an ndarray in ``kmer_index``
order, normalised within each aa-tuple group; covered for k_mer in
[1, 2].
"""

import numpy as np
import pytest

from codonbias.scores import EffectiveNumberOfCodons


def _reference_calc_bcc(enc, BNC):
    """Python-level reference: per-k-mer base-product, aa-group renormalisation."""
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    kmer_index = enc.counter.kmer_index
    aa_group = enc.counter.aa_group_kmer
    n_aa_kmer = enc._aa_deg.size

    bcc = np.array(
        [np.prod([BNC[base_to_idx[b]] for b in kmer]) for kmer in kmer_index],
        dtype=float,
    )
    aa_sums = np.bincount(aa_group, weights=bcc, minlength=n_aa_kmer)
    return bcc / aa_sums[aa_group]


@pytest.mark.parametrize("k_mer", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_calc_bcc_equivalence_random_bnc(seed, k_mer):
    enc = EffectiveNumberOfCodons(k_mer=k_mer, bg_correction=True)
    rng = np.random.default_rng(seed)
    BNC = rng.dirichlet([1, 1, 1, 1])

    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)

    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("k_mer", [1, 2])
def test_calc_bcc_equivalence_uniform_bnc(k_mer):
    """Uniform BNC (equivalent to the BCC_unif init path)."""
    enc = EffectiveNumberOfCodons(k_mer=k_mer, bg_correction=True)
    BNC = np.array([0.25] * 4)
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("k_mer", [1, 2])
def test_calc_bcc_equivalence_skewed_bnc(k_mer):
    """Extreme skew: one base dominates, another ~0."""
    enc = EffectiveNumberOfCodons(k_mer=k_mer, bg_correction=True)
    BNC = np.array([0.7, 0.28, 0.01, 0.01])
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("k_mer", [1, 2])
def test_calc_bcc_sums_to_one_per_aa(k_mer):
    """BCC is normalised within each aa-tuple group: each group sums to 1."""
    enc = EffectiveNumberOfCodons(k_mer=k_mer, bg_correction=True)
    rng = np.random.default_rng(0)
    BNC = rng.dirichlet([1, 1, 1, 1])
    bcc = enc._calc_BCC(BNC)
    aa_group = enc.counter.aa_group_kmer
    aa_sums = np.bincount(aa_group, weights=bcc, minlength=enc._aa_deg.size)
    # Drop empty groups (degenerate aa-tuples that don't occur).
    np.testing.assert_allclose(aa_sums[enc._aa_deg > 0], 1.0, rtol=1e-12)


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_calc_bcc_equivalence_other_genetic_codes(genetic_code):
    """Non-standard genetic codes have different aa->codon groupings."""
    enc = EffectiveNumberOfCodons(bg_correction=True, genetic_code=genetic_code)
    rng = np.random.default_rng(genetic_code)
    BNC = rng.dirichlet([1, 1, 1, 1])
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
