"""Equivalence test for EffectiveNumberOfCodons._calc_BCC vectorisation.

The vectorised k_mer=1 path must match the pre-vectorisation listcomp
implementation for arbitrary BNC distributions. After #18 the k_mer=1
path takes an ndarray in ACGT order; the reference still uses a Series
for its Series-indexed listcomp.
"""

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import EffectiveNumberOfCodons


def _reference_calc_bcc(enc, BNC_series):
    """Pre-vectorisation implementation (Series-indexed), inlined as reference."""
    BCC = enc.template.copy()
    BCC["bcc"] = [
        np.prod([BNC_series[c] for c in cod])
        for cod in BCC.index.get_level_values("codon")
    ]
    BCC = BCC["bcc"]
    BCC /= BCC.groupby("aa").sum()
    return BCC


def _probs_to_pair(probs):
    """Return (ndarray, Series) forms of the same BNC distribution."""
    return np.asarray(probs, dtype=float), pd.Series(probs, index=list("ACGT"))


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_calc_bcc_equivalence_random_bnc(seed):
    enc = EffectiveNumberOfCodons(bg_correction=True)
    rng = np.random.default_rng(seed)
    BNC_arr, BNC_ser = _probs_to_pair(rng.dirichlet([1, 1, 1, 1]))

    got = enc._calc_BCC(BNC_arr)
    expected = _reference_calc_bcc(enc, BNC_ser)

    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)
    assert got.index.equals(expected.index)


def test_calc_bcc_equivalence_uniform_bnc():
    """Uniform BNC (equivalent to the BCC_unif init path)."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    BNC_arr, BNC_ser = _probs_to_pair([0.25] * 4)
    got = enc._calc_BCC(BNC_arr)
    expected = _reference_calc_bcc(enc, BNC_ser)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)


def test_calc_bcc_equivalence_skewed_bnc():
    """Extreme skew: one base dominates, another ~0."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    BNC_arr, BNC_ser = _probs_to_pair([0.7, 0.28, 0.01, 0.01])
    got = enc._calc_BCC(BNC_arr)
    expected = _reference_calc_bcc(enc, BNC_ser)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)


def test_calc_bcc_sums_to_one_per_aa():
    """BCC is normalised within each amino acid group: each group sums to 1."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    rng = np.random.default_rng(0)
    BNC_arr, _ = _probs_to_pair(rng.dirichlet([1, 1, 1, 1]))
    bcc = enc._calc_BCC(BNC_arr)
    aa_sums = bcc.groupby("aa").sum()
    np.testing.assert_allclose(aa_sums.values, 1.0, rtol=1e-12)


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_calc_bcc_equivalence_other_genetic_codes(genetic_code):
    """Non-standard genetic codes have different aa→codon groupings."""
    enc = EffectiveNumberOfCodons(bg_correction=True, genetic_code=genetic_code)
    rng = np.random.default_rng(genetic_code)
    BNC_arr, BNC_ser = _probs_to_pair(rng.dirichlet([1, 1, 1, 1]))
    got = enc._calc_BCC(BNC_arr)
    expected = _reference_calc_bcc(enc, BNC_ser)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)
