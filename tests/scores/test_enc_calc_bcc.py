"""Equivalence test for EffectiveNumberOfCodons._calc_BCC vectorisation.

The vectorised k_mer=1 path must match the pre-vectorisation listcomp
implementation for arbitrary BNC distributions.
"""

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import EffectiveNumberOfCodons


def _reference_calc_bcc(enc, BNC):
    """Pre-vectorisation implementation, inlined here as the reference."""
    BCC = enc.template.copy()
    BCC["bcc"] = [
        np.prod([BNC[c] for c in cod]) for cod in BCC.index.get_level_values("codon")
    ]
    BCC = BCC["bcc"]
    BCC /= BCC.groupby("aa").sum()
    return BCC


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_calc_bcc_equivalence_random_bnc(seed):
    enc = EffectiveNumberOfCodons(bg_correction=True)
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([1, 1, 1, 1])
    BNC = pd.Series(probs, index=list("ACGT"))

    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)

    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)
    assert got.index.equals(expected.index)


def test_calc_bcc_equivalence_uniform_bnc():
    """Uniform BNC (equivalent to the BCC_unif init path)."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    BNC = pd.Series([0.25] * 4, index=list("ACGT"))
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)


def test_calc_bcc_equivalence_skewed_bnc():
    """Extreme skew: one base dominates, another ~0."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    BNC = pd.Series([0.7, 0.28, 0.01, 0.01], index=list("ACGT"))
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)


def test_calc_bcc_sums_to_one_per_aa():
    """BCC is normalised within each amino acid group: each group sums to 1."""
    enc = EffectiveNumberOfCodons(bg_correction=True)
    rng = np.random.default_rng(0)
    BNC = pd.Series(rng.dirichlet([1, 1, 1, 1]), index=list("ACGT"))
    bcc = enc._calc_BCC(BNC)
    aa_sums = bcc.groupby("aa").sum()
    np.testing.assert_allclose(aa_sums.values, 1.0, rtol=1e-12)


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_calc_bcc_equivalence_other_genetic_codes(genetic_code):
    """Non-standard genetic codes have different aa→codon groupings."""
    enc = EffectiveNumberOfCodons(bg_correction=True, genetic_code=genetic_code)
    rng = np.random.default_rng(genetic_code)
    BNC = pd.Series(rng.dirichlet([1, 1, 1, 1]), index=list("ACGT"))
    got = enc._calc_BCC(BNC)
    expected = _reference_calc_bcc(enc, BNC)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)
