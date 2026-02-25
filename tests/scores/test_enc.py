import pytest
import numpy as np
import pandas as pd
import time
from numpy.testing import assert_allclose

from codonbias.scores import EffectiveNumberOfCodons


@pytest.fixture
def enc_default():
    """Provides a default EffectiveNumberOfCodons instance."""
    return EffectiveNumberOfCodons()


@pytest.fixture
def random_seq_gen():
    """Factory fixture to generate random DNA sequences of a given length."""

    def _generate(length, seed=None):
        if seed is not None:
            np.random.seed(seed)
        bases = ['A', 'C', 'G', 'T']
        return ''.join(np.random.choice(bases, size=length))

    return _generate


def test_enc_basic_logic(enc_default):
    """Verifies fundamental scoring for standard, biased, and edge cases."""
    # Standard multi-codon sequence
    assert_allclose(enc_default.get_score("ATGCGTACG"), 59.792271, rtol=1e-5)

    # Empty string / NaN handling
    assert_allclose(enc_default.get_score(""), 61.0, rtol=1e-5)

    # Extreme bias (handled by pseudocount logic)
    assert_allclose(enc_default.get_score("ATGATGATGATG"), 61.0, rtol=1e-5)

    # Handling non-standard characters (should not crash)
    assert np.isfinite(enc_default.get_score("ATGCGNATGCGT"))


def test_enc_multiple_input_types(enc_default):
    """Verifies array-like inputs return numpy arrays."""
    seqs = ["ATGCGTACG", "ATGATGATG"]
    scores = enc_default.get_score(seqs)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2


@pytest.mark.parametrize("robust", [True, False])
@pytest.mark.parametrize("mean", ["weighted", "unweighted"])
def test_enc_parameters(robust, mean):
    """Checks all logic branches for robustness and mean calculation."""
    enc = EffectiveNumberOfCodons(robust=robust, mean=mean)
    seq = "ATGCGTACGACGTGA"
    score = enc.get_score(seq)
    assert np.isfinite(score)
    assert score > 0


def test_enc_dataframe_regression(enc_default, dataframe_regression):
    """
    Explicit regression test with 1000 sequences of varying ENC degrees.
    Generates sequences with different GC contents to ensure a range
    of scores (high and low bias) are tested and matched.
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    bases = np.array(['A', 'C', 'G', 'T'])

    all_scores = []

    # Generate 1000 sequences with varying bias
    for i in range(1000):
        # We vary the probability distribution to get different ENC results
        # Some iterations will be very biased, others will be uniform
        bias_factor = (i % 10) / 10.0  # Cycle through bias levels
        p = np.array([0.25, 0.25, 0.25, 0.25])

        # Shift probability to create high/low ENC scenarios
        if i % 2 == 0:
            p = np.array([0.1 + 0.4 * bias_factor, 0.4 - 0.3 * bias_factor, 0.2, 0.3])
            p /= p.sum()

        length = rng.integers(100, 500) * 3
        seq = "".join(rng.choice(bases, size=length, p=p))

        score = enc_default.get_score(seq)
        all_scores.append(score)

    # We do not store the sequences to keep the regression file small
    # Just the scores which represent the mathematical output
    data = pd.DataFrame({
        "iteration": np.arange(1000),
        "enc_score": all_scores
    })

    # The first time this runs, it creates a baseline file.
    # Subsequent runs compare against that file.
    dataframe_regression.check(data)
