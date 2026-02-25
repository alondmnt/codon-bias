import pytest

import numpy as np
import pandas as pd
import time
from numpy.testing import assert_allclose

from codonbias.scores import EffectiveNumberOfCodons


def test_enc_ecoli_regression(ecoli_seqs, dataframe_regression):
    """
    Regression test comparing ENC scores on E. coli genes across multiple
    parameter combinations using pytest-regressions.
    """
    results = []

    # The distinct logic branches we want to ensure stay mathematically identical
    combinations = [
        {"bg_correction": False, "robust": True, "mean": "weighted"},
        {"bg_correction": False, "robust": False, "mean": "unweighted"},
        {"bg_correction": True, "robust": True, "mean": "weighted"},
    ]

    for params in combinations:
        enc = EffectiveNumberOfCodons(**params)

        # Calculate scores for all sequences at once
        scores = enc.get_score(ecoli_seqs)

        # Format the results into a flat list of dictionaries for Pandas
        for i, score in enumerate(scores):
            results.append({
                "bg_correction": params["bg_correction"],
                "robust": params["robust"],
                "mean": params["mean"],
                "gene_index": i,
                "score": score
            })

    df = pd.DataFrame(results)

    # Round the scores to 6 decimal places to prevent floating-point micro-inconsistencies
    # between CPU architectures or Pandas vs NumPy internal float handling
    df["score"] = df["score"].round(6)

    # Sort to guarantee the exact same row order on every machine
    df = df.sort_values(by=["bg_correction", "robust", "mean", "gene_index"]).reset_index(drop=True)

    # Generates a reference CSV on the first run, and asserts against it on future runs
    dataframe_regression.check(df)


@pytest.fixture
def enc_default():
    """Provides a default EffectiveNumberOfCodons instance."""
    return EffectiveNumberOfCodons()


@pytest.fixture
def random_seq_gen():
    """Factory fixture to generate random DNA sequences of a given length."""
    rng = np.random.default_rng()

    def _generate(length, seed=None, p=None):
        nonlocal rng
        if seed is not None:
            rng = np.random.default_rng(seed)

        bases = np.array(['A', 'C', 'G', 'T'])
        return ''.join(rng.choice(bases, size=length, p=p))

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


def test_enc_dataframe_regression(enc_default, random_seq_gen, dataframe_regression):
    """
    Explicit regression test with 1000 sequences of varying ENC degrees.
    Generates sequences with different GC contents to ensure a range
    of scores (high and low bias) are tested and matched.
    """
    # Use a local generator just for the lengths
    length_rng = np.random.default_rng(42)

    all_scores = []

    for i in range(1000):
        bias_factor = (i % 10) / 10.0
        p = np.array([0.25, 0.25, 0.25, 0.25])

        if i % 2 == 0:
            p = np.array([0.1 + 0.4 * bias_factor, 0.4 - 0.3 * bias_factor, 0.2, 0.3])
            p /= p.sum()

        length = length_rng.integers(100, 500) * 3

        # Seed the fixture on the first iteration to lock in reproducibility
        current_seed = 42 if i == 0 else None

        # Generate the sequence via the fixture
        seq = random_seq_gen(length, seed=current_seed, p=p)

        score = enc_default.get_score(seq)
        all_scores.append(score)

    data = pd.DataFrame({
        "iteration": np.arange(1000),
        "enc_score": all_scores
    })

    dataframe_regression.check(data)

@pytest.mark.parametrize("pseudocount, result", [(0, 35.), (1, 44.063646)])
def test_enc_missing_3fold_deg(pseudocount, result):
    """
    Regression test for the 'degree 3 imputation'
    Isoleucine is the only 3-fold degenerate amino acid. If a sequence lacks it,
    ENC dictates its F-value should be imputed as the average of 2-fold and
    4-fold degenerate F-values.
    """
    enc_no_pseudocount = EffectiveNumberOfCodons(pseudocount=pseudocount)
    # We create a highly biased sequence containing ONLY Phenylalanine (2-fold)
    # and Alanine (4-fold). Isoleucine (3-fold) is completely absent.
    # TTT = Phenylalanine, GCT = Alanine
    seq_missing_ile = ("TTT" * 20) + ("GCT" * 20)

    score_numpy = enc_no_pseudocount.get_score(seq_missing_ile)

    assert_allclose(score_numpy, result)
