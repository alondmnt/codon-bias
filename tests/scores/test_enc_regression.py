import pytest

import numpy as np
import pandas as pd
import time
from numpy.testing import assert_allclose

from codonbias.scores import EffectiveNumberOfCodons

# Define combinations outside the test function
ECOLI_COMBINATIONS = [
    {"bg_correction": False, "robust": True, "mean": "weighted"},
    {"bg_correction": False, "robust": False, "mean": "unweighted"},
    {"bg_correction": True, "robust": True, "mean": "weighted"},
]

def format_param_id(p):
    """Formats the dictionary into a clean, descriptive string for pytest."""
    return f"bg_{p['bg_correction']}-rob_{p['robust']}-{p['mean']}"

@pytest.mark.parametrize("k_mer", [1, 2], ids=["kmer1", "kmer2"])
@pytest.mark.parametrize("params", ECOLI_COMBINATIONS, ids=format_param_id)
def test_enc_ecoli_regression(ecoli_seqs, dataframe_regression, k_mer, params):
    """
    Regression test comparing ENC scores and weights on E. coli genes across multiple
    parameter combinations using pytest-regressions.
    """
    results = []

    # Initialize with the specific parameters injected by pytest
    enc = EffectiveNumberOfCodons(k_mer=k_mer, **params)

    # Calculate scores and weights for all sequences at once
    scores = enc.get_score(ecoli_seqs)
    weights = enc.get_weights(ecoli_seqs)

    # Format the results into a flat list of dictionaries for Pandas
    for i, (score, weight_series) in enumerate(zip(scores, weights)):
        row = {
            "gene_index": i,
            "score": score
        }

        # Add individual amino acid weights to the row
        row.update({f"weight_{idx}": w for idx, w in enumerate(weight_series)})
        results.append(row)

    df = pd.DataFrame(results)

    # Round the scores and weights to 6 decimal places to prevent floating-point micro-inconsistencies
    cols_to_round = ["score"] + [col for col in df.columns if col.startswith("weight_")]
    df[cols_to_round] = df[cols_to_round].round(6)

    # Sort to guarantee the exact same row order
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    # Generates a reference CSV for THIS specific parameter combination
    dataframe_regression.check(df)

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
