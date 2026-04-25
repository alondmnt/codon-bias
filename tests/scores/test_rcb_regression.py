"""Regression test for RCB scores and weights on E. coli.

Mirrors tests/scores/test_enc_regression.py. Captures the numerical
output of the vectorised #19 implementation so future changes can be
checked against it.
"""

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import RelativeCodonBiasScore

RCB_COMBINATIONS = [
    {"directional": False, "mean": "geometric"},
    {"directional": True, "mean": "arithmetic"},
]


def format_param_id(p):
    return f"dir_{p['directional']}-{p['mean']}"


@pytest.mark.parametrize("params", RCB_COMBINATIONS, ids=format_param_id)
def test_rcb_ecoli_regression(ecoli_seqs, dataframe_regression, params):
    """Regression test: RCB scores on E. coli across parameter combinations."""
    rcb = RelativeCodonBiasScore(**params)

    scores = rcb.get_score(ecoli_seqs)

    df = pd.DataFrame(
        {
            "gene_index": np.arange(len(ecoli_seqs)),
            "score": np.round(scores, 6),
        }
    )
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)


def test_rcb_basic_logic():
    """Fundamental scoring — finite, defined on standard and edge inputs."""
    rcb = RelativeCodonBiasScore()
    assert np.isfinite(rcb.get_score("ATGCGTACG"))
    # Single-codon sequences are a degenerate case — shouldn't crash.
    assert np.isfinite(rcb.get_score("ATGATGATGATG"))


def test_rcb_multiple_input_types():
    rcb = RelativeCodonBiasScore()
    seqs = ["ATGCGTACG", "ATGATGATG"]
    scores = rcb.get_score(seqs)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2


@pytest.mark.parametrize("genetic_code", [2, 11])
@pytest.mark.parametrize("params", RCB_COMBINATIONS, ids=format_param_id)
def test_rcb_non_standard_genetic_code(genetic_code, params):
    """End-to-end smoke check on non-standard genetic codes.

    RCB's `_calc_BCC` is structurally code-independent (iterates the full
    64-codon lex product regardless), but the counter side of the
    pipeline (`get_codon_table` / `P`) depends on `genetic_code`. This
    test exercises the integration on codes 2 and 11.
    """
    rcb = RelativeCodonBiasScore(genetic_code=genetic_code, **params)
    seqs = ["ATGCGTACG" * 10, "ATGAAACCCGGGTTT" * 5, "ATGATGATGATG"]
    scores = rcb.get_score(seqs)
    assert scores.shape == (len(seqs),)
    assert np.isfinite(scores).all()
