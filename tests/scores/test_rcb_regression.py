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
