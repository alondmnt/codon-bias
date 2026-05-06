"""Regression test for RSCU scores on E. coli.

Mirrors `tests/scores/test_rcb_regression.py`. Pins `get_score` output
across the four `(directional, mean)` combinations on the first 500
E. coli sequences. The slice is explicit so the same set of inputs is
used regardless of `ECOLI_FULL` env var or the conftest's default
subsetting; baselines were generated from the pre-deep-modules main
(commit `2bc54b3`) and verify that the candidate-3 / candidate-4 /
RSCU stateless rewrite (PR 24) and subsequent counter changes have
not introduced numerical drift.
"""

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import RelativeSynonymousCodonUsage

RSCU_COMBINATIONS = [
    {"directional": False, "mean": "geometric"},
    {"directional": False, "mean": "arithmetic"},
    {"directional": True, "mean": "geometric"},
    {"directional": True, "mean": "arithmetic"},
]


def format_param_id(p):
    return f"dir_{p['directional']}-{p['mean']}"


@pytest.mark.parametrize("params", RSCU_COMBINATIONS, ids=format_param_id)
def test_rscu_ecoli_regression(ecoli_seqs, dataframe_regression, params):
    """Regression test: RSCU scores on E. coli across parameter combinations."""
    rscu = RelativeSynonymousCodonUsage(**params)

    seqs = ecoli_seqs[:500]
    scores = rscu.get_score(seqs)

    df = pd.DataFrame(
        {
            "gene_index": np.arange(len(seqs)),
            "score": np.round(scores, 6),
        }
    )
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)
