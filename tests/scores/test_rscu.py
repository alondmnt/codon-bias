"""Regression and contract tests for RelativeSynonymousCodonUsage.

The RSCU module previously had no dedicated tests; this file covers
the basics: end-to-end finite output across parameter combinations,
and the stateless contract on `self.counter` introduced when the
score path moved to `count_array`.
"""

import numpy as np
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
def test_rscu_basic_logic(params):
    rscu = RelativeSynonymousCodonUsage(**params)
    assert np.isfinite(rscu.get_score("ATGCGTACG"))
    # Single-aa repeated codons are a degenerate but well-defined case.
    assert np.isfinite(rscu.get_score("ATGATGATGATG"))


def test_rscu_multiple_input_types():
    rscu = RelativeSynonymousCodonUsage()
    seqs = ["ATGCGTACG", "ATGATGATG"]
    scores = rscu.get_score(seqs)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2


def test_rscu_does_not_mutate_counter():
    """Stateless contract: get_score must not populate self.counter.counts."""
    rscu = RelativeSynonymousCodonUsage()
    rscu.get_score("ATGAAACCCGGGTTT")
    assert not hasattr(rscu.counter, "counts")


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_rscu_non_standard_genetic_code(genetic_code):
    rscu = RelativeSynonymousCodonUsage(genetic_code=genetic_code)
    seqs = ["ATGCGTACG" * 10, "ATGAAACCCGGGTTT" * 5, "ATGATGATGATGATG"]
    scores = rscu.get_score(seqs)
    assert scores.shape == (len(seqs),)
    assert np.isfinite(scores).all()
