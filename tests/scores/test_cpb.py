"""Property tests for CodonPairBias.

CPB had no test coverage prior to this file. These tests don't pin a
specific perf or implementation strategy; they lock observable
properties: re-fits are stateless across calls, get_score is unaffected
by intervening get_weights, and the init-time weights vector has the
expected shape.
"""

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import CodonPairBias


@pytest.fixture
def ref_seq():
    return [
        "ATGAAGCGTGAAATGGCTCTGGAGCAGAAGTAA",
        "ATGCTGGAACTGAACGTGCTGGGCATCTAA",
        "ATGGTGCATCTGAGCCAGGAACGTAGCTAA",
    ]


@pytest.fixture
def query_seq():
    return "ATGCTGAAGGAGCGTGTGCTGGCTAACTAA"


def test_calc_seq_weights_idempotent(ref_seq, query_seq):
    """Re-fitting on the same seq after fitting on others must produce
    identical weights. Catches state-leak regressions if the per-call
    fit ever stops being self-contained."""
    cpb = CodonPairBias(ref_seq, k_mer=2, pseudocount=1)

    w1 = cpb._calc_seq_weights(query_seq)
    _ = cpb._calc_seq_weights("ATGAAACCCGGGTTTAAGTAA")
    w2 = cpb._calc_seq_weights(query_seq)

    pd.testing.assert_series_equal(w1, w2)


def test_get_score_unaffected_by_get_weights_calls(ref_seq, query_seq):
    """get_score reads self._weights_arr set at init; intervening
    get_weights calls must not corrupt it."""
    cpb = CodonPairBias(ref_seq, k_mer=2, pseudocount=1)

    s1 = cpb.get_score(query_seq)
    _ = cpb.get_weights(query_seq)
    s2 = cpb.get_score(query_seq)

    assert s1 == s2


def test_init_weights_shape(ref_seq):
    """Sanity: model weights cover all kmers in counter.kmer_index, and
    _weights_arr is aligned to that order."""
    cpb = CodonPairBias(ref_seq, k_mer=2, pseudocount=1)

    assert cpb._weights_arr.shape == (len(cpb.counter.kmer_index),)
    assert np.all(np.isfinite(cpb._weights_arr))
