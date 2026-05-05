"""Equivalence tests for BaseCounter.count_array.

Compares the vectorised k_mer=1 implementation against an inline reference
that reproduces the pre-vectorisation Python loop. The k_mer>1 path uses
Counter directly via the existing `count()` API and is covered separately.

Mirrors the structure of tests/stats/test_count_array.py.
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from codonbias.stats import BaseCounter


def _reference_count_array(counter, seq):
    """Pre-vectorisation implementation, inlined here as the reference."""
    if not isinstance(seq, str):
        raise ValueError(f"sequence is not a string: {type(seq)}")
    seq = seq.upper().replace("U", "T")
    last_pos = len(seq) - counter.k_mer + 1
    return pd.Series(
        Counter(
            [
                seq[i : i + counter.k_mer]
                for i in range(counter.frame - 1, last_pos, counter.step)
            ]
        ),
        dtype=int,
    )


def _canonical(counter, out):
    """Normalise a count output to a full-alphabet int Series for comparison."""
    idx = counter._init_table()
    if isinstance(out, np.ndarray):
        return pd.Series(out, index=idx).astype(int)
    return out.reindex(idx).fillna(0).astype(int)


SEQS = {
    "normal": "ATGAAACCCGGGTTTTAA",
    "lowercase": "atgaaacccgggttttaa",
    "mixed_case": "AtGaAaCcCgGgTtTtAa",
    "rna": "AUGAAACCCGGGUUUUAA",
    "with_N": "ATGNNNCCCGGGTAA",
    "ambiguous_RYMKSW": "ATGRYMKSWCCCGGGTAA",
    "whitespace_midseq": "AC GTTAAACCC",
    "only_one_base": "A",
    "short_two": "AC",
    "empty": "",
    "all_invalid": "NNNNNNNNN",
    "long_mixed": "ACGT" * 50 + "NN" + "CGTA" * 50,
}


@pytest.mark.parametrize("name,seq", list(SEQS.items()))
@pytest.mark.parametrize("step", [1, 3])
@pytest.mark.parametrize("frame", [1, 2, 3])
def test_count_array_kmer1_equivalence(name, seq, step, frame):
    counter = BaseCounter(k_mer=1, step=step, frame=frame)
    new = _canonical(counter, counter.count_array(seq))
    ref = _canonical(counter, _reference_count_array(counter, seq))
    pd.testing.assert_series_equal(new, ref, check_names=False)


def test_count_array_rejects_non_string():
    counter = BaseCounter()
    with pytest.raises(ValueError, match="sequence is not a string"):
        counter.count_array(12345)


def test_count_array_rejects_non_kmer_1():
    counter = BaseCounter(k_mer=2)
    with pytest.raises(NotImplementedError, match="k_mer=1"):
        counter.count_array("ACGTACGT")


def test_count_array_does_not_mutate_state():
    """count_array is stateless; self.counts must remain unset."""
    counter = BaseCounter()
    counter.count_array("ACGTACGT")
    assert not hasattr(counter, "counts")


def test_count_array_return_shape_and_dtype():
    """Contract: ndarray of shape (4,) in ACGT order."""
    out = BaseCounter().count_array("ACGTACGT")
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)


def test_count_kmer2_fallback_unchanged():
    """k_mer>1 still uses the Counter fallback — spot-check end-to-end."""
    seq = "ACGTACGTACGT"
    for step, frame in [(1, 1), (2, 1), (1, 2)]:
        counter = BaseCounter(k_mer=2, step=step, frame=frame)
        got = counter.count(seq).counts.astype(int)
        ref_raw = _reference_count_array(counter, seq)
        expected = ref_raw.reindex(counter._init_table()).fillna(0).astype(int)
        pd.testing.assert_series_equal(got, expected, check_names=False)


def test_count_end_to_end_gc3_example():
    """Docstring example: GC3 content via step=3, frame=3 must stay correct."""
    seq = "ATGCGCATGCGCATGCGC"  # third positions: G, C, G, C, G, C → GC3=1.0
    freq = BaseCounter(step=3, frame=3).count(seq).get_table(normed=True, pseudocount=0)
    assert pytest.approx(freq["G"] + freq["C"], rel=1e-12) == 1.0


def test_count_multi_seq_sum():
    """Multi-sequence sum_seqs path must agree with the reference."""
    seqs = ["ATGAAA", "CCCGGG", "NNN", ""]
    counter = BaseCounter(k_mer=1, step=1, frame=1)
    got = counter.count(seqs).counts
    # Sum across all sequences: A=4 (ATGAAA), C=3 (CCCGGG), G=4 (1+3), T=1.
    expected = (
        pd.Series({"A": 4, "C": 3, "G": 4, "T": 1}, dtype=float)
        .reindex(counter._init_table())
        .fillna(0)
    )
    pd.testing.assert_series_equal(got.astype(float), expected, check_names=False)
