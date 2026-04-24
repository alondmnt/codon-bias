"""Equivalence tests for BaseCounter._count_single.

Compares the vectorised k_mer=1 implementation against an inline reference
that reproduces the pre-vectorisation Python loop. The k_mer>1 path falls
back to the same Counter-based code, so it's covered implicitly.

Mirrors the structure of tests/stats/test_count_single.py.
"""

from collections import Counter

import pandas as pd
import pytest

from codonbias.stats import BaseCounter


def _reference_count_single(counter, seq):
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


def _canonical(counter, series):
    """Normalise a raw _count_single Series to the full base alphabet.

    BaseCounter.count() itself performs `.reindex(self._init_table()).fillna(0)`
    downstream, so comparing the canonical form is the contract that matters.
    """
    return series.reindex(counter._init_table()).fillna(0).astype(int)


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
def test_count_single_kmer1_equivalence(name, seq, step, frame):
    counter = BaseCounter(k_mer=1, step=step, frame=frame)
    new = _canonical(counter, counter._count_single(seq))
    ref = _canonical(counter, _reference_count_single(counter, seq))
    pd.testing.assert_series_equal(new, ref, check_names=False)


def test_count_single_rejects_non_string():
    counter = BaseCounter()
    with pytest.raises(ValueError, match="sequence is not a string"):
        counter._count_single(12345)


def test_count_single_return_is_series():
    """Contract: _count_single returns a pd.Series (k_mer=1 and k_mer>1)."""
    for k in (1, 2):
        counter = BaseCounter(k_mer=k)
        out = counter._count_single("ACGTACGT")
        assert isinstance(out, pd.Series)


def test_count_kmer2_fallback_unchanged():
    """k_mer>1 still uses the Counter fallback — spot-check end-to-end."""
    seq = "ACGTACGTACGT"
    for step, frame in [(1, 1), (2, 1), (1, 2)]:
        counter = BaseCounter(k_mer=2, step=step, frame=frame)
        got = counter.count(seq).counts.astype(int)
        ref_raw = _reference_count_single(counter, seq)
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
