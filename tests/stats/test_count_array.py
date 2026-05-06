"""Equivalence tests for CodonCounter.count_array.

Compares the vectorised implementation against an inline reference that
reproduces the pre-vectorisation Python loop. Covers edge cases around
case, U→T, ambiguous bases, non-ACGT characters mid-sequence, short
sequences, STOP handling, and non-default genetic codes. Generalised
to k_mer in [1, 3] once the dense bincount path was extended.
"""

import numpy as np
import pytest

from codonbias.stats import CodonCounter


def _reference_count_array(counter, seq):
    """Reference: codon-id sliding window over the python-level codon list.

    Lex-product order mirrors ``count_array``: a k-mer with codon indices
    ``(i0, i1, ...)`` lands at ``sum(i_j * n_codons ** (k-1-j))``.
    """
    if not isinstance(seq, str):
        raise ValueError(f"sequence is not a string: {type(seq)}")
    seq = seq.upper().replace("U", "T")
    codon_to_idx = {c: i for i, c in enumerate(counter.codon_index)}
    n_codons = len(codon_to_idx)
    k = counter.k_mer
    n_out = n_codons**k
    counts = np.zeros(n_out, dtype=float)
    span = 3 * k
    for i in range(0, len(seq) - span + 1, 3):
        idxs = [codon_to_idx.get(seq[i + 3 * j : i + 3 * (j + 1)]) for j in range(k)]
        if any(idx is None for idx in idxs):
            continue
        bucket = 0
        for j, idx in enumerate(idxs):
            bucket += idx * n_codons ** (k - 1 - j)
        counts[bucket] += 1
    return counts


SEQS = {
    "normal": "ATGAAACCCGGGTTTTAA",
    "lowercase": "atgaaacccgggttttaa",
    "mixed_case": "AtGaAaCcCgGgTtTtAa",
    "rna": "AUGAAACCCGGGUUUUAA",
    "with_N": "ATGNNNCCCGGGTAA",
    "ambiguous_RYMKSW": "ATGRYMKSWCCCGGGTAA",
    # Regression for the base-4 sentinel collision: "AC " must NOT count
    # as AGA (which would happen with naive b0*16 + b1*4 + b2 packing).
    "whitespace_midseq": "AC GTTAAACCC",
    "whitespace_collides_with_AGA": "AC ",
    "trailing_one": "ATGAAACC",
    "trailing_two": "ATGAAAC",
    "empty": "",
    "one_codon": "ATG",
    "two_codons": "ATGAAA",
    "stop_only": "TAATAGTGA",
    "all_invalid": "NNNNNNNNN",
}


@pytest.mark.parametrize("name,seq", list(SEQS.items()))
@pytest.mark.parametrize("ignore_stop", [True, False])
@pytest.mark.parametrize("k_mer", [1, 2, 3])
def test_count_array_equivalence_genetic_code_1(name, seq, ignore_stop, k_mer):
    counter = CodonCounter(ignore_stop=ignore_stop, k_mer=k_mer)

    new_counts = counter.count_array(seq)
    ref_counts = _reference_count_array(counter, seq)

    np.testing.assert_array_equal(
        new_counts, ref_counts, err_msg=f"counts mismatch on {name} (k_mer={k_mer})"
    )


@pytest.mark.parametrize("genetic_code", [2, 11])
@pytest.mark.parametrize("k_mer", [1, 2])
def test_count_array_equivalence_other_genetic_codes(genetic_code, k_mer):
    """Non-standard genetic codes have different STOP sets — confirm the
    LUT population respects the code-specific codon_index."""
    counter = CodonCounter(genetic_code=genetic_code, ignore_stop=True, k_mer=k_mer)
    seq = "ATGAAACCCGGGTTTTGATAATAGCTG"  # includes TGA / TAA / TAG
    new_counts = counter.count_array(seq)
    ref_counts = _reference_count_array(counter, seq)
    np.testing.assert_array_equal(new_counts, ref_counts)


def test_count_array_rejects_non_string():
    counter = CodonCounter()
    with pytest.raises(ValueError, match="sequence is not a string"):
        counter.count_array(12345)


def test_count_array_rejects_kmer_above_3():
    counter = CodonCounter(k_mer=4)
    with pytest.raises(NotImplementedError, match="k_mer <= 3"):
        counter.count_array("ATGAAA" * 4)


@pytest.mark.parametrize("k_mer", [1, 2, 3])
def test_count_array_does_not_mutate_state(k_mer):
    """count_array is stateless; self.counts must remain unset."""
    counter = CodonCounter(k_mer=k_mer)
    counter.count_array("ATGAAACCCGGGTTTTAA" * 2)
    assert not hasattr(counter, "counts")


@pytest.mark.parametrize("k_mer", [1, 2, 3])
def test_count_array_return_shape_and_dtype(k_mer):
    """Contract: float ndarray of shape (len(codon_index) ** k_mer,)."""
    counter = CodonCounter(k_mer=k_mer)  # default: genetic_code=1, ignore_stop=True
    counts = counter.count_array("ATGAAACCCGGG" * k_mer)
    assert isinstance(counts, np.ndarray)
    assert counts.dtype.kind == "f"
    assert counts.shape == (len(counter.codon_index) ** k_mer,)
