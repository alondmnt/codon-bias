"""Equivalence tests for CodonCounter.count_array.

Compares the vectorised implementation against an inline reference that
reproduces the pre-vectorisation Python loop. Covers edge cases around
case, U→T, ambiguous bases, non-ACGT characters mid-sequence, short
sequences, STOP handling, and non-default genetic codes.
"""

import numpy as np
import pytest

from codonbias.stats import CodonCounter


def _reference_count_array(counter, seq):
    """Pre-vectorisation implementation, inlined here as the reference."""
    if not isinstance(seq, str):
        raise ValueError(f"sequence is not a string: {type(seq)}")
    seq = seq.upper().replace("U", "T")
    codon_to_idx = {c: i for i, c in enumerate(counter.codon_index)}
    counts = np.zeros(len(codon_to_idx), dtype=float)
    for i in range(0, len(seq) - 2, 3):
        idx = codon_to_idx.get(seq[i : i + 3])
        if idx is not None:
            counts[idx] += 1
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
def test_count_array_equivalence_genetic_code_1(name, seq, ignore_stop):
    counter = CodonCounter(ignore_stop=ignore_stop)

    new_counts = counter.count_array(seq)
    ref_counts = _reference_count_array(counter, seq)

    np.testing.assert_array_equal(
        new_counts, ref_counts, err_msg=f"counts mismatch on {name}"
    )


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_count_array_equivalence_other_genetic_codes(genetic_code):
    """Non-standard genetic codes have different STOP sets — confirm the
    LUT population respects the code-specific codon_index."""
    counter = CodonCounter(genetic_code=genetic_code, ignore_stop=True)
    seq = "ATGAAACCCGGGTTTTGATAATAGCTG"  # includes TGA / TAA / TAG
    new_counts = counter.count_array(seq)
    ref_counts = _reference_count_array(counter, seq)
    np.testing.assert_array_equal(new_counts, ref_counts)


def test_count_array_rejects_non_string():
    counter = CodonCounter()
    with pytest.raises(ValueError, match="sequence is not a string"):
        counter.count_array(12345)


def test_count_array_rejects_non_kmer_1():
    counter = CodonCounter(k_mer=2)
    with pytest.raises(NotImplementedError, match="k_mer=1"):
        counter.count_array("ATGAAA")


def test_count_array_does_not_mutate_state():
    """count_array is stateless; self.counts must remain unset."""
    counter = CodonCounter()
    counter.count_array("ATGAAACCCGGG")
    assert not hasattr(counter, "counts")


def test_count_array_return_shape_and_dtype():
    """Contract: float ndarray of shape (len(codon_index),)."""
    counter = CodonCounter()  # default: genetic_code=1, ignore_stop=True
    counts = counter.count_array("ATGAAACCCGGG")
    assert isinstance(counts, np.ndarray)
    assert counts.dtype.kind == "f"
    assert counts.shape == (len(counter.codon_index),)
