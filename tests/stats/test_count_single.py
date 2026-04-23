"""Equivalence tests for CodonCounter._count_single.

Compares the vectorised implementation against an inline reference that
reproduces the pre-vectorisation Python loop. Covers edge cases around
case, U→T, ambiguous bases, non-ACGT characters mid-sequence, short
sequences, STOP handling, and non-default genetic codes.
"""

import numpy as np
import pytest

from codonbias.stats import CodonCounter


def _reference_count_single(counter, seq):
    """Pre-vectorisation implementation, inlined here as the reference."""
    if not isinstance(seq, str):
        raise ValueError(f"sequence is not a string: {type(seq)}")
    seq = seq.upper().replace("U", "T")
    counts = np.zeros(len(counter._codon_to_idx), dtype=float)
    for i in range(0, len(seq) - 2, 3):
        idx = counter._codon_to_idx.get(seq[i : i + 3])
        if idx is not None:
            counts[idx] += 1
    aa_counts = np.bincount(counter._aa_group, weights=counts, minlength=counter._n_aa)
    return counts, aa_counts


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
def test_count_single_equivalence_genetic_code_1(name, seq, ignore_stop):
    counter = CodonCounter(ignore_stop=ignore_stop)

    new_counts, new_aa = counter._count_single(seq)
    ref_counts, ref_aa = _reference_count_single(counter, seq)

    np.testing.assert_array_equal(
        new_counts, ref_counts, err_msg=f"counts mismatch on {name}"
    )
    np.testing.assert_array_equal(
        new_aa, ref_aa, err_msg=f"aa_counts mismatch on {name}"
    )


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_count_single_equivalence_other_genetic_codes(genetic_code):
    """Non-standard genetic codes have different STOP sets — confirm the
    LUT population respects the code-specific _idx_to_codon."""
    counter = CodonCounter(genetic_code=genetic_code, ignore_stop=True)
    seq = "ATGAAACCCGGGTTTTGATAATAGCTG"  # includes TGA / TAA / TAG
    new_counts, new_aa = counter._count_single(seq)
    ref_counts, ref_aa = _reference_count_single(counter, seq)
    np.testing.assert_array_equal(new_counts, ref_counts)
    np.testing.assert_array_equal(new_aa, ref_aa)


def test_count_single_rejects_non_string():
    counter = CodonCounter()
    with pytest.raises(ValueError, match="sequence is not a string"):
        counter._count_single(12345)


def test_count_single_return_shape_and_dtype():
    """Contract: (counts, aa_counts) tuple, float arrays, correct shapes."""
    counter = CodonCounter()  # default: genetic_code=1, ignore_stop=True
    counts, aa_counts = counter._count_single("ATGAAACCCGGG")
    assert isinstance(counts, np.ndarray)
    assert isinstance(aa_counts, np.ndarray)
    assert counts.dtype.kind == "f"
    assert aa_counts.dtype.kind == "f"
    assert counts.shape == (len(counter._idx_to_codon),)
    assert aa_counts.shape == (counter._n_aa,)
