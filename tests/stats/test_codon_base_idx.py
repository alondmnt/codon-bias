"""Contract tests for CodonCounter.codon_base_idx.

The LUT is shared by ENC and RCB to compute per-codon background
compositions. Lock in shape, dtype, ordering, and the k_mer=1 guard.
"""

import numpy as np
import pytest

from codonbias.stats import CodonCounter


def test_codon_base_idx_shape_and_dtype():
    counter = CodonCounter()  # genetic_code=1, ignore_stop=True
    assert counter.codon_base_idx.shape == (len(counter.codon_index), 3)
    assert counter.codon_base_idx.dtype == np.int8


@pytest.mark.parametrize("ignore_stop", [True, False])
def test_codon_base_idx_matches_codon_index(ignore_stop):
    """Row i must encode the bases of `codon_index[i]` in ACGT order."""
    counter = CodonCounter(ignore_stop=ignore_stop)
    expected_lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, codon in enumerate(counter.codon_index):
        assert tuple(counter.codon_base_idx[i].tolist()) == tuple(
            expected_lut[b] for b in codon
        )


@pytest.mark.parametrize("genetic_code", [2, 11])
def test_codon_base_idx_non_standard_genetic_codes(genetic_code):
    counter = CodonCounter(genetic_code=genetic_code, ignore_stop=True)
    assert counter.codon_base_idx.shape == (len(counter.codon_index), 3)


def test_codon_base_idx_undefined_for_kmer_gt_1():
    """k_mer>1 paths don't currently need this LUT; attribute must be absent
    so accidental reach-through fails loudly rather than silently."""
    counter = CodonCounter(k_mer=2)
    assert not hasattr(counter, "codon_base_idx")
