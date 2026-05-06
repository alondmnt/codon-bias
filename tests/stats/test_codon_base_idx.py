"""Contract tests for CodonCounter.codon_base_idx (and its k-mer extension).

The LUT is shared by ENC and RCB to compute per-codon background
compositions. Lock in shape, dtype, ordering. ``codon_base_idx_kmer``
is the lazy k-mer extension consumed by ENC k_mer>1.
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


@pytest.mark.parametrize("k_mer", [1, 2])
def test_codon_base_idx_defined_for_any_kmer(k_mer):
    """``codon_base_idx`` is per-codon and shape-stable across k_mer."""
    counter = CodonCounter(k_mer=k_mer)
    assert counter.codon_base_idx.shape == (len(counter.codon_index), 3)


@pytest.mark.parametrize("k_mer", [1, 2])
def test_codon_base_idx_kmer_shape(k_mer):
    counter = CodonCounter(k_mer=k_mer)
    n_codons = len(counter.codon_index)
    assert counter.codon_base_idx_kmer.shape == (n_codons**k_mer, 3 * k_mer)


def test_codon_base_idx_kmer_matches_kmer_index():
    """Each row of the k-mer LUT must encode the bases of the corresponding
    k-mer in ``kmer_index`` order."""
    counter = CodonCounter(k_mer=2)
    base_lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    lut = counter.codon_base_idx_kmer
    for i, kmer in enumerate(counter.kmer_index):
        assert tuple(lut[i].tolist()) == tuple(base_lut[b] for b in kmer)
