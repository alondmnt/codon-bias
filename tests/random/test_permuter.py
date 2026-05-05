import numpy as np
import pandas as pd
import pytest

from codonbias.random import Permuter


@pytest.fixture
def seqs():
    """Sequences with multiple synonymous codons per amino acid so that
    shuffling produces visibly different output. Each seq uses Ala (4
    synonyms: GCT/GCC/GCA/GCG) and Lys (2 synonyms: AAA/AAG) repeatedly,
    giving the permuter real degrees of freedom within each AA group."""
    return [
        "ATG" + "GCTGCCGCAGCG" * 3 + "AAAAAGAAAAAG" * 2,
        "ATG" + "GCCGCAGCGGCT" * 3 + "AAGAAAAAGAAA" * 2,
        "ATG" + "GCAGCGGCTGCC" * 3 + "AAAAAGAAGAAA" * 2,
    ]


def test_permuter_reproducibility(seqs):
    """Same ``random_state`` on the same input must produce bit-exact output."""
    p1 = Permuter(scope="intra_seq", n_samples=5, random_state=42, n_jobs=1)
    p2 = Permuter(scope="intra_seq", n_samples=5, random_state=42, n_jobs=1)
    out1 = p1.get_permuted_seq(seqs)
    out2 = p2.get_permuted_seq(seqs)

    pd.testing.assert_frame_equal(out1, out2)


def test_permuter_independent_groups(seqs):
    """Two equal-sized AA-groups must get decorrelated permutation streams.

    Pre-fix, ``np.random.seed(random_state)`` at the top of ``_permute_col``
    made every group's first permutation identical when group lengths
    matched. Post-fix, ``_group_idx`` derives an independent PCG64 stream
    per group, so the first `null_0` draw across groups must differ.
    """
    p = Permuter(n_samples=3, random_state=42, n_jobs=1)
    df, prop_cols = p._preprocess_df(seqs)
    permuted = p._permute_df(df, prop_cols, "seq", p.n_samples)

    # Each AA-group of the same size should permute independently.
    # Collect `null_0` per group and confirm at least one group differs
    # from the first group's permutation (decorrelation signal).
    first_perms = permuted.groupby(prop_cols)["null_0"].apply(tuple)
    distinct = first_perms.nunique()
    assert distinct > 1, (
        "expected per-group `null_0` permutations to differ; got identical "
        "streams across groups (cross-group correlation regression)"
    )


def test_permuter_changes_with_random_state(seqs):
    """Different ``random_state`` values must produce different output."""
    p1 = Permuter(scope="intra_seq", n_samples=3, random_state=42, n_jobs=1)
    p2 = Permuter(scope="intra_seq", n_samples=3, random_state=7, n_jobs=1)
    out1 = p1.get_permuted_seq(seqs)
    out2 = p2.get_permuted_seq(seqs)

    assert not out1.equals(out2)


def test_permuter_no_global_state_pollution(seqs):
    """Running the permuter must not re-seed NumPy's global RNG."""
    np.random.seed(123)
    before = np.random.random()

    np.random.seed(123)
    Permuter(
        scope="intra_seq", n_samples=3, random_state=42, n_jobs=1
    ).get_permuted_seq(seqs)
    after = np.random.random()

    assert before == after, (
        "Permuter mutated NumPy's global RNG state (NPY002 regression)"
    )


def test_permuter_smoke_shape(seqs):
    """Smoke: output has one row per input sequence, `n_samples` columns."""
    n_samples = 7
    p = Permuter(scope="intra_seq", n_samples=n_samples, random_state=42, n_jobs=1)
    out = p.get_permuted_seq(seqs)

    assert len(out) == len(seqs)
    assert out.shape[1] == n_samples
    # Each permuted sequence must be the same length as its source
    for i, seq in enumerate(seqs):
        for j in range(n_samples):
            assert len(out.iloc[i, j]) == len(seq)
