"""Permuter scope dispatch and deprecation-shim coverage.

Verifies that:
- Each scope maps to the correct internal `add_properties`.
- Unknown scope values raise.
- The deprecated `IntraSeqPermuter` / `IntraPosPermuter` shims emit
  `FutureWarning` and produce output identical to the new API under
  the same `random_state`.
"""

import warnings

import pandas as pd
import pytest

from codonbias.random import IntraPosPermuter, IntraSeqPermuter, Permuter


@pytest.fixture
def seqs():
    """Multi-synonym Ala/Lys sequences (matches test_permuter.py fixture)."""
    return [
        "ATG" + "GCTGCCGCAGCG" * 3 + "AAAAAGAAAAAG" * 2,
        "ATG" + "GCCGCAGCGGCT" * 3 + "AAGAAAAAGAAA" * 2,
        "ATG" + "GCAGCGGCTGCC" * 3 + "AAAAAGAAGAAA" * 2,
    ]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("inter_seq", []),
        ("intra_seq", ["id"]),
        ("intra_pos", ["pos"]),
    ],
)
def test_scope_maps_to_add_properties(scope, expected):
    p = Permuter(scope=scope, n_samples=1, random_state=42, n_jobs=1)
    assert p.scope == scope
    assert p.add_properties == expected


def test_default_scope_is_inter_seq():
    p = Permuter(n_samples=1, random_state=42, n_jobs=1)
    assert p.scope == "inter_seq"
    assert p.add_properties == []


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="scope must be one of"):
        Permuter(scope="cross_pos", n_samples=1, random_state=42, n_jobs=1)


@pytest.mark.parametrize(
    ("shim_cls", "scope"),
    [
        (IntraSeqPermuter, "intra_seq"),
        (IntraPosPermuter, "intra_pos"),
    ],
)
def test_shim_emits_future_warning_and_matches_new_api(seqs, shim_cls, scope):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim_out = shim_cls(n_samples=3, random_state=42, n_jobs=1).get_permuted_seq(
            seqs
        )

    assert any(issubclass(w.category, FutureWarning) for w in caught), (
        f"{shim_cls.__name__} did not emit FutureWarning"
    )

    new_out = Permuter(
        scope=scope, n_samples=3, random_state=42, n_jobs=1
    ).get_permuted_seq(seqs)

    pd.testing.assert_frame_equal(shim_out, new_out)
